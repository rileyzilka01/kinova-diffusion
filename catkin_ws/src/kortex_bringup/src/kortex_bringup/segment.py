#!/usr/bin/python3

import rospy
import zmq
import json
import numpy as np
import message_filters
from sensor_msgs.msg import PointCloud2, JointState, Image, CameraInfo, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension, Int16, Header, Float32MultiArray
from kortex_bringup.msg import Float32MultiArrayStamped
import sys
import ros_numpy
import msgpack
import cv2
from cv_bridge import CvBridge

import time
import math
import threading
from queue import Queue, Empty


class ImageSegmentationNode:
    def __init__(self):
        rospy.init_node("image_segmentation_node")

        # ZeroMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:4444")
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)

        self.socket.setsockopt(zmq.RCVTIMEO, -1)  # indefinite wait
        self.socket.setsockopt(zmq.LINGER, 0)       # Don't hang on close if server is dead

        self.bridge = CvBridge()

        self.img = None
        self.busy = False
        self.mask_arrays = []
        self.merged_mask = None
        self.centroids = []
        self.fx = 0
        self.fy = 0
        self.cx = 0
        self.cy = 0

        self.prompts = [["robot arm", "black object"], "yellow object", "cup"]

        # Send a ping test message
        try:
            ping = {"ping": True}
            self.socket.send_json(ping)

            resp = self.socket.recv_json()
            rospy.loginfo(f"✅ Connected to inference server: {resp}")
        except zmq.Again:
            rospy.logerr("❌ Could not connect to inference server (timeout)")
            sys.exit(1)

        # ROS publishers and subscribers
        self.cam_sub = rospy.Subscriber("/cam/color/image_raw", Image, self.callback)
        self.pc_sub = rospy.Subscriber('/cam/depth/color/points', PointCloud2, self.pc_callback)
        self.intrinsics_sub = rospy.Subscriber('/cam/color/camera_info', CameraInfo, self.cam_info_callback)

        self.seg_pub = rospy.Publisher("/my_gen3/segment_mask", Image, queue_size=1)
        self.seg_point_pub = rospy.Publisher("/my_gen3/segment_pc_mask", PointCloud2, queue_size=1)
        self.centroids_pub = rospy.Publisher("/my_gen3/pc_centroids", Float32MultiArrayStamped, queue_size=1)

        rospy.loginfo("🤖 Segmentation node initialized")

    def cam_info_callback(self, msg):
        K = np.array(msg.K).reshape(3,3)
        self.fx = K[0,0]
        self.fy = K[1,1]
        self.cx = K[0,2]
        self.cy = K[1,2]

        self.K = K
        self.camera_info_received = True

    def pc_callback(self, pc_msg):
        if self.merged_mask is None or self.fx == 0:
            return  # wait for mask image

        # 1 — convert PointCloud2 → XYZ NumPy
        xyz, rgb_stacked = self.pc2_to_xyz_ros_numpy(pc_msg)

        # 2 — camera intrinsics
        K = np.array([
            [self.fx, 0,        self.cx],
            [0,        self.fy, self.cy],
            [0,        0,          1]
        ], dtype=np.float32)

        # 3 — mask the point cloud and rgb
        masked_xyz, masked_rgb = self.mask_pointcloud_with_mask(xyz, rgb_stacked, self.merged_mask, K)

        # 4 — convert back to PointCloud2
        masked_pc_msg = self.xyz_rgb_to_pc2(masked_xyz, masked_rgb, frame_id=pc_msg.header.frame_id)

        # 5 — publish
        self.seg_point_pub.publish(masked_pc_msg)

        # rospy.loginfo("Published masked point cloud with %d points", masked_xyz.shape[0])

        # 6 - get the centroids
        self.centroids = []
        # rospy.loginfo(f"Len of masked arrays: {len(self.mask_arrays)}")
        for i in self.mask_arrays:
            if i is None: 
                self.centroids.append([0, 0, 0])
            else:
                masked_xyz = self.mask_pointcloud_with_mask(xyz, None, i, K)
                if masked_xyz.shape[0] == 0:
                    rospy.loginfo("No points found for this mask")
                    continue
                centroid = np.mean(masked_xyz, axis=0)
                self.centroids.append(centroid)
                # rospy.loginfo(f"Centroid of masked points: x={centroid[0]:.3f}, y={centroid[1]:.3f}, z={centroid[2]:.3f}")
        msg = Float32MultiArrayStamped()
        msg.data = np.array(self.centroids).flatten().tolist()
        msg.header.stamp = rospy.Time.now()
        self.centroids_pub.publish(msg)

    def callback(self, img_msg):
        self.img = img_msg
        # rospy.loginfo("Received image")

    def process_image(self):
        if self.img is None:
            rospy.loginfo("No image yet")
            self.busy = False
            return

        try: 
            start_seg = time.time()
            # Convert ROS Image -> OpenCV
            cv_img = self.bridge.imgmsg_to_cv2(self.img, "bgr8")
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, c = img_rgb.shape
            img_bytes = img_rgb.tobytes()

            header = {"height": h, "width": w, "channels": c}

            prompts_json = json.dumps({"prompts": self.prompts}).encode("utf-8")

            # rospy.loginfo("Sending...")
            self.socket.send_multipart([
                json.dumps(header).encode("utf-8"),
                img_bytes,
                prompts_json
            ])

            reply_parts = self.socket.recv_multipart()
            reply_header = json.loads(reply_parts[0].decode("utf-8"))
            mask_bytes = reply_parts[1]

            # rospy.loginfo(f"Server reply header: {reply_header}")

            tmp = []
            merged_mask = None
            self.mask_arrays = []
            for i, prompt in enumerate(reply_header.keys()):
                if reply_header[prompt]["height"] != 0:
                    mask_bytes = reply_parts[i+1]

                    # rospy.loginfo(f"Mask bytes length: {len(mask_bytes)}")

                    mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)  # or dtype=header['dtype']
                    mask_array = mask_array.reshape(header['height'], header['width'])

                    if merged_mask is None:
                        merged_mask = mask_array.copy()
                    else:
                        merged_mask = np.logical_or(merged_mask, mask_array).astype(np.uint8)
                
                    self.mask_arrays.append((mask_array * 255).astype(np.uint8))
                else:
                    self.mask_arrays.append(None)

            self.merged_mask = (merged_mask * 255).astype(np.uint8)
            ros_image_msg = self.bridge.cv2_to_imgmsg(self.merged_mask, encoding="mono8")
            self.seg_pub.publish(ros_image_msg)

           
            self.busy = False

            rospy.loginfo(f"Segmentation processing took {time.time() - start_seg:.6f} seconds")

        except Exception as e:
            rospy.logerr(f"Error during ZMQ send/recv: {e}")

    def pc2_to_xyz_ros_numpy(self, pc_msg):
        pc_np = ros_numpy.numpify(pc_msg)    # structured array
        xyz = np.stack([pc_np['x'], pc_np['y'], pc_np['z']], axis=-1)
        return xyz.astype(np.float32), pc_np['rgb'].view(np.uint32)

    def xyz_rgb_to_pc2(self, points, rgb, frame_id):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id

        if rgb.dtype != np.float32:
            rgb = rgb.astype(np.uint32).view(np.float32)

        # Build cloud array
        cloud_data = np.zeros((points.shape[0], 4), dtype=np.float32)
        cloud_data[:, 0:3] = points
        cloud_data[:, 3] = rgb

        fields = [
            PointField('x',   0,  PointField.FLOAT32, 1),
            PointField('y',   4,  PointField.FLOAT32, 1),
            PointField('z',   8,  PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.FLOAT32, 1),
        ]

        return point_cloud2.create_cloud(header, fields, cloud_data)

    def mask_pointcloud_with_mask(self, xyz, rgb, mask, K):
        # keep only points with positive depth
        # rospy.loginfo(f"XYZ shape: {xyz.shape}")
        good = xyz[:, 2] > 0
        xyz = xyz[good]

        # project to image
        pts_T = xyz.T  # (3, N)
        uv = K @ pts_T
        uv = uv[:2] / uv[2]
        uv = uv.T  # (N,2)

        u = uv[:, 0].astype(np.int32)
        v = uv[:, 1].astype(np.int32)

        H, W = mask.shape
        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)

        xyz = xyz[inside]
        u = u[inside]
        v = v[inside]

        # mask lookup
        keep = mask[v, u] > 0
        if rgb is None:
            return xyz[keep]
        else:
            return xyz[keep], rgb[keep]


if __name__ == '__main__':
    try:
        node = ImageSegmentationNode()

        # Instead of rospy.spin(), run a loop
        rate = rospy.Rate(30)  # 30 Hz loop, adjust as needed
        while not rospy.is_shutdown():
            # Grab the latest image and process it
            if node.img is not None:
                node.process_image()
            rate.sleep()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
    except rospy.ROSInterruptException:
        pass