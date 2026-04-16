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
        self.fx = 0
        self.fy = 0
        self.cx = 0
        self.cy = 0

        self.prompts = ["red tape", ["robot arm", "black object"], "red cup", "green bottle"]
        # self.prompts = ["red tape", ["robot arm", "black object"], "tennis ball", "cup"]

        self.centroids = [np.array([0, 0, 0]) for prompt in self.prompts]
        self.masks = [[None] for prompt in self.prompts]

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
        if self.fx == 0:
            return

        # 1 — convert PointCloud2 → XYZ NumPy
        xyz, rgb_stacked = self.pc2_to_xyz_ros_numpy(pc_msg)

        # 2 — camera intrinsics
        K = np.array([
            [self.fx, 0,        self.cx],
            [0,        self.fy, self.cy],
            [0,        0,          1]
        ], dtype=np.float32)

        new_centroids = []
        selected_masks = []

        # Red ring
        ring_mask = self.masks[0][0]  # guaranteed to exist

        if ring_mask is None or len(ring_mask) == 0:
            ring_centroid = np.array(self.centroids[0])
            selected_masks.append(None)
        else:
            ring_xyz = self.mask_pointcloud_with_mask(xyz, None, ring_mask, K)

            if ring_xyz.shape[0] == 0:
                ring_centroid = np.array(self.centroids[0])
                selected_masks.append(None)
            else:
                ring_centroid = np.mean(ring_xyz, axis=0)
                selected_masks.append(ring_mask)

        new_centroids.append(ring_centroid)

        if len(self.centroids) < 4:
            for i in range(4 - len(self.centroids)):
                self.centroids.append(np.array([0, 0, 0]))

        # --- 3A — End Effector (prompt 0) ---
        ee_mask = self.masks[1][0]  # guaranteed to exist

        if ee_mask is None or len(ee_mask) == 0:
            ee_centroid = np.array(self.centroids[0])
            selected_masks.append(None)
        else:
            ee_xyz = self.mask_pointcloud_with_mask(xyz, None, ee_mask, K)

            if ee_xyz.shape[0] == 0:
                ee_centroid = np.array(self.centroids[0])
                selected_masks.append(None)
            else:
                ee_centroid = np.mean(ee_xyz, axis=0)
                selected_masks.append(ee_mask)

        new_centroids.append(ee_centroid)

        # --- 3B — Other prompts ---
        i = 2
        for prompt_masks in self.masks[2:]:
            if prompt_masks is None or len(prompt_masks) == 0:
                new_centroids.append(self.centroids[i])
                selected_masks.append(None)
                continue

            best_mask = None
            best_centroid = None
            best_dist = float("inf")

            for mask in prompt_masks:
                if mask is None:
                    continue

                masked_xyz = self.mask_pointcloud_with_mask(xyz, None, mask, K)

                if masked_xyz.shape[0] == 0:
                    continue

                centroid = np.mean(masked_xyz, axis=0)

                dist = np.linalg.norm(centroid - ee_centroid)

                if dist < best_dist:
                    best_dist = dist
                    best_mask = mask
                    best_centroid = centroid

            if best_mask is None:
                new_centroids.append(self.centroids[i])
                selected_masks.append(None)
            else:
                new_centroids.append(best_centroid)
                selected_masks.append(best_mask)
            
            i += 1

        self.centroids = new_centroids

        # --- 4 — Merge selected masks ---
        merged_mask = None
        for i, mask in enumerate(selected_masks):
            if mask is None:
                continue

            if merged_mask is None:
                merged_mask = mask.copy()
            else:
                merged_mask = np.logical_or(merged_mask, mask)

        if merged_mask is None:
            # no valid masks at all → empty point cloud
            return
        else:
            merged_mask = merged_mask.astype(np.uint8)

            ros_image_msg = self.bridge.cv2_to_imgmsg((merged_mask * 255).astype(np.uint8), encoding="mono8")
            self.seg_pub.publish(ros_image_msg)

            # 5 — mask point cloud
            masked_xyz, masked_rgb = self.mask_pointcloud_with_mask(
                xyz, rgb_stacked, merged_mask, K
            )

        # 6 — convert back to PointCloud2
        masked_pc_msg = self.xyz_rgb_to_pc2(
            masked_xyz, masked_rgb, frame_id=pc_msg.header.frame_id
        )

        # 7 — publish masked cloud
        self.seg_point_pub.publish(masked_pc_msg)

        # 8 — publish centroids
        msg = Float32MultiArrayStamped()
        msg.data = np.array(self.centroids).flatten().tolist()
        msg.header.stamp = rospy.Time.now()
        self.centroids_pub.publish(msg)

    
    def callback(self, img_msg):
        self.img = img_msg

    def process_image(self):
        if self.img is None:
            rospy.loginfo("No image yet")
            self.busy = False
            return

        try: 
            start_seg = time.time()
            # Convert ROS Image -> OpenCV
            cv_img = self.bridge.imgmsg_to_cv2(self.img, "bgr8")
            success, img_encoded = cv2.imencode('.jpg', cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not success:
                raise ValueError("JPEG compression failed")
            img_bytes = img_encoded.tobytes()

            h, w = cv_img.shape[:2]
            header = {"height": h, "width": w, "format": "jpg"}

            prompts_json = json.dumps({"prompts": self.prompts}).encode("utf-8")

            # rospy.loginfo("Sending...")
            self.socket.send_multipart([
                json.dumps(header).encode("utf-8"),
                img_bytes,
                prompts_json
            ])

            reply_parts = self.socket.recv_multipart()
            reply_header = json.loads(reply_parts[0].decode("utf-8"))

            part_idx = 1  # starts after header

            self.masks = []

            for prompt in self.prompts:
                # --- FIX: Match the server's key generation logic ---
                prompt_key = "|".join(prompt) if isinstance(prompt, list) else str(prompt)
                
                # Fetch the metadata using the new key format
                prompt_masks_meta = reply_header.get(prompt_key, [])

                # No masks returned for this prompt
                if not prompt_masks_meta:
                    rospy.loginfo(f"No mask for prompt {prompt}")
                    self.masks.append([None])
                    continue

                prompt_masks = []
                for meta in prompt_masks_meta:
                    packed_bytes = reply_parts[part_idx]
                    part_idx += 1 

                    # Unpack bits, slice off any padding, and reshape
                    packed_array = np.frombuffer(packed_bytes, dtype=np.uint8)
                    unpacked = np.unpackbits(packed_array)[:h * w]
                    mask_array = unpacked.reshape(h, w).astype(np.uint8) * 255

                    prompt_masks.append(mask_array)

                self.masks.append(prompt_masks)
           
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