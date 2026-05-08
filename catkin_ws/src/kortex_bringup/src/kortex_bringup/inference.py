#!/usr/bin/python3

import rospy
import zmq
import json
import numpy as np
import message_filters
from sensor_msgs.msg import PointCloud2, JointState
from kortex_bringup.msg import Float32MultiArrayStamped
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension, Int16
import sensor_msgs.point_cloud2 as pc2
import sys
import ros_numpy
import msgpack

from kortex_driver.msg._BaseCyclic_Feedback import BaseCyclic_Feedback

import time
import math
import cv2

import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

from kinova_util import KinovaUtil

from pointcloud_processing import preprocess_point_cloud

import msgpack_numpy
msgpack_numpy.patch()
msgpack_numpy_encode = msgpack_numpy.encode
msgpack_numpy_decode = msgpack_numpy.decode


class RobotInferenceNode:
    def __init__(self, model, local):
        rospy.init_node("robot_inference_node")

        self.model = model

        # ZeroMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        if not local:
            self.socket.connect("tcp://localhost:5555")
        else:
            self.socket.connect("tcp://192.168.1.161:5555")

        self.socket.setsockopt(zmq.RCVTIMEO, 1000)

        self.socket.setsockopt(zmq.RCVTIMEO, -1)  # indefinite timeout
        self.socket.setsockopt(zmq.LINGER, 0)       # Don't hang on close if server is dead

        self.ku = KinovaUtil()

        self.pointclouds = []
        self.states = []

        self.horizon = 1
        self.n_obs = 1

        self.shared = True
        self.num_prompts = 4

        self.use_pointcloud = True
        self.quat = True

        if self.model == "hitl_hgd":
            self.use_centroids = True
            self.use_norm_diffs = True
            self.use_ee_position = False
        else:
            self.use_centroids = False
            self.use_norm_diffs = False
            self.use_ee_position = True


        if self.shared:
            # SHARED
            self.use_gripper = False
            self.joint_pos = False
            self.auto = False
            # SHARED
        else:
            # AUTO
            self.use_gripper = True
            self.joint_pos = True
            self.auto = True
            # AUTO

        self.msg = Float64MultiArray()
        if self.model == "hitl_d":
            self.msg.layout = MultiArrayLayout(dim=[
                MultiArrayDimension(label="", size=self.horizon, stride=self.horizon * 3),
                MultiArrayDimension(label="", size=3, stride=1)
            ])
        else:
            self.msg.layout = MultiArrayLayout(dim=[
                MultiArrayDimension(label="", size=self.horizon, stride=self.horizon * 6),
                MultiArrayDimension(label="", size=6, stride=1)
            ])

        # Send a ping test message
        try:
            ping_msg = msgpack.packb({"ping": True}, use_bin_type=True)
            self.socket.send(ping_msg)
            
            response_msg = self.socket.recv()
            response = msgpack.unpackb(response_msg, raw=False)
            
            rospy.loginfo("✅ Connected to inference server")
        except zmq.Again:
            rospy.logerr("❌ Could not connect to inference server (timeout)")
            sys.exit(1)

        # ROS publishers and subscribers
        if self.model == "hitl_hgd":
            if self.use_pointcloud:
                self.pc_segment_sub = message_filters.Subscriber('/my_gen3/segment_pc_mask', PointCloud2)
            self.centroids_sub = message_filters.Subscriber('/my_gen3/pc_centroids', Float32MultiArrayStamped)

        if self.use_pointcloud:
            self.depth_sub = message_filters.Subscriber("/cam/depth/color/points", PointCloud2)

        self.joint_sub = message_filters.Subscriber("/my_gen3/joint_states", JointState)
        self.tool_sub = rospy.Subscriber("/my_gen3/base_feedback", BaseCyclic_Feedback, self.tool_callback)

        if self.model == "hitl_hgd":
            if self.use_pointcloud:
                self.ts = message_filters.ApproximateTimeSynchronizer([self.pc_segment_sub, self.centroids_sub, self.joint_sub], queue_size=5, slop=0.1)
                self.ts.registerCallback(self.callbackHITLHGD)
            else:
                self.ts = message_filters.ApproximateTimeSynchronizer([self.centroids_sub, self.joint_sub], queue_size=5, slop=0.1)
                self.ts.registerCallback(self.callbackHITLHGDnopc)
        else:
            if self.use_pointcloud:
                self.ts = message_filters.ApproximateTimeSynchronizer([self.depth_sub, self.joint_sub], queue_size=5, slop=0.1)
                self.ts.registerCallback(self.callbackHITLD)
            else:
                self.ts = message_filters.ApproximateTimeSynchronizer([self.joint_sub], queue_size=5, slop=0.1)
                self.ts.registerCallback(self.callbackHITLDnopc)

        self.cmd_pub = rospy.Publisher("/my_gen3/inference", Float64MultiArray, queue_size=10)

        rospy.loginfo("🤖 Robot inference node initialized")

        # self.callback(None, None)

    def tool_callback(self, msg):
        if self.ku.get_eef_pose() is not None:
            ret = self.ku.get_eef_pose()
            rads = ret[3:6]
            quat = ret[6:]
            self.tooldata = [
                msg.base.tool_pose_x, 
                msg.base.tool_pose_y, 
                msg.base.tool_pose_z, 
                rads[0],
                rads[1],
                rads[2],
                quat[0],
                quat[1],
                quat[2],
                quat[3]
            ]
        else:
            self.tooldata = [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]

    def callbackHITLHGD(self, pc_msg, centroids_msg, joint_msg):
        try:
            # Convert PointCloud2 to numpy array [N, 3]
            start_time = time.time()
            pointcloud = preprocess_point_cloud(pc_msg, use_cuda=True, color=False)

            centroid = pointcloud.mean(axis=0)
            pointcloud = pointcloud - centroid

            end_time = time.time()
            process_time = end_time - start_time
            rospy.loginfo(f"Processing pointcloud took {process_time:.6f} seconds")
            
            agent_pos = self.get_state_array(centroids_msg, joint_msg)

            if len(self.pointclouds) != self.n_obs:
                self.pointclouds.append(pointcloud.astype(np.float16))  # keep as NumPy
                self.states.append(np.array(agent_pos, dtype=np.float16))
            if len(self.pointclouds) == self.n_obs:
                start_time = time.time()
                rospy.loginfo("Sending data to server")
                payload = {
                    "agent_pos": np.array(self.states, dtype=np.float32),
                    "point_cloud": np.array(self.pointclouds, dtype=np.float32)
                }
                
                self.send_payload_and_publish(payload)

                self.pointclouds.clear()
                self.states.clear()

        except Exception as e:
            rospy.logerr(f"Error in callback: {str(e)}")

    def callbackHITLHGDnopc(self, centroids_msg, joint_msg):
        try:
            agent_pos = self.get_state_array(centroids_msg, joint_msg)

            if len(self.states) != self.n_obs:
                self.states.append(np.array(agent_pos, dtype=np.float16))
            if len(self.states) == self.n_obs:
                start_time = time.time()
                rospy.loginfo("Sending data to server")
                payload = {
                    "agent_pos": np.array(self.states, dtype=np.float32),
                }
                
                self.send_payload_and_publish(payload)

                self.states.clear()

        except Exception as e:
            rospy.logerr(f"Error in callback: {str(e)}")


    def callbackHITLD(self, pc_msg, joint_msg):
        try:
            # Convert PointCloud2 to numpy array [N, 3]
            start_time = time.time()

            pointcloud = preprocess_point_cloud(
                pc_msg, 
                use_cuda=True, 
                color=False if self.model == "hitl_hgd" else True, 
                model=self.model,
                num_points=1024 if self.model == "hitl_hgd" else 2048)
            centroid = pointcloud.mean(axis=0)
            pointcloud = pointcloud - centroid

            end_time = time.time()
            process_time = end_time - start_time
            rospy.loginfo(f"Processing pointcloud took {process_time:.6f} seconds")
            
            agent_pos = self.get_state_array(None, joint_msg)

            if len(self.pointclouds) != self.n_obs:
                self.pointclouds.append(pointcloud.astype(np.float16))  # keep as NumPy
                self.states.append(np.array(agent_pos, dtype=np.float16))
            if len(self.pointclouds) == self.n_obs:
                start_time = time.time()
                rospy.loginfo("Sending data to server")
                payload = {
                    "agent_pos": np.array(self.states, dtype=np.float32),
                    "point_cloud": np.array(self.pointclouds, dtype=np.float32)
                }
                
                self.send_payload_and_publish(payload)

                self.pointclouds.clear()
                self.states.clear()

        except Exception as e:
            rospy.logerr(f"Error in callback: {str(e)}")


    def callbackHITLDnopc(self, joint_msg):
        try:
            agent_pos = self.get_state_array(None, joint_msg)

            if len(self.states) != self.n_obs:
                self.states.append(np.array(agent_pos, dtype=np.float16))
            if len(self.states) == self.n_obs:
                start_time = time.time()
                rospy.loginfo("Sending data to server")
                payload = {
                    "agent_pos": np.array(self.states, dtype=np.float32),
                }
                
                self.send_payload_and_publish(payload)

                self.states.clear()

        except Exception as e:
            rospy.logerr(f"Error in callback: {str(e)}")


    def get_state_array(self, centroids_msg, joint_msg):
        differences = []
        if self.use_centroids:
            centroids = list(centroids_msg.data)
            if len(centroids) < (3*self.num_prompts):
                centroids += [0] * ((3*self.num_prompts)-len(centroids))
            
            for i in range(1, self.num_prompts): #skip the red ring
                for j in range(i+1, self.num_prompts):
                    differences += [centroids[i*3] - centroids[j*3], centroids[(i*3)+1] - centroids[(j*3)+1], centroids[(i*3)+2] - centroids[(j*3)+2]]

        norm_diffs = []
        if self.use_norm_diffs:
            ee_vec = np.array(centroids[:3]) - np.array(centroids[3:6])
            ee_unit_vec = self.normalize(ee_vec)
            
            for i in range(self.num_prompts-2):
                raw_target_dist = np.array(differences[i*3:(i+1)*3])
                target_vec = self.normalize(raw_target_dist)
            
                diff = self.unit_vector_diff(ee_unit_vec, target_vec)
                norm_diffs.append(diff)
        print(norm_diffs)

        # GET ROBOT STATE
        robot_state = []
        if self.joint_pos:
            robot_state += joint_msg.position[:7]

        if self.use_gripper:
            gripper_state = joint_msg.position[7]
            gripper_state = 1 if gripper_state > 0.3 else -1
            robot_state += [gripper_state]

        if self.use_ee_position:
            robot_state += self.tooldata[:3]

        if self.use_centroids:
            robot_state += differences

        if self.use_norm_diffs:
            robot_state += norm_diffs
        # GET ROBOT STATE

        return robot_state

    def normalize(self, a, eps=1e-6):
        mag = np.linalg.norm(a)
        if mag > eps:
            a_norm = a / mag
        else:
            a_norm = a

        return a_norm

    def send_payload_and_publish(self, payload):
        # Send via ZeroMQ
        start_time = time.time()
        self.socket.send(msgpack.packb(payload, default=msgpack_numpy_encode, use_bin_type=True))

        try:
            response = self.socket.recv()
        except zmq.Again:
            rospy.logwarn("Inference server timeout")
            return

        result = msgpack.unpackb(response, object_hook=msgpack_numpy_decode, raw=False)

        # Extract and publish action
        # ABSOLUTE
        action = result["action"][0]
        # ABSOLUTE

        # DIFF
        # action = result["action"]
        # DIFF
        rospy.loginfo(f"ACTION: {action}")
        if self.model == "hitl_hgd":
            self.msg.data = [x for row in action for x in row] # publish as 6d
        else:
            self.msg.data = [float(math.degrees(x)) for row in action for x in row]
        self.cmd_pub.publish(self.msg)

        # rospy.loginfo(f"Published action: {action}")
        end_time = time.time()
        sending_time = end_time - start_time
        rospy.loginfo(f"Sending and publishing took {sending_time:.6f} seconds")
        time.sleep(0.4)

    def unit_vector_diff(self, a, b, eps=1e-8):
        # Normalize to unit vectors
        a_unit = a / (np.linalg.norm(a) + eps)
        b_unit = b / (np.linalg.norm(b) + eps)
        
        # Return the L2 distance between the tips of the vectors
        return np.linalg.norm(a_unit - b_unit, axis=-1)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python record.py <model> <local_server>")
        sys.exit(1)
    if sys.argv[2] not in ["0", "1"]:
        print("local_server must be 0 or 1")
        sys.exit(1)
    if sys.argv[1] not in ["hitl_d", "hitl_hgd"]:
        print("Model does not exist, must be <hitl_d, hitl_hgd>")
        sys.exit(1)
    model = sys.argv[1]
    try:
        node = RobotInferenceNode(model=model, local=0 if sys.argv[2] == "0" else 1)
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
    except rospy.ROSInterruptException:
        pass