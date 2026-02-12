#!/usr/bin/python3

from kortex_driver.msg._BaseCyclic_Feedback import BaseCyclic_Feedback
import rospy
import message_filters
from sensor_msgs.msg import Image, JointState, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Float64, Int16
from kortex_bringup.msg import Float32MultiArrayStamped
from control_utils.ik_utils import png_control, cartesian_control, joint_control
from sensor_msgs.msg import Joy

from cv_bridge import CvBridge

import threading
import termios
import tty
import select
import cv2
import numpy as np
import os
import sys
import curses
import tf2_ros
import ros_numpy
from tqdm import tqdm
import time

from kinova_util import KinovaUtil

from pointcloud_processing import preprocess_point_cloud

bridge = CvBridge()

class Recorder(png_control):
    def __init__(self, dataset):
        super(Recorder, self).__init__(None)
        rospy.init_node('record_py')

        self.ku = KinovaUtil()

        self.tool_sub = rospy.Subscriber("/my_gen3/base_feedback", BaseCyclic_Feedback, self.tool_callback)

        # data folder setup stuff
        self.base_path = os.path.join(os.getcwd(), "data", dataset)
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

        # episode will be appended if we already started recording
        existing = [
            int(name) for name in os.listdir(self.base_path)
            if os.path.isdir(os.path.join(self.base_path, name)) and name.isdigit()
        ]
        self.episode_num = max(existing) + 1 if existing else 0

        joints_sub = message_filters.Subscriber('/my_gen3/joint_states', JointState)
        depth_sub = message_filters.Subscriber('/cam/depth/color/points', PointCloud2)
        pc_segment_sub = message_filters.Subscriber('/my_gen3/segment_pc_mask', PointCloud2)
        centroids_sub = message_filters.Subscriber('/my_gen3/pc_centroids', Float32MultiArrayStamped)
        self.joy_sub = rospy.Subscriber("/joy", Joy, self.joy_callback)

        ts = message_filters.ApproximateTimeSynchronizer([joints_sub, pc_segment_sub, centroids_sub], 100, slop=10)
        ts.registerCallback(self.syncCallback)

        # desyncing check setup
        self.last_sync_time = rospy.Time.now()
        self.sync_timeout = rospy.Duration(1.0)
        rospy.Timer(rospy.Duration(0.5), self.checkDesync)

        # self.home_array = np.array([0.1, 65, -179.9, -120, 0, 100, -90])
        # self._call_clear_faults()
        # self.send_joint_angles(self.home_array)

        self.last_press = 'end'
        self.stage = 1
        self.time_last = time.time()
        self.centroids = None

        rospy.loginfo("Ready to record")
    
    
    def __del__(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_attrs)

    def tool_callback(self, msg):
        if self.ku.get_eef_pose() is not None:
            rads = self.ku.get_eef_pose()[3:]
            self.tooldata = [
                msg.base.tool_pose_x, 
                msg.base.tool_pose_y, 
                msg.base.tool_pose_z, 
                rads[0],
                rads[1],
                rads[2],
            ]
        else:
            self.tooldata = [
                0,
                0,
                0,
                0,
                0,
                0,
            ]

    def syncCallback(self, joint_msg, depth, centroid_msg):
        # it is possible something breaks and we become desynced, dont want to record episode in this case
        self.last_sync_time = rospy.Time.now()

        # rospy.loginfo(f"joints: {joints.header.stamp.to_sec()}")
        # rospy.loginfo(f"pc_segment: {pc_segment.header.stamp.to_sec()}")
        # rospy.loginfo(f"centroids: {centroids.header.stamp.to_sec()}")

        # havent started yet
        if not hasattr(self, "curr_low_dim"):
            return
       
        try:
            # numpy_pc = np.column_stack([ros_numpy.numpify(depth)[f] for f in ("x", "y", "z", "rgb")])
            self.curr_depth.append(depth)

            data = {
                'joints': {
                    'position': joint_msg.position,
                    'velocity': joint_msg.velocity,
                },
                "ee_position": self.tooldata[:3],
                "ee_orientation": self.tooldata[3:],
                "centroids": centroid_msg.data
            }
            
            self.curr_low_dim.append(data)
        except Exception as e:
            rospy.logerr("error in sync: %s", e)
   
    def checkDesync(self, event):
        now = rospy.Time.now()
        time_since_last_sync = now - self.last_sync_time

        if time_since_last_sync > self.sync_timeout:
            rospy.logwarn("Desynced, cancelling episode. Press space to rerun or q to exit")
            self._call_clear_faults()

    def joy_callback(self, msg):
        self.buttons = msg.buttons
        if msg.buttons[7]:
            if self.last_press == 'start' and time.time() - self.time_last > 0.5:
                rospy.loginfo("Episode ended")
                self.handleEpisodeEnd()
                self.last_press = 'end'
                self.time_last = time.time()
            elif self.last_press == 'end' and time.time() - self.time_last > 0.5:
                rospy.loginfo("Episode started")
                self.handleEpisodeStart()
                self.last_press = 'start'
                self.time_last = time.time()

    def handleEpisodeStart(self):
        rospy.loginfo("Starting episode " + str(self.episode_num))
        self.last_sync_time = rospy.Time.now()
        self.curr_low_dim = []
        self.curr_depth =  []
        self.stage = 1
   
    def handleEpisodeEnd(self):
        rospy.loginfo(f"Episode {self.episode_num} recorded")

        # create episode folder
        current_folder = os.path.join(self.base_path, str(self.episode_num))
        os.makedirs(current_folder)

        indexes = self.select_evenly_spaced([i for i in range(len(self.curr_low_dim))], max_length=1024)

        for i, idx in enumerate(tqdm(indexes)):
            # save each frame
            frame_folder = os.path.join(current_folder, str(i))
            os.makedirs(frame_folder)
            np.save(os.path.join(frame_folder, "low_dim.npy"), self.curr_low_dim[idx])
            # np.save(os.path.join(frame_folder, "depth.npy"), self.curr_depth[idx])
            np.save(os.path.join(frame_folder, "depth.npy"), preprocess_point_cloud(self.curr_depth[idx], color=False))

        rospy.loginfo(f"Finished saving episode {self.episode_num}")
        self.episode_num += 1

    def select_evenly_spaced(self, array, max_length=48):
        n = len(array)
        if n <= max_length:
            return array
        indices = np.linspace(0, n - 1, max_length, dtype=int)
        return [array[i] for i in indices]


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python record.py <dataset_name>")
        sys.exit(1)

    dataset = sys.argv[1]

    recorder = Recorder(dataset)

    rospy.spin()