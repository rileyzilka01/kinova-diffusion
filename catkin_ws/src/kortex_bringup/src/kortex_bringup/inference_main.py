#!/usr/bin/python3

import rospy
import numpy as np
from kortex_driver.srv import *
from kortex_driver.msg import *
from sensor_msgs.msg import Joy, PointCloud2, JointState
from control_utils.ik_utils import png_control, cartesian_control, joint_control, xbox_control
from control_utils.kinova_gen3 import RGBDVision
import time
import math
import zmq
import json
import message_filters
from kinova_util import KinovaUtil
import sys
import ros_numpy
import msgpack
from pointcloud_processing import preprocess_point_cloud

from std_msgs.msg import Int32MultiArray, Float32, Float64MultiArray, Int16
from kortex_driver.msg._BaseCyclic_Feedback import BaseCyclic_Feedback

import msgpack_numpy
msgpack_numpy.patch()
msgpack_numpy_encode = msgpack_numpy.encode
msgpack_numpy_decode = msgpack_numpy.decode

class CustomCommand():
    def __init__(self, ax, mode, trans_gain, rot_gain, wrist_gain):
        self.ax = ax
        self.mode = mode
        self.trans_gain = trans_gain
        self.rot_gain = rot_gain
        self.wrist_gain = wrist_gain

def gen_iris(base):
    class IrisRecord(base):
        def __init__(self):
            super(IrisRecord, self).__init__(None)
            self.ku = KinovaUtil()
            self.mode = 0 # modes for control
            self.prev_button_2 = 0 # prev button 2 to prevent double clicks
            self.prev_gripper_cmd = 0.0 # prev gripper cmd
            self.gripper_cmd = 0.0 # gripper cmd
            self.axes_vector = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # joystick cmd
            self.home_array = np.array([0.1, 65, -179.9, -120, 0, 100, -90]) # home array in deg
            # self.home_array = np.array([30, 14, 148, -80, 57, 3, -140]) # home array in deg
            self.send_joint_angles(self.home_array) #sends robot home
            self.window_center = (424, 240)
            self.grip_center = None
            self.orientation = None
            self.infer = False
            self.stage = 1
            self.time_last = time.time()
            self.joy_type = 1
            self.auto = False
            self.agent_pos = None
            self.prev_robot_pos = None
            self.pc = None
            self.pointclouds = []
            self.states = []
            self.first = True
            self.trial_started = False
            self.trial_time = None
            self.reset_count = 0
            self.mode_switches = 0

            self.joy_sub = rospy.Subscriber("/joy", Joy, self.joy_callback)
            self.tool_sub = rospy.Subscriber("/my_gen3/base_feedback", BaseCyclic_Feedback, self.tool_callback)
            self.orient_sub = rospy.Subscriber("/my_gen3/inference", Float64MultiArray, self.orient_callback)

            if self.auto:
                # ZeroMQ setup
                self.context = zmq.Context()
                self.socket = self.context.socket(zmq.REQ)
                self.socket.connect("tcp://192.168.1.161:5555")
                self.socket.setsockopt(zmq.RCVTIMEO, 1000)

                self.socket.setsockopt(zmq.RCVTIMEO, 2000)  # 2s timeout
                self.socket.setsockopt(zmq.LINGER, 0)       # Don't hang on close if server is dead

                self.n_obs = 2
                try:
                    ping_msg = msgpack.packb({"ping": True}, use_bin_type=True)
                    self.socket.send(ping_msg)
                    
                    response_msg = self.socket.recv()
                    response = msgpack.unpackb(response_msg, raw=False)
                    
                    rospy.loginfo("✅ Connected to inference server")
                except zmq.Again:
                    rospy.logerr("❌ Could not connect to inference server (timeout)")
                    sys.exit(1)

                self.cam_sub = rospy.Subscriber("/cam/depth/color/points", PointCloud2, self.pc_callback)
                self.joint_sub = rospy.Subscriber("/my_gen3/joint_states", JointState, self.joint_callback)

                self.cmd_pub = rospy.Publisher("/my_gen3/inference", Float64MultiArray, queue_size=10)

                rospy.loginfo("🤖 Robot inference node initialized")
            
            self.run = True

            self.custom_commands = []
            print(f"Current mode = {self.mode}\a", end="\r")
            
        def mode_switch(self):
            self.mode = (self.mode + 1) % 2
            self.mode_switches += 1
            print(f"Current mode = {self.mode}\a", end="\r")
            return

        def joint_callback(self, joint_msg):
            self.agent_pos = list(joint_msg.position[:8])
            self.agent_pos[7] = 1 if self.agent_pos[7] > 0.7 else 0

        def pc_callback(self, pc_msg):
            self.pc = preprocess_point_cloud(pc_msg, use_cuda=True)

        def tool_callback(self, msg):
            rads = self.ku.get_eef_pose()[3:]
            self.tooldata = [
                msg.base.tool_pose_x, 
                msg.base.tool_pose_y, 
                msg.base.tool_pose_z, 
                math.degrees(rads[0]),
                math.degrees(rads[1]),
                math.degrees(rads[2]),
            ]

        def orient_callback(self, msg):
            shape = [dim.size for dim in msg.layout.dim]
            temp = np.array(msg.data).reshape(shape)
            self.orientation = temp[0]
            # ABSOLUTE
            self.target = self.orientation
            # ABSOLUTE

            # DIFF
            # self.old_position = self.tooldata[3:]
            # self.target = self.old_position + self.orientation
            # DIFF

        def joy_callback(self, msg):
            self.buttons = msg.buttons
            
            if self.joy_type == 0:
                self.axes_vector = msg.axes
                MAXV_GR = 0.3

                # check for gripper commands
                if msg.buttons[0]: # trigger button - close gripper
                    self.gripper_cmd = -1 * MAXV_GR
                elif msg.buttons[1]: # button by thumb - open gripper
                    self.gripper_cmd = MAXV_GR
                else: # both buttons 0 and 1 are zero
                    self.gripper_cmd = 0.0

                if msg.buttons[2] == 0 and self.prev_button_2 == 1:
                    self.mode_switch()
                    
                self.prev_button_2 = msg.buttons[2]
                    
                if msg.buttons[3]:
                    if time.time() - self.time_last > 0.5:
                        self.infer = not self.infer
                        rospy.loginfo(f"Inference: {'True' if self.infer else 'False'}")
                        self.time_last = time.time()
                        self.mode_switches += 1

                if msg.buttons[4]:
                    pass

                if msg.buttons[5]:
                    pass

                if msg.buttons[6]:
                    pass

                if msg.buttons[7]:
                    pass

                if msg.buttons[8]:
                    # button 8 pressed, send robot home
                    if time.time() - self.time_last > 0.5:
                        self.run = False
                        rospy.loginfo(f"Trial State: PAUSED")
                        self.send_joint_speeds_command(np.zeros(7))
                        self.send_joint_angles(self.home_array)
                        rospy.loginfo("Button 8 pressed: sending robot to starting position")
                        self.reset_count += 1
                        self.run = True
                        self.time_last = time.time()
                        self.time_paused = time.time()

                if msg.buttons[9]:
                    if not self.trial_started  and time.time() - self.time_last > 0.5:
                        self.trial_started = True
                        self.trial_time = time.time()
                        self.reset_count = 0
                        self.mode_switches = 0
                        rospy.loginfo(f"Trial State: STARTED")
                        self.time_last = time.time()
                    elif self.trial_started  and time.time() - self.time_last > 0.5:
                        self.trial_started = False
                        rospy.loginfo(f"Trial Time: {time.time() - self.trial_time}")
                        rospy.loginfo(f"Trial State: SUCCESS")
                        rospy.loginfo(f"Trial Reset Count: {self.reset_count}")
                        rospy.loginfo(f"Trial Mode Switches: {self.mode_switches}")
                        self.time_last = time.time()

                if msg.buttons[10]:
                    if time.time() - self.time_last > 0.5:
                        rospy.loginfo(f"Trial State: RESUMED")
                        temp = time.time() - self.time_paused
                        self.trial_time += temp
                        self.time_paused = 0
                        self.time_last = time.time()

                if msg.buttons[11]:
                    if self.trial_started and time.time() - self.time_last > 0.5:
                        self.trial_started = False
                        rospy.loginfo(f"Trial Time: {180}")
                        rospy.loginfo(f"Trial State: FAIL")
                        rospy.loginfo(f"Trial Reset Count: {self.reset_count}")
                        rospy.loginfo(f"Trial Mode Switches: {self.mode_switches}")
                        self.time_last = time.time()


            elif self.joy_type == 1:
                MAXV_GR = 0.3
                roll = msg.buttons[1] - msg.buttons[3]
                self.axes_vector = [msg.axes[1], msg.axes[0], (1/(msg.axes[5]+1.1) - 1/(msg.axes[2]+1.1))/10, -msg.axes[4]/2, msg.axes[3], roll]

                if msg.buttons[0]:
                    pass

                if msg.buttons[1]:
                    pass

                if msg.buttons[2]:
                    pass
                    
                if msg.buttons[3]:
                    pass

                if msg.buttons[4]: # LB - open gripper
                    self.gripper_cmd = MAXV_GR

                elif msg.buttons[5]: # RB - close gripper
                    self.gripper_cmd = -1 * MAXV_GR

                else: # both buttons 0 and 1 are zero
                    self.gripper_cmd = 0.0

                if msg.buttons[6]:
                    # Start button pressed, send robot home
                    self.run = False
                    self.send_joint_speeds_command(np.zeros(7))
                    self.send_joint_angles(self.home_array)
                    rospy.loginfo("Start button pressed: sending robot to starting position")
                    self.run = True

                if msg.buttons[7]:
                    if time.time() - self.time_last > 0.5:
                        self.infer = not self.infer
                        rospy.loginfo(f"Inference: {'True' if self.infer else 'False'}")
                        self.time_last = time.time()
                
                if msg.buttons[8]:
                    pass

                if msg.buttons[9]:
                    pass

                if msg.buttons[10]:
                    pass

        def get_orientation(self):
            if self.orientation is not None:
                # rospy.loginfo(f"Target: {self.target}")
                # rospy.loginfo(f"Current: {self.tooldata[3:]}")
                diff = [abs(self.target[i] - self.tooldata[3+i]) for i in range(3)]
                # rospy.loginfo(f"Diff: {diff}")
                if self.joy_type == 0:
                    velocities = np.array([
                        (-1 if diff[2] > 180 else 1)*0.05 * ((self.target[2] - self.tooldata[5]) if abs(self.target[2] - self.tooldata[5]) > 0.5 else 0), #Correct axes placement
                        (-1 if diff[0] > 180 else 1)*0.05 * ((self.target[0] - self.tooldata[3]) if abs(self.target[0] - self.tooldata[3]) > 0.5 else 0), #Correct axes placement
                        (1 if diff[1] > 180 else -1)*0.05 * ((self.target[1] - self.tooldata[4]) if abs(self.target[1] - self.tooldata[4]) > 0.5 else 0), #Correct axes placement
                    ])
                if self.joy_type == 1:
                    velocities = np.array([
                        0,
                        0,
                        0,
                        (-1 if diff[0] > 180 else 1)*0.05 * ((self.target[0] - self.tooldata[3]) if abs(self.target[0] - self.tooldata[3]) > 0.5 else 0),
                        (1 if diff[2] > 180 else -1)*0.05 * ((self.target[2] - self.tooldata[5]) if abs(self.target[2] - self.tooldata[5]) > 0.5 else 0), #Correct axes placement,
                        (-1 if diff[1] > 180 else 1)*0.05 * ((self.target[1] - self.tooldata[4]) if abs(self.target[1] - self.tooldata[4]) > 0.5 else 0),
                    ])
                # rospy.loginfo(f"Resulting velocities: {velocities}")

                self.custom_commands.append(CustomCommand(velocities, 1, 1, 1, 1))

        def auto_pos(self):
            if self.pc is not None:
                MAXV_GR = 0.7
                if self.first:
                    self.pointclouds = [self.pc.astype(np.float16), self.pc.astype(np.float16)]
                    self.states = [np.zeros(8), np.zeros(8)]
                    self.prev_robot_pos = self.agent_pos
                if len(self.pointclouds) == self.n_obs:
                    start_time = time.time()
                    rospy.loginfo("Sending data to server")
                    payload = {
                        "agent_pos": np.array(self.states),
                        "point_cloud": np.array(self.pointclouds)
                    }
                    
                    # Send via ZeroMQ
                    self.socket.send(msgpack.packb(payload, default=msgpack_numpy_encode, use_bin_type=True))

                    try:
                        response = self.socket.recv()
                    except zmq.Again:
                        rospy.logwarn("Inference server timeout")
                        return

                    result = msgpack.unpackb(response, object_hook=msgpack_numpy_decode, raw=False)

                    # Extract and publish action
                    action = result["action"][0][:8]
                    actions = [[float(math.degrees(x)) for x in row[:7]]+[row[7]] for row in action]
                    # actions = actions[14:]
                    
                    for temp in actions:
                        print(temp)

                    self.pointclouds = []
                    self.states = []
                    self.run = False
                    for a in range(len(actions)):
                        self.send_joint_speeds_command(np.zeros(7))
                        rospy.sleep(0.1)
                        # actions[a][2] = 0
                        to_pos = [math.degrees(self.agent_pos[z]) + actions[a][z] for z in range(7)]
                        rospy.loginfo(f"AGENT POS: {self.agent_pos}")
                        rospy.loginfo(f"PREDICTION: {actions[a]}")
                        rospy.loginfo(f"TO_POS: {to_pos}")
                        self.send_joint_angles(np.array(to_pos))
                        rospy.loginfo("Sending joint commands")
                        rospy.sleep(0.5)

                        if a >= len(actions)-2:
                            self.pointclouds.append(self.pc)
                            current = self.agent_pos
                            diffs = [current[z] - self.prev_robot_pos[z] for z in range(7)] + [self.agent_pos[7]]
                            self.states.append(np.array(diffs, dtype=np.float16))
                        
                        self.prev_robot_pose = self.agent_pos

                        if actions[a][7] < -0.7:
                            success = self.send_gripper_command(MAXV_GR)
                        elif actions[a][7] > 0.7: 
                            success = self.send_gripper_command(-1 * MAXV_GR)
                        else:
                            success = self.send_gripper_command(0)
                    
                    self.run = True
                    
                    if self.first:
                        self.first = False

        def step(self):
            if self.run:
                self.custom_commands = []

                if self.infer:
                    self.get_orientation()
                if self.auto:
                    self.auto_pos()
                else:
                    # step according to rospy rate
                    if self.joy_type == 0:
                        super().step(self.axes_vector, self.mode, self.custom_commands)
                    elif self.joy_type == 1:
                        super().step(self.axes_vector, self.custom_commands)
                    if self.gripper_cmd != self.prev_gripper_cmd:
                        success = self.send_gripper_command(self.gripper_cmd)
                        self.prev_gripper_cmd = self.gripper_cmd

    return IrisRecord()

def main():
    controller = xbox_control# can replace with png_control, cartesian_control or joint_control or xbox_control
    robot = gen_iris(controller)
    
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        robot.step()
        rate.sleep()


if __name__ == "__main__":
    try:
        rospy.init_node('iris_control', anonymous=True)
        main()
    except rospy.ROSInterruptException:
        print("ROSInterruptException")


