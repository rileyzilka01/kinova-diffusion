#!/usr/bin/python3

import rospy
import numpy as np
from kortex_driver.srv import *
from kortex_driver.msg import *
from sensor_msgs.msg import Joy
from control_utils.ik_utils import png_control, cartesian_control, joint_control, xbox_control
from control_utils.kinova_gen3 import RGBDVision

from std_msgs.msg import Int32MultiArray, Float32, Int16

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
            self.mode = 0 # modes for control
            self.automatic = 0 # mode for whether it approaches automatically
            self.prev_button_2 = 0 # prev button 2 to prevent double clicks
            self.prev_gripper_cmd = 0.0 # prev gripper cmd
            self.gripper_cmd = 0.0 # gripper cmd
            self.axes_vector = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # joystick cmd
            # self.home_array = np.array([30, 14, 148, -80, 57, 3, -140]) # home array in deg
            self.home_array = np.array([0.1, 65, -179.9, -120, 0, 100, -90]) # home array in deg
            self.send_joint_angles(self.home_array) #sends robot home
            self.window_center = (424, 240)
            self.grip_center = None
            self.stage = 1

            self.custom_commands = []
            print(f"Current mode = {self.mode}\a", end="\r")

            self.tool_sub = rospy.Subscriber("/my_gen3/base_feedback", BaseCyclic_Feedback, self.tool_callback)
            self.joy_sub = rospy.Subscriber("/joy", Joy, self.joy_callback)

            self.stage_pub = rospy.Publisher("/my_gen3/inference/stage", Int16, queue_size=10)

        def mode_switch(self):
            self.mode = (self.mode + 1) % 2
            print(f"Current mode = {self.mode}\a", end="\r")
            return

        def tool_callback(self, msg):
            self.tooldata = [msg.base.tool_pose_x, msg.base.tool_pose_y, msg.base.tool_pose_z, msg.base.tool_pose_theta_x, msg.base.tool_pose_theta_y, msg.base.tool_pose_theta_z]
            # print("TOOL DATA", self.tooldata)

        def joy_callback(self, msg):
            self.joy_type = 1 #0 for joystick 1 for xbox controller
            self.buttons = msg.buttons

            # check for gripper commands
            if self.joy_type == 0:
                self.axes_vector = msg.axes
                MAXV_GR = 0.3

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
                    self.automatic = not self.automatic

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
                    self.run = False
                    self.send_joint_speeds_command(np.zeros(7))
                    self.send_joint_angles(self.home_array)
                    rospy.loginfo("Button 8 pressed: sending robot to starting position")
                    self.run = True

                if msg.buttons[9]:
                    pass

                if msg.buttons[10]:
                    pass
                    
                if msg.buttons[11]:
                    pass

            elif self.joy_type == 1:
                MAXV_GR = 0.3
                roll = msg.buttons[1] - msg.buttons[3]
                self.axes_vector = [msg.axes[1], msg.axes[0], (1/(msg.axes[5]+1.1) - 1/(msg.axes[2]+1.1))/10, -msg.axes[4]/2, msg.axes[3], roll]
                self.axes_vector = [self.axes_vector[i]/2 for i in range(len(self.axes_vector))]
                # self.axes_vector = [msg.axes[1], msg.axes[0], msg.axes[2], msg.axes[4], msg.axes[3], msg.axes[5]]

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
                    pass
                
                if msg.buttons[8]:
                    pass

                if msg.buttons[9]:
                    pass

                if msg.buttons[10]:
                    pass

        def step(self):
            if self.run:
                self.custom_commands = []
                
                # step according to rospy rate
                # super().step(self.axes_vector, self.mode, self.custom_commands)
                super().step(self.axes_vector, self.custom_commands)
                if self.gripper_cmd != self.prev_gripper_cmd:
                    success = self.send_gripper_command(self.gripper_cmd)
                    self.prev_gripper_cmd = self.gripper_cmd

    return IrisRecord()

def main():
    controller = xbox_control # can replace with cartesian_control or joint_control or png_control or xbox_control
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


