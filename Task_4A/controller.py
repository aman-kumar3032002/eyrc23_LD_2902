'''
# Team ID:          2902
# Theme:            Luminosity Drone
# Author List:      Aman Kumar, Shivam Kumar, Manila Raj Putra, Harsh Gulzar, 
# Filename:         LD_2902_controller.py
# Functions:        __init__,whycon_poses_callback(),pid_tune_roll_callback(),pid_tune_pitch_callback(),pid_tune_throttle_callback(),pid(),publish_data_to_rpi,shutdown_hook(),arm(),disarm(),main()
# Global variables: MIN_ROLL,BASE_ROLL,MAX_ROLL,SUM_ERROR_ROLL_LIMIT,MIN_ROLL_PITCH,BASE_PITCH,MAX_PITCH, SUM_ERROR_PITCH_LIMIT, MIN_THROTTLE,BASE_THROTTLE,MAX_THROTTLE,SUM_ERROR_THROTTLE_LIMIT,BASE_YAW
'''

#!/usr/bin/env python3
"""
Controller for the drone
"""

# standard imports
import copy
import time

# third-party imports
import scipy.signal
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from geometry_msgs.msg import PoseArray
from pid_msg.msg import PidTune
from swift_msgs.msg import PIDError, RCMessage
from swift_msgs.srv import CommandBool


# battery voltage 12.5 - 12 v
#global variable for roll ----------------------------------------------------------------------------------------------
MIN_ROLL = 1420                       
BASE_ROLL = 1488 
MAX_ROLL = 1550
SUM_ERROR_ROLL_LIMIT =100

#gloabal vaiables for pitch---------------------------------------------------------------------------------------------
MIN_PITCH = 1420    #MIN_PITCH = 
BASE_PITCH = 1490
MAX_PITCH = 1550
SUM_ERROR_PITCH_LIMIT = 100

#gloabal vaiable for throttle-------------------------------------------------------------------------------------------
MIN_THROTTLE = 1420
BASE_THROTTLE = 1453
MAX_THROTTLE= 1550
SUM_ERROR_THROTTLE_LIMIT = 80000

#gloabal variable for yaw-----------------------------------------------------------------------------------------------
MIN_YAW= 1000
BASE_YAW = 1500
MAX_YAW = 2000
SUM_ERROR_ROLL_LIMIT = 400
#-----------------------------------------------------------------------------------------------------------------------
DRONE_WHYCON_POSE = [[], [], []]

class DroneController():
    def __init__(self,node):
        self.node= node
        
        self.rc_message = RCMessage()                                                                                           #self.rc_message                   : object of the the RCMessage class
        self.drone_whycon_pose_array = PoseArray()                                                                              #self.drone_whycon_pose_array      : object of the PoseArray class
        self.last_whycon_pose_received_at = 0                                                                                   #self.last_whycon_poses_received_at: stores the last_whycon_pose_received location
        self.commandbool = CommandBool.Request()                                                                                #self.commandbool                  : object of the CommandBool.Request class
        service_endpoint = "/swift/cmd/arming"
        self.arming_service_client = self.node.create_client(CommandBool,service_endpoint)
        self.set_points = [0, 0, 22]         # Setpoints for x, y, z respectively      
        self.drone_position = [0.0, 0.0, 0.0]                                                                                   #self.drone_postion: stores the current position of the drone
        self.error      = [0.0, 0.0, 0.0]                                                                                       #self.error        : list, of error for roll, pitch and throttle respectively  
        self.prev_error = [0.0, 0.0, 0.0]                                                                                       #self.prev_error   : list to store the previous errors
        self.error_diff = [0.0, 0.0, 0.0]                                                                                       #self.error_diff   : stores the error difference as a list
        self.error_sum  = [0.0, 0.0, 0.0]                                                                                       #self.error_sum    : store the error_sum as a list
        
        #Defining the PID constants for battery level ~12.5 to  ~12.1       
        #values for the PID----------------------------------------------------------------------------------------------------
        self.Kp = [ 4.29, 4.9 ,4.53]                                                                                            #self.kp: stores the Kp values for all three axis - [roll,pitch,throttle]
        self.Ki = [ 0.004, 0.004, 0.00885]                                                                                          #self.ki: stores the Ki values for all three axis - [roll,pitch,thorttle]   
        self.Kd = [80.1, 110.1, 117.8]                                                                                         #self.kd: stores the kd values for all three axis - [roll,pitch,throttle]                   
        #--------------------------------------------------------------------------------------------------------------------

        # Create subscriber for WhyCon 
        
        self.whycon_sub = node.create_subscription(PoseArray,"/whycon/poses",self.whycon_poses_callback,1)                      #self.pid__whycon_sub: subscriber for the whycon poses
        self.pid_roll   = node.create_subscription(PidTune,"/pid_tuning_roll",self.pid_tune_roll_callback,1)                    #self.pid_roll   : subscriber for roll axis
        self.pid_pitch  = node.create_subscription(PidTune,"/pid_tuning_pitch",self.pid_tune_pitch_callback,1)                  #self.pid_pitch  : subscriber for pitch axis
        self.pid_alt    = node.create_subscription(PidTune,"/pid_tuning_throttle",self.pid_tune_throttle_callback,1)            #self.pid_alt    : subscriber for the alt axis

        # Create publisher for publishing errors for plotting in plotjuggler 
        
        self.pid_error_pub = node.create_publisher(PIDError, "/luminosity_drone/pid_error",1)        
        self.rc_pub = node.create_publisher(RCMessage, "/swift/rc_command",1) 

    def whycon_poses_callback(self, msg):
        self.last_whycon_pose_received_at = self.node.get_clock().now().seconds_nanoseconds()[0]
        self.drone_whycon_pose_array = msg
        self.drone_position[0] = self.drone_whycon_pose_array.poses[0].position.x              
        self.drone_position[1] = self.drone_whycon_pose_array.poses[0].position.y
        self.drone_position[2] = self.drone_whycon_pose_array.poses[0].position.z
        
 
    def pid_tune_roll_callback(self, msg):
        self.Kp[0] = msg.kp * 0.01
        self.Ki[0] = msg.ki * 0.0001
        self.Kd[0] = msg.kd * 0.1
        
    def pid_tune_pitch_callback(self, msg):
        self.Kp[1] = msg.kp * 0.01
        self.Ki[1] = msg.ki * 0.0001
        self.Kd[1] = msg.kd * 0.1
        
    def pid_tune_throttle_callback(self, msg):
        self.Kp[2] = msg.kp * 0.01
        self.Ki[2] = msg.ki * 0.0001
        self.Kd[2] = msg.kd * 0.1
        
    # --------------------------------------------------------------------------------PID algorithm----------------------------------------------------------------------------
    def pid(self):          
        try:
            self.error[0] = self.drone_position[0] - self.set_points[0]                                                       #self.erorr[0]: calculating the error of the drone in x axis
            self.error[1] = self.drone_position[1] - self.set_points[1]                                                       #self.erorr[1]: calculating the error of the drone in y axis
            self.error[2] = self.drone_position[2] - self.set_points[2]                                                       #self.erorr[2]: calculating the error of the drone in z axis
            #----------------error of all the cordinates(integral)---------------------------------------------------------------------------------------------
            self.error_sum[0] = (self.error_sum[0] + (self.error[0]))
            self.error_sum[1] = (self.error_sum[1] + (self.error[1]))
            self.error_sum[2] = (self.error_sum[2] + (self.error[2]))
            
            #limiting the error_sum(integral)------------------------------------------------------------------------------------------------------------------       
            if self.error_sum[0] > SUM_ERROR_ROLL_LIMIT:
                self.error_sum[0] = SUM_ERROR_ROLL_LIMIT
            if self.error_sum[0] < -SUM_ERROR_ROLL_LIMIT:
                self.error_sum[0] = -SUM_ERROR_ROLL_LIMIT
            if self.error_sum[1] > SUM_ERROR_PITCH_LIMIT:
                self.error_sum[1] = SUM_ERROR_PITCH_LIMIT
            if self.error_sum[1] < -SUM_ERROR_PITCH_LIMIT:
                self.error_sum[1] = -SUM_ERROR_PITCH_LIMIT
            if self.error_sum[2] > SUM_ERROR_THROTTLE_LIMIT:
                self.error_sum[2] = SUM_ERROR_THROTTLE_LIMIT
            if self.error_sum[2] < -SUM_ERROR_THROTTLE_LIMIT:          
                self.error_sum[2] = -SUM_ERROR_THROTTLE_LIMIT
                
            #--------------------------------------------------------------------------------------------------------------------------------------------------   
            #derivative----------------------------------------------------------------------------------------------------------------------------------------            
            self.error_diff[0] = (self.error[0]-self.prev_error[0])   
            self.error_diff[1] = (self.error[1]-self.prev_error[1])
            self.error_diff[2] = (self.error[2]-self.prev_error[2])

            # Write the PID equations and calculate the self.rc_message.rc_throttle, self.rc_message.rc_roll, self.rc_message.rc_pitch
            self.rc_message.rc_roll     = BASE_ROLL - int((self.Kp[0]*self.error[0])+(self.Kd[0]*self.error_diff[0])+(self.error_sum[0]*self.Ki[0]))		#self.rc_message_rc_roll: speed for the roll 
            self.rc_message.rc_pitch    = BASE_PITCH + int((self.Kp[1]*self.error[1])+(self.Kd[1]*self.error_diff[1])+(self.error_sum[0]*self.Ki[0]))		#self.rc_message_rc_pitch: speed for the pitch
            self.rc_message.rc_throttle = BASE_THROTTLE + int((self.Kp[2]*self.error[2])+ (self.Kd[2]*self.error_diff[2])+(self.error_sum[2]*self.Ki[2]))   #self.rc_message_rc_throttle: speed for the throttle 
            
            #setting prev_error to the current_error-----------------------------------------------------------------------------------------------------------
            self.prev_error[0] = self.error[0]                                                                                                              #self.prev_error[0]:storing the current error[0] as prev_error[0]
            self.prev_error[1] = self.error[1]                                                                                                              #self.prev_error[1]:storing the current error[1] as prev_error[1]
            self.prev_error[2] = self.error[2]                                                                                                              #self.prev_error[2]:storing the current error[2] as prev_error[2]

        #calling the function self.publish_data_to_rpi---------------------------------------------------------------------------------------------------------
            self.publish_data_to_rpi( roll = self.rc_message.rc_roll, pitch = self.rc_message.rc_pitch, throttle = self.rc_message.rc_throttle)  
                        
        #publishing data to plotjuggler------------------------------------------------------------------------------------------------------------------------                  
            self.pid_error_pub.publish(
            PIDError(
                    roll_error=float(self.error[0]),
                    pitch_error=float(self.error[1]),
                    throttle_error=float(self.error[2]),
                    yaw_error=-0.8,
                    zero_error=0.8,
                    )
                )
        
        except Exception as e:
            print(e)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    def publish_data_to_rpi(self, roll, pitch, throttle):

        self.rc_message.rc_throttle = int(throttle)                      #self.rc_message.rc_throttle: storing the integer value of the throttle
        self.rc_message.rc_roll = int(roll)                              #self.rc_message.rc_roll    : storing the integer value of the roll
        self.rc_message.rc_pitch = int(pitch)                            #self.rc_message.rc_pitch   : storing the integer value of the pitch

        # Send constant 1500 to rc_message.rc_yaw
        self.rc_message.rc_yaw = int(1500)

        #-------------------------------------------------------- BUTTERWORTH FILTER-------------------------------------------------------------------------
        span = 15
        for index, val in enumerate([self.rc_message.rc_roll, self.rc_message.rc_pitch, self.rc_message.rc_throttle]):
            DRONE_WHYCON_POSE[index].append(val)
            if len(DRONE_WHYCON_POSE[index]) == span:
                DRONE_WHYCON_POSE[index].pop(0)
            if len(DRONE_WHYCON_POSE[index]) != span-1:
                return
            order = 3
            fs = 60
            fc = 5
            nyq = 0.5 * fs
            wc = fc / nyq
            b, a = scipy.signal.butter(N=order, Wn=wc, btype='lowpass', analog=False, output='ba')
            filtered_signal = scipy.signal.lfilter(b, a, DRONE_WHYCON_POSE[index])
            if index == 0:
                self.rc_message.rc_roll = int(filtered_signal[-1])
            elif index == 1:
                self.rc_message.rc_pitch = int(filtered_signal[-1])
            elif index == 2:
                self.rc_message.rc_throttle = int(filtered_signal[-1])

        #-----------------------------------------------bounding the values for roll ,pitch and throttle-----------------------------------------------------
        if self.rc_message.rc_roll > MAX_ROLL:                           #checking range i.e. bet 1000 and 2000
            self.rc_message.rc_roll = MAX_ROLL
        elif self.rc_message.rc_roll < MIN_ROLL:
            self.rc_message.rc_roll = MIN_ROLL
                
        if self.rc_message.rc_pitch > MAX_PITCH:                        #checking range i.e. bet 1000 and 2000
            self.rc_message.rc_pitch = MAX_PITCH
        elif self.rc_message.rc_pitch < MIN_PITCH:
            self.rc_message.rc_pitch = MIN_PITCH
                
        if self.rc_message.rc_throttle > MAX_THROTTLE:                  #checking range i.e. bet 1000 and 2000
            self.rc_message.rc_throttle = MAX_THROTTLE
        elif self.rc_message.rc_throttle < MIN_THROTTLE:
            self.rc_message.rc_throttle = MIN_THROTTLE
        #---------------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------publishing data to rpi------------------------------------------------------------------------------------
   
        self.rc_pub.publish(self.rc_message)
        
    # This function will be called as soon as this rosnode is terminated. So we disarm the drone as soon as we press CTRL + C. 
    # If anything goes wrong with the drone, immediately press CTRL + C so that the drone disamrs and motors stop 

    def shutdown_hook(self):
        self.node.get_logger().info("Calling shutdown hook")
        self.disarm()

    # Function to arm the drone 
    def arm(self):
        self.node.get_logger().info("Calling arm service")
        self.commandbool.value = True
        self.future = self.arming_service_client.call_async(self.commandbool)

    # Function to disarm the drone 
    def disarm(self):
        self.node.get_logger().info("Calling arm service")
        self.commandbool.value = False
        self.future = self.arming_service_client.call_async(self.commandbool)


def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node('controller')
    node.get_logger().info(f"Node Started")
    node.get_logger().info("Entering PID controller loop")

    controller = DroneController(node)
    controller.arm()
    node.get_logger().info("Armed")

    try:
        while rclpy.ok():
            controller.pid()
            if node.get_clock().now().to_msg().sec - controller.last_whycon_pose_received_at > 1:
                node.get_logger().error("Unable to detect WHYCON poses")
            rclpy.spin_once(node)
        

    except Exception as err:
        print(err)

    finally:
        controller.shutdown_hook()
        node.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    #calling the main function
    main()          