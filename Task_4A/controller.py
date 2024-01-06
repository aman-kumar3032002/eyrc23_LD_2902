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
from swift_msgs import Float64

MIN_ROLL = 1000
BASE_ROLL = 1500
MAX_ROLL = 2000
SUM_ERROR_ROLL_LIMIT = 100

MIN_PITCH = 1000
BASE_PITCH = 1500
MAX_PITCH = 2000
SUM_ERROR_PITCH_LIMIT = 100

MIN_THROTTLE = 1000
BASE_THROTTLE = 1500
MAX_THROTTLE = 2000
SUM_ERROR_THROTTLE_LIMIT = 100

MIN_YAW = 1500
BASE_YAW = 1500
MAX_YAW = 1500
SUM_ERROR_YAW_LIMIT = 100


DRONE_WHYCON_POSE = [[], [], []]

# Similarly, create upper and lower limits, base value, and max sum error values for roll and pitch

class DroneController():
    def __init__(self,node):
        self.node= node
        
        self.rc_message = RCMessage()
        self.drone_whycon_pose_array = PoseArray()
        self.last_whycon_pose_received_at = 0
        self.commandbool = CommandBool.Request()
        service_endpoint = "/swift/cmd/arming"

        self.arming_service_client = self.node.create_client(CommandBool,service_endpoint)
        self.drone_position = [0.0, 0.0, 0.0]                                                                                   #self.drone_postion: stores the current position of the drone
        self.set_points = [0, 0, 22]                                                                                            #self.set_points   : list of setpoints, for x, y, z respectively                     
        self.error      = [0.0, 0.0, 0.0]                                                                                       #self.error        : list, of error for roll, pitch and throttle respectively  
        self.prev_error = [0.0, 0.0, 0.0]                                                                                       #self.prev_error   : list to store the previous errors
        self.error_diff = [0.0, 0.0, 0.0]                                                                                       #self.error_diff   : list to store the error difference
        self.error_sum  = [0.0, 0.0, 0.0]                                                                                       #self.error_sum    : list to store the sum of the errors

        #pid variables---------------------------------------------------------------------------------------------------------------------------------------------------------
        self.Kp = [ 0 * 0.01  , 0 * 0.01  , 0 * 0.01  ]
        self.Kd = [ 0 * 0.01  , 0 * 0.01  , 0 * 0.01  ]
        self.Ki = [ 0 * 0.01  , 0 * 0.01  , 0 * 0.01  ]
        #----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        #----------------------------------------------------------------Subscribers-------------------------------------------------------------------------------------------
        self.whycon_sub = node.create_subscription(PoseArray,"/whycon/poses",self.whycon_poses_callback,1)                      #self.whycon_sub : subscriber for the whycon 
        self.pid_roll   = node.create_subscription(PidTune,"/pid_tuning_altitude",self.pid_tune_roll_callback,1)                #self.pid_roll   : subscriber for roll axis
        self.pid_pitch  = node.create_subscription(PidTune,"/pid_tuning_altitude",self.pid_tune_pitch_callback,1)               #self.pid_pitch  : subscriber for pitch axis
        self.pid_alt    = node.create_subscription(PidTune,"/pid_tuning_altitude",self.pid_tune_throttle_callback,1)            #self.pid_alt    : subscriber for the alt axis
        
        #----------------------------------------------------------------------------------------------------------------------------------------------------------------------
                
        #Publisher for sending commands to drone-------------------------------------------------------------------------------------------------------------------------------
        self.rc_pub = node.create_publisher(RCMessage, "/swift/rc_command",1)
        
        #Creating publisher for publishing errors for plotting in plotjuggler--------------------------------------------------------------------------------------------------       
        self.pid_error_pub       = node.create_publisher(PIDError, "/luminosity_drone/pid_error",1)                             #self.pid_error_pub: publisher for pid error   
        self.pid_roll_error_pub  = node.create_publisher(Float64,"/alt_error",1)                                                #self.pid_error_pub: publisher for roll error 
        self.pid_pitch_error_pub = node.create_publisher(Float64,"/alt_error",1)                                                #self.pid_error_pub: publisher for pitch error 
        self.pid_alt_error_pub   = node.create_publisher(Float64,"/alt_error",1)                                                #self.pid_error_pub: publisher for alt error 
               
    def whycon_poses_callback(self, msg):
        self.last_whycon_pose_received_at = self.node.get_clock().now().seconds_nanoseconds()[0]
        self.drone_whycon_pose_array = msg
        
        self.drone_position[0] =self.drone_whycon_pose_array.poses[0].position.x
        self.drone_position[1] =self.drone_whycon_pose_array.poses[0].position.y
        self.drone_position[2] =self.drone_whycon_pose_array.poses[0].position.z

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

    def pid(self):        
        # try:
        #     self.error[0] = self.drone_position[0] - self.set_points[0] 
        #     self.error[1] = self.drone_position[1] - self.set_points[1] 
        #     self.error[2] = self.drone_position[2] - self.set_points[2] 

        #     self.error_sum[0] = self.error_sum[0] + self.error[0]
        #     self.error_sum[1] = self.error_sum[1] + self.error[1]
        #     self.error_sum[2] = self.error_sum[2] + self.error[2]

        #     self.error_diff[0] = self.error[0] - self.prev_error[0]
        #     self.error_diff[1] = self.error[1] - self.prev_error[1]
        #     self.error_diff[2] = self.error[2] - self.prev_error[2]           
        # except:
        #     pass
            
        #error of all the cordinates(for proportional)-------------------   
        self.error[0] = self.drone_position[0] - self.set_points[0] 
        self.error[1] = self.drone_position[1] - self.set_points[1] 
        self.error[2] = self.drone_position[2] - self.set_points[2] 

        #error of all the cordinates(integral)---------------------------
        self.error_sum[0] = self.error_sum[0] + self.error[0]
        self.error_sum[1] = self.error_sum[1] + self.error[1]
        self.error_sum[2] = self.error_sum[2] + self.error[2]
        
        if self.integral[0] > SUM_ERROR_ROLL_LIMIT:
            self.integral[0] = SUM_ERROR_ROLL_LIMIT
        if self.integral[0] < -SUM_ERROR_ROLL_LIMIT:
            self.integral[0] = -SUM_ERROR_ROLL_LIMIT

        if self.integral[1] > SUM_ERROR_PITCH_LIMIT:
            self.integral[1] = SUM_ERROR_PITCH_LIMIT
        if self.integral[1] < -SUM_ERROR_PITCH_LIMIT:
            self.integral[1] = -SUM_ERROR_PITCH_LIMIT

        if self.integral[2] > SUM_ERROR_THROTTLE_LIMIT:
            self.integral[2] = SUM_ERROR_THROTTLE_LIMIT
        if self.integral[2] < -SUM_ERROR_THROTTLE_LIMIT:
            self.integral[2] = -SUM_ERROR_THROTTLE_LIMIT

        #error of all the cordinates(derivative)-------------------------
        self.error_diff[0] = self.error[0] - self.prev_error[0]
        self.error_diff[1] = self.error[1] - self.prev_error[1]
        self.error_diff[2] = self.error[2] - self.prev_error[2] 

        #saving current error in previous error------------------------
        self.prev_error[0] = self.error[0]
        self.prev_error[1] = self.error[1]
        self.prev_error[2] = self.error[2]

        # 1 : calculating Error, Derivative, Integral for Pitch error : y axis

        # 2 : calculating Error, Derivative, Integral for Alt error : z axis


        # Write the PID equations and calculate the self.rc_message.rc_throttle, self.rc_message.rc_roll, self.rc_message.rc_pitch
        self.rc_message.rc_roll     = BASE_ROLL - int((self.Kp[0]*self.error[0])+(self.Kd[0]*self.error_diff[0]))		#roll 
        self.rc_message.rc_pitch    = BASE_PITCH +int((self.Kp[1]*self.error[1])+(self.Kd[1]*self.error_diff[1]))		#pitch
        self.rc_message.rc_throttle = BASE_THROTTLE + int((self.Kp[2]*self.error[2])+ (self.Kd[2]*self.error_diff[2])+(self.error_sum[2]*self.Ki[2])) #throttle
   
        
    #------------------------------------------------------------------------------------------------------------------------


        self.publish_data_to_rpi(self.rc_message.rc_roll,self.rc_message.rc_pitch,self.rc_message.rc_throttle)
              
        # Publish alt error, roll error, pitch error for plotjuggler debugging
        self.pid_error_pub.publish(
            PIDError(
                roll_error=float(self.error[0]),
                pitch_error=float(self.error[1]),
                throttle_error=float(self.error[2]),
                yaw_error=-0.0,
                zero_error=0.0,
            )
        )


    def publish_data_to_rpi(self, roll, pitch, throttle):

        self.rc_message.rc_throttle = int(throttle)
        self.rc_message.rc_roll = int(roll)
        self.rc_message.rc_pitch = int(pitch)

        # Send constant 1500 to rc_message.rc_yaw
        self.rc_message.rc_yaw = int(BASE_YAW)

        # BUTTERWORTH FILTER
        # span = 15
        # for index, val in enumerate([roll, pitch, throttle]):
        #     DRONE_WHYCON_POSE[index].append(val)
        #     if len(DRONE_WHYCON_POSE[index]) == span:
        #         DRONE_WHYCON_POSE[index].pop(0)
        #     if len(DRONE_WHYCON_POSE[index]) != span-1:
        #         return
        #     order = 3
        #     fs = 60
        #     fc = 5
        #     nyq = 0.5 * fs
        #     wc = fc / nyq
        #     b, a = scipy.signal.butter(N=order, Wn=wc, btype='lowpass', analog=False, output='ba')
        #     filtered_signal = scipy.signal.lfilter(b, a, DRONE_WHYCON_POSE[index])
        #     if index == 0:
        #         self.rc_message.rc_roll = int(filtered_signal[-1])
        #     elif index == 1:
        #         self.rc_message.rc_pitch = int(filtered_signal[-1])
        #     elif index == 2:
        #         self.rc_message.rc_throttle = int(filtered_signal[-1])

        if self.rc_message.rc_roll > MAX_ROLL:     #checking range i.e. bet 1000 and 2000
            self.rc_message.rc_roll = MAX_ROLL
        elif self.rc_message.rc_roll < MIN_ROLL:
            self.rc_message.rc_roll = MIN_ROLL

        if self.rc_message.rc_pitch > MAX_PITCH:     #checking range i.e. bet 1000 and 2000
            self.rc_message.rc_pitch = MAX_PITCH
        elif self.rc_message.rc_pitch < MIN_PITCH:
            self.rc_message.rc_pitch = MIN_PITCH

        if self.rc_message.rc_throttle > MAX_THROTTLE:     #checking range i.e. bet 1000 and 2000
            self.rc_message.rc_throttle = MAX_THROTTLE
        elif self.rc_message.rc_throttle < MIN_THROTTLE:
            self.rc_message.rc_throttle = MIN_THROTTLE
            
        # Similarly add bounds for pitch yaw and throttle 

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

        # Create the disarm function

        pass


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
            rclpy.spin_once(node) # Sleep for 1/30 secs
        

    except Exception as err:
        print(err)

    finally:
        controller.shutdown_hook()
        node.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    main()