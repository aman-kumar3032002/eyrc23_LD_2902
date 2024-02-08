'''
# Team ID:          2902
# Theme:            Luminosity Drone
# Author List:      Aman Kumar, Shivam Kumar, Manila Raj Putra, Harsh Gulzar, 
# Filename:         LD_2902_waypoint_controller.py
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
import cv2 as cv
from rclpy.node import Node
from rclpy.clock import Clock
from geometry_msgs.msg import PoseArray
from pid_msg.msg import PidTune
from swift_msgs.msg import PIDError, RCMessage
from swift_msgs.srv import CommandBool
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


# battery voltage 12.5 - 12 v
#global variable for roll ----------------------------------------------------------------------------------------------
MIN_ROLL = 1400                       
BASE_ROLL = 1487
# BASE_ROLL = [1488,0,0]
MAX_ROLL = 1600
SUM_ERROR_ROLL_LIMIT =100

#gloabal vaiables for pitch---------------------------------------------------------------------------------------------
MIN_PITCH = 1400 
BASE_PITCH = 1490
# BASE_PITCH =[1490,0,0]
MAX_PITCH = 1600
SUM_ERROR_PITCH_LIMIT = 100

#gloabal vaiable for throttle-------------------------------------------------------------------------------------------
MIN_THROTTLE = 1400
BASE_THROTTLE = 1453
# BASE_THROTTLE = [1453,0,0]
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
        self.node = node
        
        self.cvBridge = CvBridge()
        self.rc_message = RCMessage()                                                                                           #self.rc_message                   : object of the the RCMessage class
        self.drone_whycon_pose_array = PoseArray()                                                                              #self.drone_whycon_pose_array      : object of the PoseArray class
        self.last_whycon_pose_received_at = 0                                                                                   #self.last_whycon_poses_received_at: stores the last_whycon_pose_received location
        self.commandbool = CommandBool.Request()                                                                                #self.commandbool                  : object of the CommandBool.Request class
        service_endpoint = "/swift/cmd/arming"
        self.arming_service_client = self.node.create_client(CommandBool,service_endpoint)
        
        #the set point at index 4 and 5 ie , [0,0,26] and [0,0,27] are given for landing purposes-------------------------------
        self.set_points = [[0,0,23],[2,3,23],[-1,2,25],[-3,-3,25],[0,0,26],[0,0,27]]                                            # Setpoints for x, y, z respectively      
        #-----------------------------------------------------------------------------------------------------------------------
        self.drone_position = [0.0, 0.0, 0.0]                                                                                   #self.drone_postion: stores the current position of the drone
        self.error      = [0.0, 0.0, 0.0]                                                                                       #self.error        : list, of error for roll, pitch and throttle respectively  
        self.prev_error = [0.0, 0.0, 0.0]                                                                                       #self.prev_error   : list to store the previous errors
        self.error_diff = [0.0, 0.0, 0.0]                                                                                       #self.error_diff   : stores the error difference as a list
        self.error_sum  = [0.0, 0.0, 0.0]                                                                                       #self.error_sum    : store the error_sum as a list
        self.current_setpoint_index = 0
        self.islanded = False                                                                                                   #self.islanded     : variable to store the curent state of drone landing
        
        #Defining the PID constants for battery level ~12.5 to  ~12.1               
        #values for the PID----------------------------------------------------------------------------------------------------
        self.Kp = [ 4.39, 4.9 ,4.53]                                                                                            #self.kp: stores the Kp values for all three axis - [roll,pitch,throttle]
        self.Ki = [ 0.004, 0.004, 0.00885]                                                                                      #self.ki: stores the Ki values for all three axis - [roll,pitch,thorttle]   
        self.Kd = [80.1, 110.1, 117.8]                                                                                          #self.kd: stores the kd values for all three axis - [roll,pitch,throttle]                   
        #----------------------------------------------------------------------------------------------------------------------

        #subscriber for WhyCon--------------------------------------------------------------------------------------        
        self.whycon_sub = node.create_subscription(PoseArray,"/whycon/poses",self.whycon_poses_callback,1)                      #self.pid__whycon_sub: subscriber for the whycon poses
        #subscriber for pid_roll------------------------------------------------------------------------------------
        self.pid_roll   = node.create_subscription(PidTune,"/pid_tuning_roll",self.pid_tune_roll_callback,1)                    #self.pid_roll   : subscriber for roll axis
        #subscriber for pid_pitch-----------------------------------------------------------------------------------
        self.pid_pitch  = node.create_subscription(PidTune,"/pid_tuning_pitch",self.pid_tune_pitch_callback,1)                  #self.pid_pitch  : subscriber for pitch axis
        #subscriber for pid_alt-------------------------------------------------------------------------------------
        self.pid_alt    = node.create_subscription(PidTune,"/pid_tuning_throttle",self.pid_tune_throttle_callback,1)            #self.pid_alt    : subscriber for the alt axis

        #subscriber for Rpi ---------
        self.alien_finder = node.create_subscription(Image, "/video_frames", self.image_callback,10)
        

        #Publisher for publishing errors for plotting in plotjuggler------------------------------------------------        
        self.pid_error_pub = node.create_publisher(PIDError, "/luminosity_drone/pid_error",1)                                   #self.pid_error_pub: publisher to publish pid_error    
        self.rc_pub = node.create_publisher(RCMessage, "/swift/rc_command",1)                                                   #self.rc_pub: publisher to publish rc_command


        
    def whycon_poses_callback(self, msg):
        self.last_whycon_pose_received_at = self.node.get_clock().now().seconds_nanoseconds()[0]
        self.drone_whycon_pose_array = msg
        self.drone_position[0] = msg.poses[0].position.x              
        self.drone_position[1] = msg.poses[0].position.y
        self.drone_position[2] = msg.poses[0].position.z
        
 
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

    def image_callback(self, img_msg):
        
        # cv_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")
        # # Process the image using OpenCV
        # cv2.imshow("Raspberry Pi Camera", cv_image)
        # cv2.waitKey(1)
        self.cv_image = self.bridge.imgmsg_to_cv2(img_msg,"passthrough")                        #converting the ros images to openc cv images

        grayscale_image = cv.cvtColor(self.cv_image,cv.COLOR_BGR2GRAY)                          #grayscaling the images
        blurred_image = cv.GaussianBlur(grayscale_image, (5, 5), 0)                             #blurring the images
        threshold = cv.threshold(blurred_image, 225, 255, 0)[1] 
        self.contours = cv.findContours(threshold,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2]      
        for c,contour in enumerate(self.contours):
            # (frame_centre_x,frame_centre_y),radius = cv.minEnclosingCircle(contour)
            # center = (int(frame_centre_x),int(frame_centre_y))
            cv.drawContours(self.cv_image,[contour],-1,(0,0,255),3)   
            a,b,c,d = cv.boundingRect(cv.drawContours(np.zeros_like(grayscale_image),self.contours,-1,1))
            cv.rectangle(self.cv_image,(a-4,b-4),(a+c+10,b+d+10),(0,0,255),2)                  #drawing a recatangle around the countours(organisms)
            contour_X = (a+c/2)/100                                                            #contour_x , stores the x coordinate of the contour, (divided by 100 to get a sigle digit value)
            contour_Y = (b+d/2)/100                                                            #contour_y , stores the y coordinate of the contour, (divided by 100 to get a sigle digit value)
            self.organism_centroids= [contour_X,contour_Y]                                     #storing the organism centroid
        

    def orgainsm_type(self,number_of_contours):
        """
    Purpose:
    -------------------------------------------
    converts the number of contours to the type of organism
    
    Input Arguments:
    -------------------------------------------
    number of contours
    
    Returns:
    -------------------------------------------
    type of organism(string)
    
    Example Call:
    -------------------------------------------
    organism_type(len(num_of_contours))
        """
        if number_of_contours ==2:
            self.alien.organism_type = "alien_a"
        elif number_of_contours == 3:
            self.alien.organism_type = "alien_b"
        elif number_of_contours == 4:
            self.alien.organism_type = "alien_c"             
        return self.alien.organism_type
            

    
    def drone_landing(self):
        
        self.error[0] = self.drone_position[0] - self.set_points[self.current_setpoint_index][0] 
        self.error[1] = self.drone_position[1] - self.set_points[self.current_setpoint_index][1]
        
        self.error_sum[0] = (self.error_sum[0] + (self.error[0]))                                                         #self.error_sum[0]:sum of the errors for roll
        self.error_sum[1] = (self.error_sum[1] + (self.error[1]))                                                         #self.error_sum[1]:sum of the errors for pitch
        
        if self.error_sum[0] > SUM_ERROR_ROLL_LIMIT:
            self.error_sum[0] = SUM_ERROR_ROLL_LIMIT
        if self.error_sum[0] < -SUM_ERROR_ROLL_LIMIT:
            self.error_sum[0] = -SUM_ERROR_ROLL_LIMIT
        if self.error_sum[1] > SUM_ERROR_PITCH_LIMIT:
            self.error_sum[1] = SUM_ERROR_PITCH_LIMIT
        if self.error_sum[1] < -SUM_ERROR_PITCH_LIMIT:
            self.error_sum[1] = -SUM_ERROR_PITCH_LIMIT
        
        self.error_diff[0] = (self.error[0]-self.prev_error[0])   
        self.error_diff[1] = (self.error[1]-self.prev_error[1])
        
        self.rc_message.rc_roll     = BASE_ROLL - int((self.Kp[0]*self.error[0])+(self.Kd[0]*self.error_diff[0])+(self.error_sum[0]*self.Ki[0]))		#self.rc_message_rc_roll: speed for the roll 
        self.rc_message.rc_pitch    = BASE_PITCH + int((self.Kp[1]*self.error[1])+(self.Kd[1]*self.error_diff[1])+(self.error_sum[0]*self.Ki[0]))		#self.rc_message_rc_pitch: speed for the pitch
        self.rc_message.rc_throttle = 1440                                                                                                              #sending constant value to the throttle
        
        self.prev_error[0] = self.error[0]                                                                                                              #self.prev_error[0]:storing the current error[0] as prev_error[0]
        self.prev_error[1] = self.error[1] 
    
        self.publish_data_to_rpi( roll = self.rc_message.rc_roll, pitch = self.rc_message.rc_pitch, throttle = self.rc_message.rc_throttle) 
        if -0.6<self.error[0]<0.6 and -0.6<self.error[1]<0.6 and self.drone_position[2]< 27:
            self.disarm()
            #setting self.islanded to True if the drone lands
            self.islanded = True             

    
    # --------------------------------------------------------------------------------PID algorithm----------------------------------------------------------------------------
    def pid(self):          
        try:
            self.error[0] = self.drone_position[0] - self.set_points[self.current_setpoint_index][0]                                                       #self.erorr[0]: calculating the error of the drone in x axis
            self.error[1] = self.drone_position[1]-  self.set_points[self.current_setpoint_index][1]                                                       #self.erorr[1]: calculating the error of the drone in y axis
            self.error[2] = self.drone_position[2] - self.set_points[self.current_setpoint_index][2]                                                       #self.erorr[2]: calculating the error of the drone in z axis
            #----------------error of all the cordinates(integral)---------------------------------------------------------------------------------------------
            self.error_sum[0] = (self.error_sum[0] + (self.error[0]))                                                         #self.error_sum[0]:sum of the errors for roll
            self.error_sum[1] = (self.error_sum[1] + (self.error[1]))                                                         #self.error_sum[1]:sum of the errors for pitch
            self.error_sum[2] = (self.error_sum[2] + (self.error[2]))                                                         #self.error_sum[2]:sum of the errors for throttle
            
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
                    yaw_error=-0.0,                                                                                         
                    zero_error=0.0,
                    )
                )
            if self.current_setpoint_index < (len(self.set_points)-1):
                if -0.8<self.error[0]<0.8 and -0.8<self.error[1]<0.8 and -0.8<self.error[2]<0.8:
                    # print(f"setpoint {self.set_points[self.current_setpoint_index]} is achieved")
                    self.current_setpoint_index += 1
                
        except Exception as e:
            print(e)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    def publish_data_to_rpi(self, roll, pitch, throttle):
        self.rc_message.rc_throttle = int(throttle)                                                                           #self.rc_message.rc_throttle: storing the integer value of the throttle
        self.rc_message.rc_roll     = int(roll)                                                                               #self.rc_message.rc_roll    : storing the integer value of the roll
        self.rc_message.rc_pitch    = int(pitch)                                                                              #self.rc_message.rc_pitch   : storing the integer value of the pitch       
        self.rc_message.rc_yaw      = int(1500)                                                                               #self.rc_message.rc_yaw     : storing the constant value to yaw

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
        self.node.get_logger().info("Calling disarm service")
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
            #if the setpoint is set to the last set_point, then the drone landing function will run
            if controller.current_setpoint_index == (len(controller.set_points)-1) and controller.islanded == False:
                #calling the drone_landing  function------------
                controller.drone_landing()
            else:
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