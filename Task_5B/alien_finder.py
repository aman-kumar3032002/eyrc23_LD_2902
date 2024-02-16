#!/usr/bin/env python3
'''
# Team ID:          2902
# Theme:            Luminosity Drone
# Author List:      Aman Kumar, Shivam Kumar, Manila Raj Putra, Harsh Gulzar 
# Filename:         LD_2902_alien_finder.py
# Functions:        __init__,whycon_poses_callback(),pid_tune_roll_callback(),pid_tune_pitch_callback(),pid_tune_throttle_callback(),pid(),publish_data_to_rpi,shutdown_hook(),arm(),disarm(),main(),image_callback(),organism_type()

# Global variables: MIN_ROLL,BASE_ROLL,MAX_ROLL,SUM_ERROR_ROLL_LIMIT,MIN_ROLL_PITCH,BASE_PITCH,MAX_PITCH, SUM_ERROR_PITCH_LIMIT, MIN_THROTTLE,BASE_THROTTLE,MAX_THROTTLE,SUM_ERROR_THROTTLE_LIMIT,BASE_YAW
'''
# standard imports
import copy
import time
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup,ReentrantCallbackGroup
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
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2 as cv
from swift_msgs.msg import *
from loc_msg.msg import Biolocation


#global variable for roll ----------------------------------------------------------------------------------------------
MIN_ROLL = 1420                    
BASE_ROLL = 1490
MAX_ROLL = 1550
SUM_ERROR_ROLL_LIMIT =1000

#gloabal vaiables for pitch---------------------------------------------------------------------------------------------
MIN_PITCH = 1420
BASE_PITCH = 1480
MAX_PITCH = 1550
SUM_ERROR_PITCH_LIMIT = 1000

#gloabal vaiable for throttle-------------------------------------------------------------------------------------------
MIN_THROTTLE = 1400
BASE_THROTTLE = 1453
MAX_THROTTLE= 1550
SUM_ERROR_THROTTLE_LIMIT = 8000

#gloabal variable for yaw-----------------------------------------------------------------------------------------------
MIN_YAW= 1500
BASE_YAW = 1500
MAX_YAW = 1500
SUM_ERROR_ROLL_LIMIT = 400
#-----------------------------------------------------------------------------------------------------------------------
DRONE_WHYCON_POSE = [[], [], []]

class DroneController(Node):
    def __init__(self,node):
        self.node= node        
        my_mutual_group_1 = MutuallyExclusiveCallbackGroup()
        my_reentrant_group = ReentrantCallbackGroup() 
        my_mutual_group_2 = MutuallyExclusiveCallbackGroup()
        
        self.rc_message = RCMessage()                                                                                           #self.rc_message                   : object of the the RCMessage class
        self.drone_whycon_pose_array = PoseArray()                                                                              #self.drone_whycon_pose_array      : object of the PoseArray class
        self.last_whycon_pose_received_at = 0                                                                                   #self.last_whycon_poses_received_at: stores the last_whycon_pose_received location
        self.commandbool = CommandBool.Request()                                                                                #self.commandbool                  : object of the CommandBool.Request class
        service_endpoint = "/swift/cmd/arming"
        self.arming_service_client = self.node.create_client(CommandBool,service_endpoint)
        self.bridge =  CvBridge()
  
        self.set_points = [[3.6, 6.6, 22],[-0.3, 6.5, 22],[-3.7, 6.5, 22],
                           [-7.3, 3.1,22],[-7.4,-0.3,22],[-7.3, -3.7, 22],[-7.3,-7.1,22],
                           [-3.6, -7.1,22],[-3.6, -3.7, 22],[-3.6, -0.3,22],[-3.6, 3.0,22],
                           [-0.0, 3.1, 22],[-0.0, -0.3,22],[-0.1, -3.7, 22],[-0.0, -7.3,22],
                           [3.7, -7.3,22],[7.2, -7.1, 22],[7.4, -3.7, 22],[3.6, -3.7,22],
                           [3.6, -0.3, 22],[7.3, -0.3,22],[7.2, 3.1,22],[3.6, 3.1,22],
                           [7.2, 6.5, 25],[7.2,6.5,27],[8.3,7.3,27],[8.3,7.3,28]]    
                                          
        self.drone_position = [8.0, 7.0, 27.0]                                                                                  #self.drone_postion: stores the current position of the drone
        #-----------------------------------------------------------------------------------------------------------------------
        self.error      = [0.0, 0.0, 0.0]                                                                                       #self.error        : list, of error for roll, pitch and throttle respectively  
        self.prev_error = [0.0, 0.0, 0.0]                                                                                       #self.prev_error   : list to store the previous errors
        self.error_diff = [0.0, 0.0, 0.0]                                                                                       #self.error_diff   : stores the error difference as a list
        self.error_sum  = [0.0, 0.0, 0.0]                                                                                       #self.error_sum    : store the error_sum as a list
        self.current_setpoint_index = 0                                                                                         #self.current_setpoint_index : Stores the index of current setpoint
        
        self.islanded = False                                                                                                   #self.islanded     : variable to store the curent state of drone landing
        self.contours = []                                                                                                      #self.contours: stores the number of contours        
        self.alien = Biolocation()                                                                                              #self.alien: making an object of Biolocation()
        self.alien.organism_type = ''                                                                                           #self.organism_type: stores the organism type
        self.alien.whycon_x = self.drone_position[0]                                                                            # whycon_x, stores the current x coordinate of the drone
        self.alien.whycon_y = self.drone_position[1]                                                                            # whycon_y, stores the current y coordinate of the drone
        self.alien.whycon_z = self.drone_position[2]                                                                            # whycon_z, stores the current z coordinate of the drone
        self.dividing_factor = 100                                                                                              #self.dividing_factor : Stores the dividing factor used for normalizing the organism centroid and frame centroid
        self.frame_centroids = [63,63]                                                                                          #self.frame_centroid: centroid of the camera frame
        self.min_centroid_error = -0.08                                                                                         #stores the min value of the error, required to publish the coordinates
        self.max_centroid_error = 0.08                                                                                          #stores the max value of the error , required to publish the coordinates
        self.organism_centroids  = [0,0]                                                                                        #self.organism_centroid: stroes the centroid of the detected cluster
        self.isPublished = 0                                                                                                    #self.isPpublished: checks whether message is published or not
        
        self.rc_message.rc_pitch = 0                                                                                            #self.rc_message.rc_pitch: stores the value pitch 
        self.rc_message.rc_roll = 0                                                                                             #self.rc_message.rc_roll: store the value of roll
        self.rc_message.rc_throttle = 0                                                                                         #self.rc_message.rc_throttle: stores the value of the throttle                
        #values for the PID----------------------------------------------------------------------------------------------------
        self.Kp = [5.58, 5.80 ,3.80]                                                                                            #self.kp: stores the Kp values for all three axis - [roll,pitch,throttle]
        self.Ki = [0.051, 0.0318, 0.080]                                                                                        #self.ki: stores the Ki values for all three axis - [roll,pitch,thorttle]   
        self.Kd = [180.1, 180.1, 150.8]                                                                                         #self.kd: stores the kd values for all three axis - [roll,pitch,throttle]                   
        #----------------------------------------------------------------------------------------------------------------------

        #subscriber for WhyCon--------------------------------------------------------------------------------------        
        self.whycon_sub = node.create_subscription(PoseArray,"/whycon/poses",self.whycon_poses_callback,10, callback_group = my_mutual_group_1)                      #self.pid__whycon_sub: subscriber for the whycon poses
        #subscriber for pid_roll------------------------------------------------------------------------------------
        self.pid_roll   = node.create_subscription(PidTune,"/pid_tuning_roll",self.pid_tune_roll_callback,1)                    #self.pid_roll   : subscriber for roll axis
        #subscriber for pid_pitch-----------------------------------------------------------------------------------
        self.pid_pitch  = node.create_subscription(PidTune,"/pid_tuning_pitch",self.pid_tune_pitch_callback,1)                  #self.pid_pitch  : subscriber for pitch axis
        #subscriber for pid_alt-------------------------------------------------------------------------------------
        self.pid_alt    = node.create_subscription(PidTune,"/pid_tuning_throttle",self.pid_tune_throttle_callback,1)            #self.pid_alt    : subscriber for the alt axis
        #subscribing to /video_frames-------------------------------------------------------------------------------
        self.alien_subscriber = node.create_subscription(Image,"/video_frames", self.image_callback,10,callback_group = my_mutual_group_2)


        #Publisher for publishing errors for plotting in plotjuggler------------------------------------------------        
        self.pid_error_pub = node.create_publisher(PIDError, "/luminosity_drone/pid_error",1)                                   #self.pid_error_pub: publisher to publish pid_error    
        self.rc_pub = node.create_publisher(RCMessage, "/swift/rc_command",1)                                                   #self.rc_pub: publisher to publish rc_command
        #creating the publisher for publishing on /astrobiolocation-------------------------------------------------
        self.astrobiolocation_pub = node.create_publisher(Biolocation,'/astrobiolocation',1)               
        #publishing the rc_command at 30hz--------------------------------------------------------------------------
        node.create_timer(0.0333, self.pid,callback_group = my_mutual_group_1)    

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
        """
    Purpose:
    -------------------------------------------
    callback function for deploying image processing for detection of the organism on the planet
    
    Input Arguments:
    -------------------------------------------
    img_msg
    
    Returns:
    -------------------------------------------
    None
    
    Example Call:
    -------------------------------------------
    image_callback(img_msg)
        """
        self.cv_image = self.bridge.imgmsg_to_cv2(img_msg)                                              #converting the ros images to openc cv images
        scale_percent = 20                                                                              #reducing image quality by 20% of the acutual quality
        width = int(self.cv_image.shape[1]*scale_percent/100)                                           #resizing width 
        height = int(self.cv_image.shape[0]*scale_percent/100)                                          #resizing height
        dim = (width,height)
        #resizing the image-----------------------------------------------------
        self.resized = cv.resize(self.cv_image,dim,interpolation = cv.INTER_AREA)  
        
        grayscale_image = cv.cvtColor(self.resized,cv.COLOR_BGR2GRAY)                                 #grayscaling the images
        blurred_image = cv.GaussianBlur(grayscale_image, (5, 5), 0)                                        #blurring the images
        threshold = cv.threshold(blurred_image, 225, 255, 0)[1] 
        self.contours = cv.findContours(threshold,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2] 
        for width,contour in enumerate(self.contours):
            #drawing contours---------------------------------------------------
            cv.drawContours(self.resized,[contour],-1,(0,0,255),3)    
            x_coordinate,y_coordinate,width,height = cv.boundingRect(cv.drawContours(np.zeros_like(grayscale_image),self.contours,-1,1))
            cv.rectangle(self.resized,(x_coordinate-4,y_coordinate-4),(x_coordinate+width+10,y_coordinate+height+10),(0,0,255),2)                   #drawing a recatangle around the countours(organisms)
            contour_X = (x_coordinate+width/2)                                                                                                      #contour_x , stores the x coordinate of the contour, 
            contour_Y = (y_coordinate+height/2)                                                                                                     #contour_y , stores the y coordinate of the contour,
            self.organism_centroids= [contour_X,contour_Y]  
            
        # cv.imshow("Raspberry Pi Camera", self.resized)
        # cv.waitKey(1)
    
    def organism_type(self,number_of_contours):
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
        #if number of contours is 2, then the function will return "alien_a"
        if number_of_contours ==2:
            self.alien.organism_type = 'alien_a'
        #if number of contours is 3, then the function will return "alien_b"
        elif number_of_contours == 3:
            self.alien.organism_type = 'alien_b'
        #if number of contours is 4, then the function will return "alien_c"
        elif number_of_contours ==4:
            self.alien.organism_type = 'alien_c'
        #if number of contours is 5, then the function will return "alien_d"
        elif number_of_contours == 5:
            self.alien.organism_type = 'alien_d'
        return self.alien.organism_type

    # --------------------------------------------------------------------------------PID algorithm----------------------------------------------------------------------------
    def pid(self): 
        """
    Purpose:
    -------------------------------------------
    stablises the drone at a setpoint
    
    Input Arguments:
    -------------------------------------------
    None
    
    Returns:
    -------------------------------------------
    None
    
    Example Call:
    -------------------------------------------
    pid()
        """         
        try:
            #if no contour is detected then the drone will move according to its setpoints
            if(len(self.contours)<1):
                self.error[0] = self.drone_position[0] - self.set_points[self.current_setpoint_index][0]                                                       #self.erorr[0]: calculating the error of the drone in x axis
                self.error[1] = self.drone_position[1] - self.set_points[self.current_setpoint_index][1]                                                       #self.erorr[1]: calculating the error of the drone in y axis
                self.error[2] = self.drone_position[2] - self.set_points[self.current_setpoint_index][2]                                                       #self.erorr[2]: calculating the error of the drone in z axis
                self.isPublished = 0
            #if a contour is detected, the drone will try to align the centroid of the camera frame to the centroid of the cluster
            if(len(self.contours)>1):
                self.error[0] = ((self.frame_centroids[0]/self.dividing_factor) - (self.organism_centroids[0]/self.dividing_factor))
                self.error[1] = ((self.frame_centroids[1]/self.dividing_factor) - (self.organism_centroids[1]/self.dividing_factor))                         
                self.error[2] = self.drone_position[2] - self.set_points[self.current_setpoint_index][2]
            #if contour is detected and msg is published
            if(len(self.contours)>1 and self.isPublished == 1):
                self.error[0] = self.drone_position[0] - self.set_points[self.current_setpoint_index][0] 
                self.error[1] = self.drone_position[1] - self.set_points[self.current_setpoint_index][1] 
                self.error[2] = self.drone_position[2] - self.set_points[self.current_setpoint_index][2]
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
            self.rc_message.rc_pitch    = BASE_PITCH + int((self.Kp[1]*self.error[1])+(self.Kd[1]*self.error_diff[1])+(self.error_sum[1]*self.Ki[1]))		#self.rc_message_rc_pitch: speed for the pitch
            self.rc_message.rc_throttle = BASE_THROTTLE + int((self.Kp[2]*self.error[2])+(self.Kd[2]*self.error_diff[2])+(self.error_sum[2]*self.Ki[2]))   #self.rc_message_rc_throttle: speed for the throttle 
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
            if (self.current_setpoint_index < (len(self.set_points)-1) and len(self.contours)<1):
                if -0.6<self.error[0]<0.6 and -0.6<self.error[1]<0.6 and -0.6<self.error[2]<0.6:
                    print(f"setpoint {self.set_points[self.current_setpoint_index]} is achieved")
                    self.current_setpoint_index += 1          
            
            if((self.min_centroid_error< self.error[0]<self.max_centroid_error) and (self.min_centroid_error< self.error[1]< self.max_centroid_error) and len(self.contours)> 0 and self.isPublished == 0):
                self.alien.organism_type = self.organism_type(len(self.contours))                                                                           #self.alien.organism_type: stores the result of the oraganism_type function
                self.alien.whycon_x = self.drone_position[0]                                                                                                #stores the whycon-x of the drone
                self.alien.whycon_y = self.drone_position[1]                                                                                                #stores the whycon-y of the drone
                self.alien.whycon_z = self.drone_position[2]                                                                                                #stores the whycon-z of the drone
                #publishing on astrobiolocation-----------------------------------------------------
                self.astrobiolocation_pub.publish(self.alien)  
                #updating the variable isPublished if /astrobiolocation is published----------------
                self.isPublished = 1
                #updating the setpoint if /astrobiolocation is published----------------------------
                self.current_setpoint_index += 1 
                #printing if alien is detected------------------------------------------------------
                print("cute alien detected")   
                    
            #if the drone reaches the last setpoint, then the drone will disarm-----------------
            if((len(self.set_points)-1) == self.current_setpoint_index):
                self.islanded = True
                self.disarm()

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
    executor = MultiThreadedExecutor()
    node = rclpy.create_node('controller')
    executor.add_node(node)
    node.get_logger().info(f"Node Started")
    node.get_logger().info("Entering PID controller loop")
    controller = DroneController(node)
    controller.arm()
    node.get_logger().info("Armed")
    
    try:           
        while rclpy.ok():
            if controller.islanded == False:
                controller.pid()
            if node.get_clock().now().to_msg().sec - controller.last_whycon_pose_received_at > 1:
                node.get_logger().error("Unable to detect WHYCON poses")           
            executor.spin()
            
            
    except Exception as err:
        print(err)

    finally:
        controller.shutdown_hook()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    #calling the main function
    main()          
