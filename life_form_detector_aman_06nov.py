#!/usr/bin/env python3
'''
Team ID: 2902
Theme: Luminosity Drone
Author List: Aman Kumar, Harsh gulzar, Manila Raj Putra, Shivam Kumar
Filename: LD_2902_position_hold.py
Functions: disarm(),arm(),whycon_callback(),altitude_set_pid(),pitch_set_pid(),roll_set_pid(),pid()
Global variables: None

This python file runs a ROS-node of name drone_control which holds the position of Swift-Drone on the given dummy.
This node publishes and subsribes the following topics:

        PUBLICATIONS			SUBSCRIPTIONS
        /drone_command			/whycon/poses
        /alt_error				/pid_tuning_altitude
        /pitch_error			/pid_tuning_pitch
        /roll_error				/pid_tuning_roll
                    
                                

Rather than using different variables, use list. eg : self.setpoint = [1,2,3], where index corresponds to x,y,z ...rather than defining self.x_setpoint = 1, self.y_setpoint = 2
CODE MODULARITY AND TECHNIQUES MENTIONED LIKE THIS WILL HELP YOU GAINING MORE MARKS WHILE CODE EVALUATION.	
'''

# Importing the required libraries

from swift_msgs.msg import *
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Int16
from std_msgs.msg import Int64
from std_msgs.msg import Float64
from pid_tune.msg import PidTune
from sensor_msgs.msg import Image
import cv2 as cv
from cv_bridge import CvBridge,CvBridgeError
from skimage import measure
import math
import rospy
import time
import numpy as np
from luminosity_drone.msg import Biolocation



class swift():
    """docstring for swift"""
    def __init__(self):
        
        rospy.init_node('drone_control')	                         # initializing ros node with name drone_control
        self.drone_position = [0.0,0.0,0.0]	
        self.bridge = CvBridge()

        # # [x_setpoint, y_setpoint, z_setpoint]

        # self.setpoint = [[0,0,0],[0,0,30],
        #                 [3,0,30],[3,3,30],[-3,3,30],[-3,-3,30],[0,-3,30],
        #                 [5,-3,30],[5,0,30],[5,5,30],[0,5,30],[-5,5,30],[-5,0,30],[-5,-5,30],[0,-5,30],
        #                 [7,-5,30],[7,0,30],[7,7,30],[0,7,30],[-7,7,30],[-7,0,30],[-7,-7,30],[0,-7,30],
        #                 [10,10,35],[11,11,37]]  
        # self.final_setpoint = [[9,9,30],[11,11,37]]
        
        self.setpoint = [[0,0,0],[0,0,30],[8,8,30],[8,0,30],[8,-8,30],[0,-8,30],[-8,-8,30],[-8,0,30],[-8,8,30],[0,8,30],[8,8,33],[11,11,37],[11,11,37]
                        ] 
         
        # self.setpoint = [[0,0,0],[0,0,30],[8,8,30],[8,-8,30],[0,-8,30],[-8,-8,30],[-8,8,30],[8,8,33],[11,11,37],[11,11,37]
        #                 ]          
        # self.setpoint = [[0,0,0],[0,0,30],
        #                 [4,0,30],[4,4,30],[-4,4,30],[-4,-4,30],[0,-4,30],
        #                 [6,-4,30],[6,0,30],[6,6,30],[0,6,30],[-6,6,30],[-6,0,30],[-6,-6,30],[0,-6,30],
        #                 [9,-6,30],[9,0,30],[9,9,30],[0,9,30],[-9,9,30],[-9,0,30],[-9,-9,30],[0,-9,30],
        #                 [9,9,30],[11,11,37.5],[11,11,37]] 
        # self.setpoint = [[0,0,0],[0,0,30],
        #                 [4,0,30],[4,4,30],[-4,4,30],[-4,-4,30],
        #                 [6,-4,30],[6,6,30],[-6,6,30],[-6,-6,30],
        #                 [9,-6,32],[9,9,32],[-9,9,32],[-9,0,32],[-9,-9,32],[0,-9,32],[9,-9,32],[9,0,32],
        #                 [9,9,32],[11,11,37.5],[11,11,37]]   
        # self.setpoint = [[0,0,0],[0,0,30],[0,0,35],
        #                 [6,-4,30],[6,6,30],[-6,6,30],[-6,-6,30],
        #                 [9,-6,30],[9,10,30],[-9,10,30],[-9,0,30],[-9,-10,30],[0,-10,30],[9,-10,30],[9,0,35],
        #                 [9,10,30],[11,11,37.5],[11,11,37]] 
        
        # self.setpoint = [[0,0,0],[0,0,30],
        #                 [2,0,30],[11,11,37]]                  
        #                 ]   # whycon marker at the position of the dummy given in the scene. Make the whycon marker associated with position_to_hold dummy renderable and make changes accordingly
        # self.setpoint =  [[0,0,0],[0,0,23],
        #                  [3,0,23],[6,0,23],[9,0,23]]   
        # self.setpoint = [[0,0,0],[0,0,23],[7,7,23],[8,8,35],[9,9,35],[11,11,37.5],[11,11,35]
        #                 ] 
        
        
        #defining the varibale for centroids of frame and the organisms                                                
        self.frame_centroid = [0,0]  
        self.organism_centroid = [0,0]     
                 
        #Declaring a cmd of message type swift_msgs and initializing values
        self.cmd = swift_msgs()
        self.cmd.rcRoll = 1500
        self.cmd.rcPitch = 1500
        self.cmd.rcYaw = 1500
        self.cmd.rcThrottle = 1000
        self.cmd.rcAUX1 = 1500
        self.cmd.rcAUX2 = 1500
        self.cmd.rcAUX3 = 1500
        self.cmd.rcAUX4 = 1500


        #initial setting of Kp, Kd and ki for [roll, pitch, throttle]. eg: self.Kp[2] corresponds to Kp value in throttle axis
        #after tuning and computing corresponding PID parameters, change the parameters
        self.Kp = [20,20,40]                                 #Kp values for roll,pitch,throttle
        self.Ki = [0,0,0.0029523]                            #Ki values for roll,pitch,throttle
        self.Kd = [30000,30000,46000]                        #Kd values for roll,pitch,throttle
   
        #-----------------------Add other required variables for pid here ----------------------------------------------
        self.error = [0.0,0.0,0.0]                 #variable to store error ,i.e differnce between set points and current cordinates
        self.prev_error = [0.0,0.0,0.0]            #variable to store previous error
        self.min_values = 1000                     #variable to store the maximum values of [roll,pitch,throttle]
        self.max_values = 2000                     #varibale to srtore the maximum valus of [roll,pitch,throttle]
        self.error_sum = [0.0,0.0,0.0]			   #varibale to store the sum of the errors
        self.error_diff = [0.0,0.0,0.0]			   #varibale to store the difference of the errore
        self.now = 0.00							   #varibale to store current time
        self.last_time = 0.0000					   #self.last_time = self.now
        self.time_change = 0.00					   #differnce between the current_time and last_time
        
        # # This is the sample time in which you need to run pid. Choose any time which you seem fit. Remember the stimulation step time is 50 ms
        self.sample_time = 0.033# in seconds


        #Declaring a cmd of message type Biolocation and initializing values
        self.alien = Biolocation()
        self.alien.organism_type = " " 
        self.flag = 0
        self.count = 0
        self.num_of_leds = 0       
        self.i = 0

        # Publishing /drone_command, /alt_error, /pitch_error, /roll_error
        self.command_pub = rospy.Publisher('/drone_command', swift_msgs, queue_size=1)
        #------------------------Add other ROS Publishers here-----------------------------------------------------
        self.alt_error_pub = rospy.Publisher('/alt_error',Float64, queue_size=1)    
        self.pitch_error_pub = rospy.Publisher('/pitch_error',Float64, queue_size=1)
        self.roll_error_pub = rospy.Publisher('/roll_error',Float64, queue_size=1)
        
        #publishing /astrobiolocation
        self.astrobiolocation_pub = rospy.Publisher('/astrobiolocation',Biolocation,queue_size = 1)
        
        #-----------------------------------------------------------------------------------------------------------
        # Subscribing to /whycon/poses, /pid_tuning_altitude, /pid_tuning_pitch, pid_tuning_roll
        rospy.Subscriber('whycon/poses', PoseArray, self.whycon_callback)
    
        #-------------------------Add other ROS Subscribers here--------------------------------------------------------
        rospy.Subscriber('/pid_tuning_altitude',PidTune,self.altitude_set_pid)
        rospy.Subscriber('/pid_tuning_pitch',PidTune,self.pitch_set_pid)
        rospy.Subscriber('/pid_tuning_roll',PidTune,self.roll_set_pid)
        
        #Subscribing to /swift/camera_rgb/image_raw
        rospy.Subscriber('/swift/camera_rgb/image_raw',Image,self.image_callback)
        
        #------------------------------------------------------------------------------------------------------------

        self.arm() # ARMING THE DRONE

    # Disarming condition of the drone
    def disarm(self):

        self.cmd.rcAUX4 = 1100
        self.command_pub.publish(self.cmd)
        rospy.sleep(1)


    # Arming condition of the drone : Best practise is to disarm and then arm the drone.
    def arm(self):

        # self.disarm()

        self.cmd.rcRoll = 1500
        self.cmd.rcYaw = 1500
        self.cmd.rcPitch = 1500
        self.cmd.rcThrottle = 1000
        self.cmd.rcAUX4 = 1500
        self.command_pub.publish(self.cmd)	                # Publishing /drone_command
        rospy.sleep(1)

    # Whycon callback function
    # The function gets executed each time when /whycon node publishes /whycon/poses 
    def whycon_callback(self,msg):
        self.drone_position[0] = msg.poses[0].position.x

        #--------------------Set the remaining co-ordinates of the drone from msg----------------------------------------------
        self.drone_position[1] = msg.poses[0].position.y
        self.drone_position[2] = msg.poses[0].position.z	
        
        #---------------------------------------------------------------------------------------------------------------
    # Callback function for /pid_tuning_altitude
    # This function gets executed each time when /tune_pid publishes /pid_tuning_altitude
    def altitude_set_pid(self,alt):
        self.Kp[2] = alt.Kp * 0.4                     
        self.Ki[2] = alt.Ki * 0.00018
        self.Kd[2] = alt.Kd * 6
    #----------------------------Define callback function like altitide_set_pid to tune pitch, roll--------------
    # Callback function for /pid_tuning_pitch
    # This function gets executed each time when /tune_pid publishes /pid_tuning_pitch
    def pitch_set_pid(self,pitch):
        self.Kp[1] = pitch.Kp * 0.3 
        self.Ki[1] = pitch.Ki * 0.0
        self.Kd[1] = pitch.Kd * 6
        
    # Callback function for /pid_tuning_roll
    # This function gets executed each time when /tune_pid publishes /pid_tuning_roll
    def roll_set_pid(self,roll):
        self.Kp[0] = roll.Kp * 0.3 
        self.Ki[0] = roll.Ki * 0.0
        self.Kd[0] = roll.Kd * 6

    #----------------------------------------------------------------------------------------------------------------------     
    def pid(self):
        
    #-----------------------------Write the PID algorithm here--------------------------------------------------------------       
        self.now = int(round(time.time()*1000))                                          #storing current time in senconds
        self.time_change = self.now - self.last_time									 #difference between current time and last time
         
        if(self.time_change >self.sample_time):
            if(self.last_time !=0):
                #error of all the cordinates(for proportional)				
                self.error[0] = self.drone_position[0] - self.setpoint[self.i][0]   #for roll
                self.error[1] = self.drone_position[1] - self.setpoint[self.i][1]   #for pitch
                self.error[2] = self.drone_position[2] - self.setpoint[self.i][2]   #for throttl

                
                #sum of errors (for integral)				
                self.error_sum[2] = self.error_sum[2]+(self.error[2]*self.time_change)     #for throttle

                #difference of the errors(for derivative)
                self.error_diff[0] = (self.error[0] - self.prev_error[0])/self.time_change   #for roll
                self.error_diff[1] = (self.error[1] - self.prev_error[1])/self.time_change   #for pitch
                self.error_diff[2] = (self.error[2] - self.prev_error[2])/self.time_change   #for throttle	

                #pid outoput for each axis
                self.cmd.rcRoll = 1500-int((self.Kp[0]*self.error[0])+(self.Kd[0]*self.error_diff[0]))		#roll 
                self.cmd.rcPitch = 1500+int((self.Kp[1]*self.error[1])+(self.Kd[1]*self.error_diff[1]))		#pitch
                self.cmd.rcThrottle = 1500+int((self.Kp[2]*self.error[2])+ (self.Kd[2]*self.error_diff[2])+(self.error_sum[2]*self.Ki[2])) #throttle

            #limiting the max and min values for each axis
                #throtle conditions
                if self.cmd.rcThrottle >2000:
                    self.cmd.rcThrottle = self.max_values
                if self.cmd.rcThrottle <1000:
                    self.cmd.rcThrottle = self.min_values
                
                #Pitch conditions
                if self.cmd.rcPitch >2000:
                    self.cmd.rcPitch = self.max_values
                if self.cmd.rcPitch <1000:
                    self.cmd.rcPitch = self.min_values
                                    
                #Roll conditions
                if self.cmd.rcRoll >2000:
                    self.cmd.rcRoll = 2000
                if self.cmd.rcRoll <1000:
                    self.cmd.rcRoll = self.min_values
                    
                self.command_pub.publish(self.cmd) 	 #publishing values on rostopic 'drone_command'

                self.prev_error[0]= self.error[0]    # storing previous roll error
                self.prev_error[1]= self.error[1]	 # storing previous pitch error
                self.prev_error[2]= self.error[2]	 # storing previous throttle error
            
            #updating the last time value 
            self.last_time = self.now	              

            # values for displaying on plotjuggler
            self.alt_error_pub.publish(self.error[2])   
            self.pitch_error_pub.publish(self.error[1])
            self.roll_error_pub.publish(self.error[0])	
            
            if(-0.2< swift_drone.error[0]<0.2 and -0.2< swift_drone.error[1]< 0.2 and -0.2< swift_drone.error[2]<0.2 and self.i < (len(swift_drone.setpoint)-1)):
                self.i = self.i+1  
                
    
    #------------------------------------------------------------------------------------------------------------------------
    
    def image_callback(self, img_msg):
        self.cv_image = self.bridge.imgmsg_to_cv2(img_msg,"passthrough")
        
        width = 200
        heigth = 100
        x = int((self.cv_image.shape[1]-width)/2)
        y = int((self.cv_image.shape[0]-heigth)/2)
        grayscale_image = cv.cvtColor(self.cv_image,cv.COLOR_BGR2GRAY)
        blurred_image = cv.GaussianBlur(grayscale_image, (5, 5), 0) 
        threshold = cv.threshold(blurred_image, 225, 255, 0)[1] 
        self.contours = cv.findContours(threshold,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2]      
        for c,contour in enumerate(self.contours):
            points = cv.moments(contour)
            (x,y) ,radius = cv.minEnclosingCircle(contour)
            center = (int(x),int(y))
            cv.drawContours(self.cv_image,[contour],-1,(0,0,255),3)
            cv.putText(self.cv_image,'+'+str(c+1),(center[0],center[1]),cv.FONT_HERSHEY_COMPLEX
                           ,0.6,(0,255,0),2)
            a,b,c,d = cv.boundingRect(cv.drawContours(np.zeros_like(grayscale_image),self.contours,-1,1))
                
            cv.rectangle(self.cv_image,(a-4,b-4),(a+c+10,b+d+10),(0,0,255),2)          
        cv.imshow("image_window",self.cv_image) 
        cv.waitKey(1)     

    
    def orgainsm_type(self,number_of_contours):
        if number_of_contours ==2:
            self.alien.organism_type = "alien_a"
        elif number_of_contours == 3:
            self.alien.organism_type = "alien_b"
        elif number_of_contours == 4:
            self.alien.organism_type = "alien_c" 
            
        return self.alien.organism_type
        
    def organism_data(self):     
        self.alien.organism_type = self.orgainsm_type(number_of_contours=len(self.contours))                                
        self.alien.whycon_x = self.drone_position[0]
        self.alien.whycon_y = self.drone_position[1]
        self.alien.whycon_z = self.drone_position[2]
        
    # def organism_data_publisher(self):
    #     if()
    # def path(self):
    #     if(swift_drone.i < (len(swift_drone.setpoint)-1)):
    #         swift_drone.pid() 
    #     elif(-0.2< swift_drone.error[0]<0.2 and -0.2< swift_drone.error[1]< 0.2 and -0.2< swift_drone.error[2]<0.2 and swift_drone.i == (len(swift_drone.setpoint)-1)):
    #         swift_drone.disarm()   
    def pid_hold_allign(self):
        
    #-----------------------------Write the PID algorithm here--------------------------------------------------------------       
        self.now = int(round(time.time()*1000))                                          #storing current time in senconds
        self.time_change = self.now - self.last_time									 #difference between current time and last time
         
        if(self.time_change >self.sample_time):
            if(self.last_time !=0):
                #error of all the cordinates(for proportional)				
                self.error[0] = (self.frame_centroid[0] - self.organism_centroid[0])/100  #for roll
                self.error[1] = (self.frame_centroid[1] - self.organism_centroid[1])/100  #for pitch 
                self.error[2] = self.drone_position[2] - self.setpoint[self.i][2]   #for throttl

                
                #sum of errors (for integral)				
                self.error_sum[2] = self.error_sum[2]+(self.error[2]*self.time_change)     #for throttle

                #difference of the errors(for derivative)
                self.error_diff[0] = (self.error[0] - self.prev_error[0])/self.time_change   #for roll
                self.error_diff[1] = (self.error[1] - self.prev_error[1])/self.time_change   #for pitch
                self.error_diff[2] = (self.error[2] - self.prev_error[2])/self.time_change   #for throttle	

                #pid outoput for each axis
                self.cmd.rcRoll = 1500-int((self.Kp[0]*self.error[0])+(self.Kd[0]*self.error_diff[0]))		#roll 
                self.cmd.rcPitch = 1500+int((self.Kp[1]*self.error[1])+(self.Kd[1]*self.error_diff[1]))		#pitch
                self.cmd.rcThrottle = 1500+int((self.Kp[2]*self.error[2])+ (self.Kd[2]*self.error_diff[2])+(self.error_sum[2]*self.Ki[2])) #throttle

            #limiting the max and min values for each axis
                #throtle conditions
                if self.cmd.rcThrottle >2000:
                    self.cmd.rcThrottle = self.max_values
                if self.cmd.rcThrottle <1000:
                    self.cmd.rcThrottle = self.min_values
                
                #Pitch conditions
                if self.cmd.rcPitch >2000:
                    self.cmd.rcPitch = self.max_values
                if self.cmd.rcPitch <1000:
                    self.cmd.rcPitch = self.min_values
                                    
                #Roll conditions
                if self.cmd.rcRoll >2000:
                    self.cmd.rcRoll = 2000
                if self.cmd.rcRoll <1000:
                    self.cmd.rcRoll = self.min_values
                    
                self.command_pub.publish(self.cmd) 	 #publishing values on rostopic 'drone_command'

                self.prev_error[0]= self.error[0]    # storing previous roll error
                self.prev_error[1]= self.error[1]	 # storing previous pitch error
                self.prev_error[2]= self.error[2]	 # storing previous throttle error
            
            #updating the last time value 
            self.last_time = self.now	              

            # values for displaying on plotjuggler
            self.alt_error_pub.publish(self.error[2])   
            self.pitch_error_pub.publish(self.error[1])
            self.roll_error_pub.publish(self.error[0])	
            
            # if(-0.2< swift_drone.error[0]<0.2 and -0.2< swift_drone.error[1]< 0.2 and -0.2< swift_drone.error[2]<0.2 and self.i < (len(swift_drone.setpoint)-1)):
            #     self.i = self.i+1  
                
                
    def hold_allign():
        pass 
if __name__ == '__main__':

    swift_drone = swift()
    r = rospy.Rate(30) #specify rate in Hz based upon your desired PID sampling time, i.e. if desired sample time is 33ms specify rate as 30Hz
    while not rospy.is_shutdown():          
        if(swift_drone.i < (len(swift_drone.setpoint)-1)):
            swift_drone.pid()            
        elif(len(swift_drone.setpoint)-1):
            swift_drone.disarm()
            
        if(swift_drone.orgainsm_type(len(swift_drone.contours)) =="alien_b" and swift_drone.flag < 1):
            swift_drone.organism_data()             
            swift_drone.astrobiolocation_pub.publish(swift_drone.alien) 
            swift_drone.flag = 1     
        r.sleep() 
    