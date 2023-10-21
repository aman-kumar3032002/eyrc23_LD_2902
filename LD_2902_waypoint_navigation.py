#!/usr/bin/env python3
'''
Team ID: 2902
Theme: Luminosity Drone
Author List: Aman Kumar, Harsh gulzar, Manila Raj Putra, Shivam Kumar
Filename: waypoint_navigation.py
Functions: pid()
Global variables: set_point_iterator
'''
#importing modules and libraries
from position_hold import swift                                                                               #importing the position_hold module that was created in task_1a
from swift_msgs.msg import *
import time
import rospy

set_point_iterator = 0                                                                                        #set_point_iterator : iterates through the set_points when condition is full filled

#making an object of the swift class
swift_drone = swift()                                                                                         #swift_drone: object of the swift class that was made in the position_hold module

swift_drone.setpoint = [[0,0,0],[0,0,23],[2,0,23],[2,2,23],[2,2,25],[-5,2,25],
                        [-5,-3,25],[-5,-3,21],[7,-3,21],[7,0,21],[0,0,19]]                                    #swift_drone.setpoint: stores the corrdinates of the setpoints

swift_drone.Kp = [20,20,40]  
swift_drone.Ki = [0,0,0.0029523]  
swift_drone.Kd = [30000,30000,46000]  

def pid():
    """
    Purpose:
    -------------------------------------------
    PID algorithm for the stablization of the swift drone
    
    Input Arguments:
    -------------------------------------------
    None
    
    Returns:
    -------------------------------------------
    None
    
    Example Call:
    -------------------------------------------
    called automatically when the gazebo is running
    """
    swift_drone.now = int(round(time.time()*1000))                                                                              #swift_drone.now: storing integer value of current time in seconds
    swift_drone.time_change = swift_drone.now - swift_drone.last_time									                        #swift_drone.time_change: stores the difference between current time and last time
    
    if(swift_drone.time_change >swift_drone.sample_time):
        if(swift_drone.last_time !=0):
            
            #-------------------------------------------------error calculation for proportional, integral and derivative----------------------------------------
            #error of all the cordinates(for proportional)				
            swift_drone.error[0] = swift_drone.drone_position[0] - swift_drone.setpoint[set_point_iterator][0]                  #swift_drone.error[0]: error in the roll axis
            swift_drone.error[1] = swift_drone.drone_position[1] - swift_drone.setpoint[set_point_iterator][1]                  #swift_drone.error[1]: error in the pitch axis
            swift_drone.error[2] = swift_drone.drone_position[2] - swift_drone.setpoint[set_point_iterator][2]                  #swift_drone.error[2]: error in the throttle axis

            #sum of errors (for integral)				
            swift_drone.error_sum[2] = swift_drone.error_sum[2]+(swift_drone.error[2]*swift_drone.time_change)                  #swift_drone.error_sum[2] : error sum in the throttle axis

            #difference of the errors(for derivative)
            swift_drone.error_diff[0] = (swift_drone.error[0] - swift_drone.prev_error[0])/swift_drone.time_change              #swift_drone.error_diff[0] : error change in the roll axis
            swift_drone.error_diff[1] = (swift_drone.error[1] - swift_drone.prev_error[1])/swift_drone.time_change              #swift_drone.error_diff[1] : error change in the pitch axis
            swift_drone.error_diff[2] = (swift_drone.error[2] - swift_drone.prev_error[2])/swift_drone.time_change              #swift_drone.error_diff[2] : error change in the throttle axis
            #-----------------------------------------------------------------------------------------------------------------------------------------------------

            #pid output for each axis
            swift_drone.cmd.rcRoll = 1500-int((swift_drone.Kp[0]*swift_drone.error[0])+(swift_drone.Kd[0]*swift_drone.error_diff[0]))		#roll
            swift_drone.cmd.rcPitch = 1500+int((swift_drone.Kp[1]*swift_drone.error[1])+(swift_drone.Kd[1]*swift_drone.error_diff[1]))		#pitch
            swift_drone.cmd.rcThrottle = 1500+int((swift_drone.Kp[2]*swift_drone.error[2])+ (swift_drone.Kd[2]*swift_drone.error_diff[2])+(swift_drone.error_sum[2]*swift_drone.Ki[2])) #throttle

            #------------------------------------------------------limiting the max and min values for each axis -------------------------------------------------
            #throtle conditions
            if swift_drone.cmd.rcThrottle >2000:
                swift_drone.cmd.rcThrottle = swift_drone.max_values
            if swift_drone.cmd.rcThrottle <1000:
                swift_drone.cmd.rcThrottle = swift_drone.min_values
                
             #Pitch conditions
            if swift_drone.cmd.rcPitch >2000:
                swift_drone.cmd.rcPitch = swift_drone.max_values
            if swift_drone.cmd.rcPitch <1000:
                swift_drone.cmd.rcPitch = swift_drone.min_values
                                    
            #Roll conditions
            if swift_drone.cmd.rcRoll >2000:
                swift_drone.cmd.rcRoll = swift_drone.max_values
            if swift_drone.cmd.rcRoll <1000:
                swift_drone.cmd.rcRoll = swift_drone.min_values
            #-------------------------------------------------------------------------------------------------------------------------------------------------------

            swift_drone.command_pub.publish(swift_drone.cmd) 	                        #publishing values on rostopic 'drone_command'
            
            swift_drone.prev_error[0]= swift_drone.error[0]                             # swift_drone.prev_error[0]: storing previous roll error
            swift_drone.prev_error[1]= swift_drone.error[1]	                            # swift_drone.prev_error[1]: storing previous pitch error
            swift_drone.prev_error[2]= swift_drone.error[2]	                            # swift_drone.prev_error[2]: storing previous throttle error
            
        #updating the last time value 
        swift_drone.last_time = swift_drone.now	              

        # values for displaying on plotjuggler
        swift_drone.roll_error_pub.publish(swift_drone.error[0])
        swift_drone.pitch_error_pub.publish(swift_drone.error[1])
        swift_drone.alt_error_pub.publish(swift_drone.error[2])   

data_rate = rospy.Rate(30)                                                           #frequency of the data transfer in Hz
while not rospy.is_shutdown():
    pid()                                                                            #calling the pid function
    #checking if drone is in error range
    if((-0.2< swift_drone.error[0]<0.2) and (-0.2< swift_drone.error[1]< 0.2) and (-0.2< swift_drone.error[2]<0.2) and (set_point_iterator < (len(swift_drone.setpoint)-1))):
        #adding 1 to move to next setpoint in the set_point list
        set_point_iterator =set_point_iterator + 1
    data_rate.sleep()