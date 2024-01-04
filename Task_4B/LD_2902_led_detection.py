'''
# Team ID:          2902
# Theme:            Luminosity Drone
# Author List:      Manila Raj Putra, Aman Kumar, Harsh Gulzar, Shivam Kumar
# Filename:         LD_2902_led_detection.py
# Functions:        __init__(), call_image(), preprocess_image(), detect_contours(), detect_cluster(), write_data()
# Global variables: None
'''

# importing the necessary packages
from imutils import contours
from skimage import measure
from sklearn.cluster import KMeans
import numpy as np
import cv2
import argparse

class Detection():

   def __init__(self):
      """
        Purpose:
        -------------------------------------------
        Initilizing the newly created object.
    
        Input Arguments:
        -------------------------------------------
        self
    
        Returns:
        -------------------------------------------
        None
    
        Example Call:
        -------------------------------------------
        
      """ 
      self.organism_type_map = {
            2 : "alien_a",
            3 : "alien_b",
            4 : "alien_c",
            5 : "alien_d"
      }                                                                            #self.organism_type_map: dictonary to stores all the names of the organism as value and led count as their key
      self.centroids = []        						                                #self.centroids: list to store the centroids of the organisms 
      self.organisms_type = []                                                     #self.organism_type: list to store the type of organism detected by the algorithm
      self.threshold_area = 50                                                     #self.threshold_area: store max threshold area to form cluster 

   def call_image(self):
      """
        Purpose:
        -------------------------------------------
        Function to read the images using argument parser
    
        Input Arguments:
        -------------------------------------------
        self
    
        Returns:
        -------------------------------------------
        None
    
        Example Call:
        -------------------------------------------
        call_image()
      """
      parser = argparse.ArgumentParser()                                           #parser: Intializes the Parser argument  
      
      #Adding parser argument named --image and help text.
      parser.add_argument('--image', help = 'Enter Image File Name')     
                
      args = parser.parse_args()                                                   #args: Parse command-line arguments using the argparse module
      #reading the path of the image --------------------------------------------
      self.image_path = args.image                                                 #self.image_path: Storing the image path using command line 
      self.image = cv2.imread(self.image_path,1)                                   #self.image: Reading image using CV.imread function

   def preprocess_image(self):
      """
        Purpose:
        -------------------------------------------
        Applies filter to remove noise from the images
    
        Input Arguments:
        -------------------------------------------
        self
    
        Returns:
        -------------------------------------------
        None
    
        Example Call:
        -------------------------------------------
        preprocess_image()
      """
      grayscale_image = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)                 #Converting 'image' to grayscale and storing it 
      blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)                  #Blurring 'grayscale_image' using gaussianblur and storing it in blurred_image varible
   # threshold the image to reveal light regions in the blurred image------------------------------------------------------------------
      self.threshold = cv2.threshold(blurred_image, 225, 255, 0)[1]                 #Thresholding 'blurred_image' with thresh value- 225,255,0 and storing it

   # perform a series of erosions and dilations to remove any small blobs of noise from the thresholded image--------------------------
      self.threshold = cv2.erode(self.threshold, None, iterations=2)                #Performing erosion with 2 iterations
      self.threshold = cv2.dilate(self.threshold, None, iterations=4)               #Performing dilations with 4 iterations


   def detect_contours(self):
      """
        Purpose:
        -------------------------------------------
        Detects the conoturs using OpenCV
    
        Input Arguments:
        -------------------------------------------
        self
    
        Returns:
        -------------------------------------------
        None
    
        Example Call:
        -------------------------------------------
        detect_contours()
      """
      self.contours, heirarchy = cv2.findContours(self.threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)     #self.contours: Finding Contours 
      #Sorting Contours based on x-coordinates(left to right)-------------------------------------------------
      self.contours = sorted(self.contours, key=lambda c: cv2.boundingRect(c)[0])    
      #Drawing boundary around detected contours--------------------------------------------------------------                         
      cv2.drawContours(self.image,self.contours,-1,(0,0,255),2)                                               

   def detect_cluster(self):
      """
        Purpose:
        -------------------------------------------
        Detects the Led Clusters 
    
        Input Arguments:
        -------------------------------------------
        self
    
        Returns:
        -------------------------------------------
        None
    
        Example Call:
        -------------------------------------------
        detect_cluster()
      """
      grouped_contours = [self.contour for self.contour in self.contours if cv2.contourArea(self.contour) > self.threshold_area]                                                            #grouped_contours: grouping the number of contours according to contour area
      contour_centers = np.array([[(center_x + width) / 2, (center_y + height) / 2] for center_x, center_y, width, height in [cv2.boundingRect(contour) for contour in grouped_contours]])  #contour_centers: Calculating centers of grouped contours using bounding rect method
      
      # Determine the number of clusters according to threshold area and number of detected contours
      if cv2.contourArea(self.contour) > self.threshold_area :
         if len(self.contours) > 13 :
             num_clusters = 4
         elif len(self.contours) > 8 and len(self.contours) < 13 :
            num_clusters = 3 
         elif len(self.contours) > 5 and len(self.contours) < 10 :
            num_clusters = 2 
         else:num_clusters = 1

      
      #Apply KMeans Clustering-----------------------------------------------------------------------------------
      kmeans = KMeans(n_clusters = num_clusters,n_init= "auto")
      kmeans.fit(contour_centers)                       #Fitting contours centers to Kmenas labels
      labels = kmeans.labels_

      #Group contours based on the KMeans labels-----------------------------------------------------------------
      self.clusters = {i: [] for i in range(num_clusters)}
      
      #looping over grouped contours
      for i, contour in enumerate(grouped_contours):
         self.clusters[labels[i]].append(contour)

      self.grouped_image = self.image.copy()           #self.grouped_image: Storing a copy of image

      #looping over num_clusters----------------------------------------------------------------------------------
      for i in range(num_clusters):
         if len(self.clusters[i]) > 0:
            # Combine all contours in the cluster
            combined_contour = np.vstack(self.clusters[i])
            
            # Calculate the convex hull
            convex_hull = cv2.convexHull(combined_contour)
            
            # Get the bounding rectangle for the convex hull
            point_x, point_y, width, height = cv2.boundingRect(convex_hull)        
           
            # Drawing the bounding rectangle for each clusters
            cv2.rectangle(self.grouped_image, (point_x, point_y), (point_x + width, point_y + height), (0, 255, 0), 2)
                                
            points = cv2.moments(convex_hull)                               #storing moments in points
            centroid_X = float(points["m10"] / points["m00"])               #Calculating centroid X-coordintes                                                          
            centroid_Y = float(points["m01"] / points["m00"])               #Calculating centroid Y-coordinates
                                     #["m00"] -- represnts Total Area of Object
                                     #["m10"] -- Gives center of mass in X-Axis
                                     #["m01"] -- Gives center of mass in Y-Axis
            centroid = (centroid_X,centroid_Y)                              #centroid: tupple to store X and Y coordinates of convex_hull
            self.centroids.append(centroid)                                 #self.centroids: Appending the X and Y coordinates to list
            
            self.number_of_contours = len(self.clusters[i])                 #self.number_of_contours: Store the number of contours and the corresponding organism type
            organism = self.organism_type_map.get(self.number_of_contours, "unknown")   
            self.organisms_type.append(organism)                            #self.organisms_type: Store the type of different organisms


   def write_data(self):
      """
        Purpose:
        -------------------------------------------
        Writing data in the format of an image and text
    
        Input Arguments:
        -------------------------------------------
        self
    
        Returns:
        -------------------------------------------
        None
    
        Example Call:
        -------------------------------------------
        Run when we have to write the result data in text and image format
      """

      #-----------------Opening a text file for writing image data------------------------
      with open(f"{self.image_path}_detection_results.txt", "w") as file:                
         #------------------------Looping over the contours-------------------------------
         for i, contour in enumerate(self.clusters):
             #----------------------writing the organism type-----------------------------
            file.write(f"Organism Type: {self.organisms_type[i]}\n")
             #-------------writing the centroid of the organism cluster-------------------                
            file.write(f"Centroid: {self.centroids[i]}\n\n")     
                      
         #----------Saving the image with name, imageName_detection_result.png------------   
      cv2.imwrite(f"{self.image_path}_detection_results.png", self.grouped_image)        
                        
         #--------------------------Closing the text file---------------------------------
      file.close()                                                                      

if __name__ == '__main__' :

   detection = Detection()    #detection: object of  the Detection Class
   #reading images using image parser---------------------------
   detection.call_image()
   
   #preprocessing the images to remove noise--------------------
   detection.preprocess_image()
   
   #detecting countour in the processed image-------------------
   detection.detect_contours()
   
   #detecting clusters -----------------------------------------
   detection.detect_cluster()
   
   #writing data in a file--------------------------------------
   detection.write_data()
