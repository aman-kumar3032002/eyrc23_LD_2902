'''
# Team ID:          2902
# Theme:            Luminosity Drone
# Author List:      Manila Raj Putra, Aman Kumar, Harsh Gulzar, Shivam Kumar
# Filename:         LD_2902_led_detection.py
# Functions:        __init__(), call_image(), preprocess_image(), detect_contours(), detect_cluster(), write_data()
# Global variables: self.organism_type_map = {}, self.centroids = [], self.organisms_type = [], self.threshold_area, 
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
      '''
      Purpose:
      ---
      To initializes all various variables here.

      '''

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
      parser = argparse.ArgumentParser()                                           #parser: Intializes the Parser argument  
      parser.add_argument('--image', help = 'Enter Image File Name')               #Adding parser argument named --image and help text.
      args = parser.parse_args()                                                   #args: Parse command-line arguments using the argparse module

      self.image_path = args.image                                                 #self.image_path: Storing the image path using command line 

      self.image = cv2.imread(self.image_path,1)                                   #self.image: Reading image using CV.imread function

   def preprocess_image(self):
      grayscale_image = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)                 #Converting 'image' to grayscale and storing it 
      blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)                  #Blurring 'grayscale_image' using gaussianblur and storing it in blurred_image varible
      # threshold the image to reveal light regions in the blurred image
      self.threshold = cv2.threshold(blurred_image, 225, 255, 0)[1]                 #Thresholding 'blurred_image' with thresh value- 225,255,0 and storing it

      # perform a series of erosions and dilations to remove any small blobs of noise from the thresholded image
      self.threshold = cv2.erode(self.threshold, None, iterations=2)                #Performing erosion with 2 iterations
      self.threshold = cv2.dilate(self.threshold, None, iterations=4)               #Performing dilations with 4 iterations

      # perform a connected component analysis on the thresholded image, then initialize a mask to store only the "large" components
      labels = measure.label(self.threshold,  background=0)                         #Performing connected component analysis 
      mask = np.zeros(self.threshold.shape, dtype="uint8")                          #Initializing an empty mask value with same threshold shape

      #  loop over the unique components
      for label in np.unique(labels):

	   # if this is the background label, ignore it
         if label == 0:                                                            #If label is 0 , continue
              continue

	   # otherwise, construct the label mask and count the number of pixels 
      labelMask = np.zeros(self.threshold.shape, dtype="uint8")                    #Initializing an empty label mask with same threshold
      labelMask[labels == label] = 255                                             #Setting Current pixel level to 255
      numPixels = cv2.countNonZero(labelMask)                                      #Counting Non Zero pixels in label mask 
    
	   # if the number of pixels in the component is sufficiently large, then add it to our mask of "large blobs"
      # if numPixels > 100:
      self.mask = cv2.add(mask, labelMask) 
      self.image_pass = self.mask.copy()


   def detect_contours(self):
      self.contours, heirarchy = cv2.findContours(self.threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)     #Finding Contours 
      self.contours = sorted(self.contours, key=lambda c: cv2.boundingRect(c)[0])                             #Sorting Contours based on x-coordinates(left to right)
      cv2.drawContours(self.image,self.contours,-1,(0,0,255),2)                                               #Drawing boundary around detected contours

   def detect_cluster(self):
      grouped_contours = [self.contour for self.contour in self.contours if cv2.contourArea(self.contour) > self.threshold_area]  #grouped_contours: grouping the number of contours according to contour area
      contour_centers = np.array([[(center_x + width) / 2, (center_y + height) / 2] for center_x, center_y, width, height in [cv2.boundingRect(contour) for contour in grouped_contours]])  #contour_centers: Calculating centers of grouped contours using bounding rect method
      print(cv2.contourArea(self.contour))

      # Determine the number of clusters according to threshold area and number of contours
      if cv2.contourArea(self.contour) > self.threshold_area :
         if len(self.contours) > 13 :
             num_clusters = 4
         elif len(self.contours) > 8 and len(self.contours) < 13 :
            num_clusters = 3 
         elif len(self.contours) > 5 and len(self.contours) < 10 :
            num_clusters = 2 
         else:num_clusters = 1

      
      #Apply KMeans Clustering
      kmeans = KMeans(n_clusters = num_clusters)
      kmeans.fit(contour_centers)
      labels = kmeans.labels_

      #Group contours based on the KMeans labels
      self.clusters = {i: [] for i in range(num_clusters)}
      for i, contour in enumerate(grouped_contours):
         self.clusters[labels[i]].append(contour)

      self.grouped_image = self.image.copy()           #self.grouped_image: Storing a copy of image

      #looping over num_clusters
      for i in range(num_clusters):
         if len(self.clusters[i]) > 0:
            # Combine all contours in the cluster
            combined_contour = np.vstack(self.clusters[i])
            
            # Calculate the convex hull
            convex_hull = cv2.convexHull(combined_contour)
            
            # Get the bounding rectangle for the convex hull
            point_x, point_y, width, height = cv2.boundingRect(convex_hull)
           
            # Draw the bounding rectangle for each clusters
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

      #Open a text file for writing
      with open(f"{self.image_path}_detection_results.txt", "w") as file:                
    
         # Looping over the contours
         for i, contour in enumerate(self.clusters):
            file.write(f"Organism Type: {self.organisms_type[i]}\n")
            file.write(f"Centroid: {self.centroids[i]}\n\n")     
            
              
      #Saving the image         
      cv2.imwrite(f"{self.image_path}_detection_results.png", self.grouped_image)        
                        
      # Close the text file
      file.close()                                                                       

if __name__ == '__main__' :

   #calling the functions here

   detection = Detection()
   detection.call_image()
   detection.preprocess_image()
   detection.detect_contours()
   detection.detect_cluster()
   detection.write_data()
