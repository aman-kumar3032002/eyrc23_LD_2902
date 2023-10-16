'''
# Team ID:          eYRC#LD#2902
# Theme:            Luminosity Drone
# Author List:      Manila Raj Putra, Aman Kumar, Harsh Gulzar, Shivam Kumar
# Filename:         led_detection.py
# Functions:        None
# Global variables: None
'''

# import the necessary packages
from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2
import math

# load the image
image = cv2.imread('led.jpg', 1)                                    #Reading the image named 'led.jpg' and storing in variable image

# convert it to grayscale, and blur it
grayscale_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)            #Converting 'image' to grayscale and storing it 
blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)        #Blurring 'grayscale_image' using gaussianblur and storing it in blurred_image varible

# threshold the image to reveal light regions in the blurred image
threshold = cv2.threshold(blurred_image, 225, 255, 0)[1]            #Thresholding 'blurred_image' with thresh value- 225,255,0 and storing it

# perform a series of erosions and dilations to remove any small blobs of noise from the thresholded image
threshold = cv2.erode(threshold, None, iterations=2)                #Performing erosion with 2 iterations
threshold = cv2.dilate(threshold, None, iterations=4)               #Performing dilations with 4 iterations

# perform a connected component analysis on the thresholded image, then initialize a mask to store only the "large" components
labels = measure.label(threshold,  background=0)                    #Performing connected component analysis 
mask = np.zeros(threshold.shape, dtype="uint8")                     #Initializing an empty mask value with same threshold shape

# loop over the unique components
for label in np.unique(labels):

	# if this is the background label, ignore it
    if label == 0:                                                  #If label is 0 , continue
        continue

	# otherwise, construct the label mask and count the number of pixels 
    labelMask = np.zeros(threshold.shape, dtype="uint8")            #Initializing an empty label mask with same threshold
    labelMask[labels == label] = 255                                #Setting Current pixel level to 255
    numPixels = cv2.countNonZero(labelMask)                         #Counting Non Zero pixels in label mask 
    
	# if the number of pixels in the component is sufficiently large, then add it to our mask of "large blobs"
    if numPixels > 300:
        mask = cv2.add(mask, labelMask)                             #If Number of Pixels greater than 300 , adding labelmask to mask 
        
# find the contours in the mask, then sort them from left to right
contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Finding Contours 
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])                       #Sorting Contours based on x-coordinates

# loop over the contours

# Initialize lists to store centroid coordinates and area
centroids = []                                                      #Centroids List to store centroid coordinates
areas = []                                                          #Areas List to store area

#Looping over enumerate contours
for i, contour in enumerate(contours):
    #calculating moments
    points = cv2.moments(contour)                                   #storing moments in points
    centroid_X = float(points["m10"] / points["m00"])               #Calculating centroid X-coordintes                                                          
    centroid_Y = float(points["m01"] / points["m00"])               #Calculating centroid Y-coordinates
    #["m00"] -- represnts Total Area of Object
    #["m10"] -- Gives center of mass in X-Axis
    #["m01"] -- Gives center of mass in Y-Axis
    centroid = (centroid_X,centroid_Y)                              #Storing Coordinates
    
    (x,y),radius = cv2.minEnclosingCircle(contour)                  #Calculating radius, x&y points using minEnclosing circle method
    radius = int(radius)                                            #Coverting radius to integer
    center = (int(x),int(y))                                        #converting centers to integer   

    # Calculate the area of the contour
    area = math.pi * (radius**2)                                    #Calculating Area of circle

    # Draw the bright spot on the image
    cv2.drawContours(image,[contour],-1,(0,0,255),2)                #Drawing Contour on the Image
    cv2.putText(image,'+'+str(i+1),(center[0],center[1]-radius-10),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,0,255),2) #Adding a label above the contoured image

    # Append centroid coordinates and area to the respective lists
    centroids.append(centroid)                                      #Appending Centroids to 'centroids' list
    areas.append(area)                                              #Appending Areas to 'areas' list

# Save the output image as a PNG file
cv2.imwrite("led_detection_results.png", image)                     #Saving the image with detected LEDs

a = len(contours)                                                   #Storing the number of Contours

# Open a text file for writing
with open("led_detection_results.txt", "w") as file:                #Opening a TextFile for writing
    # Write the number of LEDs detected to the file
    file.write(f"No. of LEDs detected: {a}\n")
    # Looping over the contours
    for i, contour in enumerate(contours):
        # Centroid coordinates and area for each LED
         file.write(f"Centroid #{i + 1}: {centroid}\nArea #{i + 1}: {area}\n")       
                  
# Close the text file
file.close()  #Closing the Text File