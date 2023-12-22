from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2
import math 
import argparse

#pass argumnts here

class detection():

   def __init__(self):
      # initialize nodes here
      # Declare Variables Here

      self.organism_type = " "
      self.centroids = []
      self.number_contours = 0

   def parse_image(self):
      self.parser = argparse.ArgumentParser()
      self.parser.add_argument("--image", help = "Enter Image File Name")
      self.args = self.parser.parse_args()

   def image_call(self):
      self.image_read = cv2.imread(self.args.image,1)
      self.gray_image = cv2.cvtColor(self.image_read, cv2.COLOR_BGR2GRAY)
      blurred_image = cv2.GaussianBlur(self.gray_image, (5,5),0)

      self.threshold = cv2.threshold(blurred_image, 225, 255, 0)[1]          
      self.threshold = cv2.erode(self.threshold, None, iterations=2)               
      self.threshold = cv2.dilate(self.threshold, None, iterations=4) 

   
   def draw_contours(self):

      self.contours, _ = cv2.findContours(self.threshold,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      self.contours = sorted(self.contours, key=lambda c:cv2.boundingRect(c)[0])
      self.number_contours = len(self.contours)

      for i, contour in enumerate(self.contours):
         points = cv2.moments(contour)

         (x,y),radius = cv2.minEnclosingCircle(contour)

         cv2.drawContours(self.image_read,[contour],-1,(0,0,255),2)
         a,b,c,d = cv2.boundingRect(cv2.drawContours(np.zeros_like(self.gray_image),self.contours,-1,1))
         cv2.rectangle(self.image_read,(a-4,b-4),(a+c+10,b+d+10),(0,0,255),2) 
      
      cv2.putText(self.image_read,self.organism_type,(a,b),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,255,0),2)
            



   def organism_data(self,number_contours):
      if number_contours <= 1:
         self.organism_type = "alien"
      elif number_contours == 2 :
         self.organism_type = "alien_a"
      elif number_contours == 3 :
         self.organism_type = "alien_b"
      elif number_contours == 4 :
         self.organism_type = "alien_c"
      elif number_contours == 5 :
         self.organism_type = "alien_d"
      elif number_contours >= 6 :
         self.organism_type = "Dher Sare aliens"

   
   
   def writing_image(self):
      
      cv2.imwrite(f"{self.args.image}_detection_result.png",img.image_read)
      
      with open(f"{self.args.image}_detection_result.txt","w") as file:

         file.write(f"Organism Type : {self.organism_type}\n")
         file.write(f"Number of Contours : {self.number_contours}\n")
         file.write(f"Centroid : {self.centroids}\n")
   
      
              



if __name__ == '__main__':

   img = detection()
   img.parse_image()
   img.image_call()
   img.draw_contours()
   img.organism_data(img.number_contours)
   img.writing_image()

   
      
