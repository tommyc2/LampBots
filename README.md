# LAMP-Bot-Project
 LAMP Bot Team Project 

 Title: LampBot OpenCV Integration Repository

Description: Explore the LampBot OpenCV Integration repository, a hub for seamlessly combining servo controls with OpenCV functionalities. Below is an overview of key operations in the code:

# 	1.	Webcam Setup using OpenCV:

import cv2

## Open a connection to the webcam (0 denotes the default camera)
cap = cv2.VideoCapture(0)

## Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not access the webcam.")


# 	2.	Coordinate Input from PC/Laptop:
	•	Receive x, y coordinates from the PC or laptop interface.
# 	3.	Servo Control Based on Coordinates:
	•	Utilize the received coordinates to manipulate the LampBot’s servo-controlled head and body movements.
