#!/usr/bin/env python3
import numpy as np
import cv2
import serial
import time
import random


def generate_random_pattern():
    patterns = [
        (350, 200, 0),  # Example pattern 1
        (100, 200, 0),  # Example pattern 2
        (250, 150, 0),  # Example pattern 3
        (400, 250, 0),  # Example pattern 4
        (450, 200, 0),  # Example pattern 5
        (480, 200, 0),  # Example pattern 6
        (350, 200, 0),  # Example pattern 7
        (300, 200, 0),  # Example pattern 8
        (250, 200, 0),  # Example pattern 9
    ]

    return random.choice(patterns)


def onTrack1(val):
    global hueLow
    hueLow = val


def onTrack2(val):
    global hueHigh
    hueHigh = val


def onTrack3(val):
    global satLow
    satLow = val


def onTrack4(val):
    global satHigh
    satHigh = val


def onTrack5(val):
    global valLow
    valLow = val


def onTrack6(val):
    global valHigh
    valHigh = val


port = "/dev/ttyACM0"
baud_rate = 115200  # Default baud rate for micro:bit

ser = serial.Serial(port, baud_rate, timeout=1)


def send_data_to_microbit(data):
    global ser

    if data:
        try:
            ser.write(data.encode())
            print(f"Data sent: {data}", end="")

        except Exception as e:
            print(f"Error: {e}")


width = 640
height = 480


cam = cv2.VideoCapture(0)  # Change for your camera index or video file
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_FPS, 60)

cv2.namedWindow('tracker')

hueLow = 40  # Adjusted for green color
hueHigh = 80  # Adjusted for green color
satLow = 40
satHigh = 255
valLow = 40
valHigh = 255

cv2.createTrackbar('Hue Low', 'tracker', 40, 179, onTrack1)
cv2.createTrackbar('Hue High', 'tracker', 80, 179, onTrack2)
cv2.createTrackbar('Sat Low', 'tracker', 40, 255, onTrack3)
cv2.createTrackbar('Sat High', 'tracker', 255, 255, onTrack4)
cv2.createTrackbar('Val Low', 'tracker', 40, 255, onTrack5)
cv2.createTrackbar('Val High', 'tracker', 255, 255, onTrack6)

elapsed = 0
update_rate = 0.2  # How often to send data to microbit in seconds
start = time.time_ns()
send_rate = 3  # Send data every 3 seconds

while True:
    ignore, frame = cam.read()

    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowerBound = np.array([hueLow, satLow, valLow])
    upperBound = np.array([hueHigh, satHigh, valHigh])
    mask = cv2.inRange(frameHSV, lowerBound, upperBound)

    mask = cv2.erode(mask, None, iterations=10)
    mask = cv2.dilate(mask, None, iterations=10)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    z = 0  # Default value for z
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        mappingZ = w * h
        z = (mappingZ**-1)*180000
        print(f"w={w}, h={h}")
        print(f"z = {z}")
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        now = time.time_ns()
        if now - start >= 1_000_000_000 * update_rate:
            start = now
            send_data_to_microbit(f"{x + w / 2},{y + h / 2},{z}\r\n")

    else:
        now = time.time_ns()
        if now - start >= 1_000_000_000 * send_rate:
            start = now
            x, y, z = generate_random_pattern()
            data_to_send = f"{x},{y},{z}\n"
            send_data_to_microbit(data_to_send)

    cv2.imshow('camera', frame)
    cv2.waitKey(1)

ser.close()
cam.release()
cv2.destroyAllWindows()