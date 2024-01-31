#!/usr/bin/env python3
# TODO: Calibrate x/y values when all servos are installed
import numpy as np
import cv2
import serial
import time
import random
import platform
from enum import Enum

class TrackingType(Enum):
    BALL = 1
    FACE = 2

class TrackingState(Enum):
    IDLE = 1
    TRACKING = 2

class HsvValue:
    low: int = 0
    high: int = 255

    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high;

    def setLow(self, val: int): self.low = val
    def setHigh(self, val: int): self.high = val

class Hsv:
    hue = HsvValue(0, 255)
    sat = HsvValue(0, 255)
    val = HsvValue(0, 255)

    def __init__(self, hue_low = 0, hue_high = 255, sat_low = 0, sat_high = 255, val_low = 0, val_high = 255):
        self.hue.low = hue_low
        self.hue.high = hue_high
        self.sat.low = sat_low
        self.sat.high = sat_high
        self.val.low = val_low
        self.val.high = val_high

# Selects a random point from a predefined list of points until it is not `previous_result`,
# and returns it.
def new_idle_point(previous_result):
    points = [
        (350, 320),
        (100, 360),
        (250, 368),
        (400, 280),
        (450, 280),
        (480, 280),
        (350, 320),
        (300, 360),
        (250, 368),
    ]

    ret = random.choice(points)
    # Keep picking until the result is not the same as the previous one
    while ret == previous_result:
        ret = random.choice(points)
    return ret

def trackFace(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if len(faces) > 0:
        largest = (0, 0, 0, 0)
        for (x, y, w, h) in faces:
            w2 = largest[2]
            h2 = largest[3]
            if w * h > w2 * h2:
                largest = (x, y, w, h)

        x, y, w, h = largest
        if w > 100:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return (x + w / 2, y + h / 2)

    return None

def trackBall(frame, hsv):
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowerBound = np.array([ hsv.hue.low, hsv.sat.low, hsv.val.low ])
    upperBound = np.array([ hsv.hue.high, hsv.sat.high, hsv.val.high ])
    mask = cv2.inRange(frameHSV, lowerBound, upperBound)

    # Erode/dilate passes to remove small artifacts
    mask = cv2.erode(mask, None, iterations=10)
    mask = cv2.dilate(mask, None, iterations=10)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Filter contours based on area
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

        if valid_contours:
            # Get the largest contour (presumably the green ball)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Draw a rectangle around the detected ball
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            return (x + w / 2, y + h / 2)

    return None

def send_data(file, x, y):
    if file is None:
        print(f"Data: {x},{y}")
        return

    try:
        # Map `y` to a range between 70-270, as we don't need the full 180 degrees of movement.
        y = max(70, (y / 480) * 270)
        file.write(f"{x},{y}\r\n".encode())
        print(f"Data sent: {x},{y}")

    except Exception as e:
        print(f"Error: {e}")

def main():
    tracking_rate = 0.5 # How often to send data to microbit in seconds
    idle_rate = 3 # How often to send a movement when idling
    tracking_type = TrackingType.FACE

    os = platform.system()

    baud_rate = 115200  # Default baud rate for micro:bit
    width = 640
    height = 480

    cam = None
    port = None
    match os:
        case 'Windows':
            cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            port = "COM6"
        case 'Linux':
            cam = cv2.VideoCapture(0)
            port = "/dev/ttyACM0"
        case _:
            raise NotImplementedError('Unsupported OS')

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cam.set(cv2.CAP_PROP_FPS, 30)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    ser = None
    try:
       ser = serial.Serial(port, baud_rate, timeout=1)
    except Exception as e:
        print(f"Error: could not open serial port: {e}")
        print(f"Continuing without sending over serial")

    hsv = Hsv(
      hue_low = 40,
      hue_high = 80,
      sat_low = 40,
      val_low = 40,
    )

    face_cascade = None
    match tracking_type:
        case TrackingType.BALL:
            cv2.namedWindow('tracker')

            cv2.createTrackbar('Hue Low', 'tracker', 40, 179, hsv.hue.setLow)
            cv2.createTrackbar('Hue High', 'tracker', 80, 179, hsv.hue.setHigh)
            cv2.createTrackbar('Sat Low', 'tracker', 40, 255, hsv.sat.setLow)
            cv2.createTrackbar('Sat High', 'tracker', 255, 255, hsv.sat.setHigh)
            cv2.createTrackbar('Val Low', 'tracker', 40, 255, hsv.val.setLow)
            cv2.createTrackbar('Val High', 'tracker', 255, 255, hsv.val.setHigh)
        case TrackingType.FACE:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Keeps track of whether we were idling or tracking,
    # so we can print the new state to the terminal
    tracking_state = None

    # The last point returned by `new_idle_point`
    idle_point = None

    last_send_time = time.time_ns()
    while True:
        _, frame = cam.read()

        match tracking_type:
            case TrackingType.BALL:
                point = trackBall(frame, hsv)
            case TrackingType.FACE:
                point = trackFace(frame, face_cascade)

        if point: # Tracking function returned a point
            x, y = point
            now = time.time_ns()

            # Check if `tracking_rate` seconds have passed since `last_send_time`
            if now - last_send_time >= 1_000_000_000 * tracking_rate:
                if tracking_state != TrackingState.TRACKING:
                    print("\r\nTracking object...")
                    tracking_state = TrackingState.TRACKING

                last_send_time = now
                send_data(ser, x, y)
        else: # Didn't detect anything, idle
            now = time.time_ns()

            # Check if `idle_rate` seconds have passed since `last_send_time`
            if now - last_send_time >= 1_000_000_000 * idle_rate:
                if tracking_state != TrackingState.IDLE:
                    print("\r\nIdling...")
                    tracking_state = TrackingState.IDLE

                idle_point = new_idle_point(idle_point)
                last_send_time = now
                send_data(ser, *idle_point)

        cv2.imshow('camera', frame)
        cv2.waitKey(1)

    ser.close()
    cam.release()
    cv2.destroyAllWindows()

try:
    main()
except KeyboardInterrupt:
    pass
