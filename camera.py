#!/usr/bin/env python3
import numpy as np
import cv2
import serial
import serial.tools.list_ports as list_ports
import time
import random
import platform
import sys
import math
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

    def set_low(self, val: int): self.low = val
    def set_high(self, val: int): self.high = val

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

# Print to stderr
def error(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# Selects a random point from a predefined list of points until it is not `previous_result`,
# and returns it.
def new_idle_point(previous_result):
    points = [
        ( 70, 70, 90,  70),
        (113, 80, 95,  80),
        (127, 90, 99,  90),
        (135, 75, 85, 100),
        ( 98, 75, 82,  90),
        ( 84, 60, 90,  80),
        ( 70, 55, 82,  70),
    ]

    ret = random.choice(points)
    # Keep picking until the result is not the same as the previous one
    while ret == previous_result:
        ret = random.choice(points)
    return ret

def get_confidence(faces, index):
    return faces[0, 0, index, 2]

# Returns the servo values to send to a lamp for tracking a certain face
def servo_values_from_face(frame, faces, index, width, height):
    h, w = frame.shape[:2]

    # Get the upper left and lower right coordinates of the detected face
    box = faces[0, 0, index, 3:7] * np.array([w, h, w, h])
    (x, y, x1, y1) = box.astype("int")

    # Draw a green rectangle around the detected face
    cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

    # Distance in cm
    distance = (14.5 * 450) / (x1 - x)
    pixel_dist = (x1 - x) * 450 / 14.5

    # TODO: Re-enable this check somehow
    # if (distance < 60): continue

    x = x + (x1 - x) / 2 # Center x coord in box
    y = y + (y1 - y) / 2 # Center y coord in box

    # The pixel offset from the center of the camera to the x coordinate
    offset = x - width / 2

    if offset == 0:
        # If offset is zero we would get a division by zero, so special case this.
        # 0 offset means center, which is 90 degrees on the head
        angle = 90
    else:
        # Trigonometry - atan(o/a), where o is the distance and a is the offset
        angle = round(math.degrees((math.atan(pixel_dist / abs(offset)))))

        # Invert the angle if the target is left of center
        if offset < 0:
            angle = 180 - angle

    # Subtract 120 from distance, with minimum of zero for calculations
    distance = max(0, distance - 120)
    return (
        angle,
        round(y / height * 45 + 45), # Map y value to 45-90
        min(105, round(distance / 6) + 82), # Map z value to 82 + distance/6, with max of 105
        80, # TODO: Get w servo working nicely
    )

def track_face(frame, width, height, net):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame)
    net.setInput(blob)
    faces = net.forward()

    # Find the 2 faces with the highest confidence
    best = None
    second = None
    for i in range(faces.shape[2]):
        confidence = get_confidence(faces, i)
        if best is None or confidence > get_confidence(faces, best):
            second = best
            best = i
        elif second is None or confidence > get_confidence(faces, second):
            second = i

    if best is None or get_confidence(faces, best) < 0.5:
        # No confidence >= 0.5
        return None

    p1 = servo_values_from_face(frame, faces, best, width, height)

    # Return p1 twice if we don't have a second face
    p2 = p1
    if second is not None and get_confidence(faces, second) > 0.5:
        # We have a second face in frame
        p2 = servo_values_from_face(frame, faces, second, width, height)

    # If the left lamp (p2) is trying to track a target further right than the other lamp
    # then swap the points
    if p2[0] > p1[0]:
        p3 = p2
        p2 = p1
        p1 = p3

    return (p1, p2)

def track_ball(frame, width, height, hsv):
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    frame_hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([ hsv.hue.low, hsv.sat.low, hsv.val.low ])
    upper_bound = np.array([ hsv.hue.high, hsv.sat.high, hsv.val.high ])
    mask = cv2.inRange(frame_hsv, lower_bound, upper_bound)

    # Erode/dilate passes to remove small artifacts
    mask = cv2.erode(mask, None, iterations=10)
    mask = cv2.dilate(mask, None, iterations=10)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour (presumably the green ball)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Draw a rectangle around the detected ball
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        z = round(w * h / 307200 * 40) + 80
        return (
            x + w / 2,
            y + h / 2,
            z,
            70,
            )
    
    return None

def send_data(file, x1, y1, z1, w1, x2, y2, z2, w2):
    if file is not None:
        try:
            file.write(f'{x1},{y1},{z1},{w1}\n{x2},{y2},{z2},{w2}\n'.encode())

        except Exception as e:
            error(f"Error: {e}")

    print(f"Sent: {x1},{y1},{z1},{w1}    {x2},{y2},{z2},{w2}")

def open_serial():
    # See https://support.microbit.org/support/solutions/articles/19000035697-what-are-the-usb-vid-pid-numbers-for-micro-bit
    MICROBIT_PID = 0x0204
    MICROBIT_VID = 0x0d28
    BAUD_RATE = 115200  # Default baud rate for micro:bit

    serial_file = None
    for p in list_ports.comports():
        if p.vid == MICROBIT_VID and p.pid == MICROBIT_PID:
            try:
                serial_file = serial.Serial(p.device, BAUD_RATE, timeout=1)
            except Exception as e:
                error(
                    f"Could not open serial port: {e}",
                    "Continuing without sending over serial",
                    sep = '',
                )
                break;

            print(f'Found micro:bit at {p.device}')
            break
    else:
        error(
            "No micro:bit connected\n",
            "Continuing without sending over serial",
            sep='',
        )

    return serial_file

def open_camera(width, height):
    if platform.system() == 'Windows':
        cam = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    else:
        cam = cv2.VideoCapture('/dev/v4l/by-id/usb-WCM_USB_WEB_CAM-video-index0')

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cam.set(cv2.CAP_PROP_FPS, 10)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    return cam

def main():
    tracking_rate = 0.1 # How often to send data to microbit in seconds
    idle_rate = 3 # How often to send a movement when idling
    tracking_type = TrackingType.FACE

    width = 640
    height = 480

    ser = open_serial()
    cam = open_camera(width, height)

    hsv = Hsv(
      hue_low = 40,
      hue_high = 80,
      sat_low = 40,
      val_low = 40,
    )

    net = None
    match tracking_type:
        case TrackingType.BALL:
            cv2.namedWindow('tracker')

            cv2.createTrackbar('Hue Low', 'tracker', 40, 179, hsv.hue.set_low)
            cv2.createTrackbar('Hue High', 'tracker', 80, 179, hsv.hue.set_high)
            cv2.createTrackbar('Sat Low', 'tracker', 40, 255, hsv.sat.set_low)
            cv2.createTrackbar('Sat High', 'tracker', 255, 255, hsv.sat.set_high)
            cv2.createTrackbar('Val Low', 'tracker', 40, 255, hsv.val.set_low)
            cv2.createTrackbar('Val High', 'tracker', 255, 255, hsv.val.set_high)
        case TrackingType.FACE:
            net = cv2.dnn.readNetFromCaffe('weights-prototxt.txt', 'res_ssd_300Dim.caffeModel')

    # Keeps track of whether we were idling or tracking,
    # so we can print the new state to the terminal
    tracking_state = None

    # The last point returned by `new_idle_point`
    last_idle_point = None

    last_send_time = time.time_ns()
    while True:
        _, frame = cam.read()

        point1 = None
        point2 = None
        match tracking_type:
            case TrackingType.BALL:
                point1 = track_ball(frame, width, height, hsv)
                point2 = point1
            case TrackingType.FACE:
                p = track_face(frame, width, height, net)
                if p is not None: (point1, point2) = p

        if point1: # Tracking function returned a point
            now = time.time_ns()

            # Check if `tracking_rate` seconds have passed since `last_send_time`
            if now - last_send_time >= 1_000_000_000 * tracking_rate:
                if tracking_state != TrackingState.TRACKING:
                    print("\r\nTracking object...")
                    tracking_state = TrackingState.TRACKING

                last_send_time = now
                send_data(ser, *point1, *point2)
        else:
            now = time.time_ns()

            # Check if `idle_rate` seconds have passed since `last_send_time`
            if now - last_send_time >= 1_000_000_000 * idle_rate:
                if tracking_state != TrackingState.IDLE:
                    print("\r\nIdling...")
                    tracking_state = TrackingState.IDLE

                x, y, z, w = new_idle_point(last_idle_point)
                last_send_time = now
                # Currently sends the same point to both lamps
                send_data(ser, x, y, z, w, x, y, z, w)

        cv2.imshow('camera', frame)
        cv2.waitKey(1)

    ser.close()
    cam.release()
    cv2.destroyAllWindows()

try:
    main()
except KeyboardInterrupt:
    pass
