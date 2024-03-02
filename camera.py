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
from typing import Optional

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

class Point:
    x: float
    y: float

    def __init__(self: 'Point', x: float, y: float):
        self.x = x
        self.y = y

    def dist(self: 'Point', p: 'Point') -> float:
        a = (self.x - p.x)
        b = (self.y - p.y)
        return math.sqrt(a ** 2 + b ** 2)

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y

    def __mul__(self, value) -> 'Point':
        return Point(self.x * value[0], self.y * value[1])

class Rect:
    tl: Point
    br: Point

    def __init__(self: 'Rect', tl: Point, br: Point):
        self.tl = tl
        self.br = br

    def mid(self: 'Rect') -> Point:
        return Point(self.br.x - self.tl.x, self.br.y - self.tl.y)

    def __mul__(self: 'Rect', value) -> 'Rect':
        return Rect(self.tl * value, self.br * value)

    @staticmethod
    def from_points(x1, y1, x2, y2) -> 'Rect':
        return Rect(Point(x1, y1), Point(x2, y2))

# Print to stderr
def error(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# Selects a random point from a predefined list of points until it is not `previous_result`,
# and returns it.
def new_idle_point(previous_result):
    points = [
        ( 70, 70, 90, 90),
        (113, 80, 95, 90),
        (127, 90, 99, 90),
        (135, 75, 85, 90),
        ( 98, 75, 82, 90),
        ( 84, 60, 90, 90),
        ( 70, 55, 82, 90),
    ]

    ret = random.choice(points)
    # Keep picking until the result is not the same as the previous one
    while ret == previous_result:
        ret = random.choice(points)
    return ret

def get_confidence(faces, index):
    return faces[0, 0, index, 2]

def get_distance(width):
    return 14.5 * 450 / width

# Returns the servo values to send to a lamp for tracking a rectangle
def servo_values_from_rect(frame, rect: Rect, x_off: float) -> tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    x, y = (rect.tl.x, rect.tl.y)
    x1, y1 = (rect.br.x, rect.br.y)

    # Draw a green rectangle around the detected face
    cv2.rectangle(frame, (round(x), round(y)), (round(x1), round(y1)), (0, 255, 0), 2)

    # Distance in cm
    distance = get_distance(x1 - x)
    pixel_dist = (x1 - x) * 450 / 14.5

    # TODO: Re-enable this check somehow
    # if (distance < 60): continue

    x = x + (x1 - x) / 2 # Center x coord in box
    y = y + (y1 - y) / 2 # Center y coord in box

    # The pixel offset from the center of the camera to the x coordinate
    offset = x - w / 2 - + x_off

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
        round(y / h * 65 + 50), # Map y value to 50-115
        min(105, round(distance / 6) + 82), # Map z value to 82 + distance/6, with max of 105
        90, # TODO: Get w servo working nicely
    )

# Returns two rectangles bounding two faces in the given frame. Returns (r1, None) if only one
# face is detected, or (None, None) if no faces are detected.
def track_face(frame, net, last_r1: Optional[Rect], last_r2: Optional[Rect]) -> tuple[Optional[Rect], Optional[Rect]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame)
    net.setInput(blob)
    faces = net.forward()

    new_faces = []
    for f in faces[0, 0]:
        if f[2] < 0.5 or f[3] >= 1 or f[4] >= 1:
            continue

        # Filter faces closer than 60cm
        x1 = f[3] * w
        x2 = f[5] * w
        width = x2 - x1
        distance = get_distance(width)
        if distance < 60:
            continue

        new_faces.append(f)
    faces = new_faces

    if len(faces) == 0:
        return None, None

    h, w = frame.shape[:2]

    last_p1 = last_r1.mid() if last_r1 is not None else Point(0, 0)
    last_p2 = last_r2.mid() if last_r2 is not None else Point(0, 0)

    r1: Optional[Rect] = None
    r1_closest_dist = w * h

    r2: Optional[Rect] = None
    second_r2: Optional[Rect] = None
    r2_closest_dist = w * h

    for f in faces:
        rect = Rect.from_points(*f[3:7]) * [w, h]
        d1 = rect.mid().dist(last_p1)
        d2 = rect.mid().dist(last_p2)
        if d1 < r1_closest_dist:
            r1 = rect
        if d2 < r2_closest_dist:
            second_r2 = r2
            r2 = rect

    if r1 == r2:
        r2 = second_r2

    return (r1, r2)

# Given two rectangles, get the servo values for each one
def get_servo_values(
    frame,
    r1: Optional[Rect],
    r2: Optional[Rect],
) -> tuple[Optional[tuple[int, int, int, int]], Optional[tuple[int, int, int, int]]]:
    if r1 is None:
        assert r2 is None
        return None, None

    # TODO: Find correct offset
    v1 = servo_values_from_rect(frame, r1, 0)

    # Return v1 twice if we don't have a second face
    v2 = v1
    if r2 is not None:
        # We have a second face in frame
        v2 = servo_values_from_rect(frame, r2, 0)

    # If the left lamp (v2) is trying to track a target further right than the other lamp
    # then swap the points
    if v2[0] > v1[0]:
        v3 = v2
        v2 = v1
        v1 = v3

    return v1, v2

def track_ball(frame, hsv: Hsv) -> Optional[Rect]:
    height, width = frame.shape[:2]
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

        return Rect(Point(x, y), Point(x + w, y + h))
    
    return None

def send_data(file, x1, y1, z1, w1, x2, y2, z2, w2):
    if file is not None:
        try:
            file.write(f'{x1},{y1},{z1},{w1}\n{x2},{y2},{z2},{w2}\n'.encode())

        except Exception as e:
            error(f"Error: {e}")

    print(f"Sent: {x1},{y1},{z1},{w1}    {x2},{y2},{z2},{w2}")

def open_serial() -> Optional[serial.Serial]:
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

def open_camera(width: int, height: int):
    if platform.system() == 'Windows':
        cam = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    else:
        # cam = cv2.VideoCapture(0)
        cam = cv2.VideoCapture('/dev/v4l/by-id/usb-WCM_USB_WEB_CAM-video-index0')

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cam.set(cv2.CAP_PROP_FPS, 10)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    return cam

def main() -> None:
    TRACKING_RATE: float = 0.1 # How often to send data to microbit in seconds
    IDLE_RATE: float = 3 # How often to send a movement when idling
    TRACKING_TYPE: TrackingType = TrackingType.FACE

    WIDTH: int = 640
    HEIGHT: int = 480

    ser = open_serial()
    cam = open_camera(WIDTH, HEIGHT)

    HSV: Hsv = Hsv(
      hue_low = 29,
      hue_high = 100,
      sat_low = 45,
      val_low = 6,
    )

    net = None
    match TRACKING_TYPE:
        case TrackingType.BALL:
            cv2.namedWindow('tracker')

            cv2.createTrackbar('Hue Low', 'tracker', 40, 179, HSV.hue.set_low)
            cv2.createTrackbar('Hue High', 'tracker', 80, 179, HSV.hue.set_high)
            cv2.createTrackbar('Sat Low', 'tracker', 40, 255, HSV.sat.set_low)
            cv2.createTrackbar('Sat High', 'tracker', 255, 255, HSV.sat.set_high)
            cv2.createTrackbar('Val Low', 'tracker', 40, 255, HSV.val.set_low)
            cv2.createTrackbar('Val High', 'tracker', 255, 255, HSV.val.set_high)
        case TrackingType.FACE:
            net = cv2.dnn.readNetFromCaffe('weights-prototxt.txt', 'res_ssd_300Dim.caffeModel')

    # Keeps track of whether we were idling or tracking,
    # so we can print the new state to the terminal
    tracking_state = None

    # The last point returned by `new_idle_point`
    last_idle_point = None

    last_send_time = time.time_ns()
    last_r1 = None
    last_r2 = None
    while True:
        _, frame = cam.read()

        match TRACKING_TYPE:
            case TrackingType.BALL:
                r1 = track_ball(frame, HSV)
                r2 = r1
            case TrackingType.FACE:
                r1, r2 = track_face(frame, net, last_r1, last_r2)

        last_r1 = r1
        last_r2 = r2
        v1, v2 = get_servo_values(frame, r1, r2)

        if v1 is not None and v2 is not None: # Tracking function returned a point
            now = time.time_ns()

            # Check if `tracking_rate` seconds have passed since `last_send_time`
            if now - last_send_time >= 1_000_000_000 * TRACKING_RATE:
                if tracking_state != TrackingState.TRACKING:
                    print("\r\nTracking object...")
                    tracking_state = TrackingState.TRACKING

                last_send_time = now
                send_data(ser, *v1, *v2)
        else:
            now = time.time_ns()

            # Check if `idle_rate` seconds have passed since `last_send_time`
            if now - last_send_time >= 1_000_000_000 * IDLE_RATE:
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

ser = open_serial()
send_data(ser, 90, 60, 82, 90, 90, 60, 82, 90)
