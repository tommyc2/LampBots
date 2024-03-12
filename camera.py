#!/usr/bin/env python3

# Required for forward-references
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
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
import typing
ServoValues = tuple[int, int, int, int]
Frame = np.ndarray

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

class Point(typing.NamedTuple):
    x: float
    y: float

    def round(self: Point) -> tuple[int, int]:
        return round(self.x), round(self.y)

    def dist(self: Point, p: Point) -> float:
        a = (self.x - p.x)
        b = (self.y - p.y)
        return math.sqrt(a ** 2 + b ** 2)

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y

    def __mul__(self, value) -> Point:
        return Point(self.x * value[0], self.y * value[1])

class Rect(typing.NamedTuple):
    tl: Point
    br: Point

    def mid(self: Rect) -> Point:
        return Point(self.br.x - self.tl.x, self.br.y - self.tl.y)

    def __mul__(self: Rect, value) -> Rect:
        return Rect(self.tl * value, self.br * value)

    def width(self: Rect) -> float:
        return self.br.x - self.tl.x

    def height(self: Rect) -> float:
        return self.br.y - self.tl.y

    @staticmethod
    def from_points(x1, y1, x2, y2) -> Rect:
        return Rect(Point(x1, y1), Point(x2, y2))

# Program-wide class to keep all application state in a single place, rather than passing
# tons of function params each time.
class App:
    tracking_rate: float
    idle_rate: float
    tracking_type: TrackingType

    width: int
    height: int

    cam: cv2.VideoCapture
    ser: Optional[serial.Serial]

    hsv: Hsv
    net = None

    tracking_state: Optional[TrackingState] = None
    last_idle_point: Optional[Point] = None
    last_send_time: int = 0
    last_r1: Optional[Rect] = None
    last_r2: Optional[Rect] = None

    root: tk.Tk
    frame = None
    label = None

    def __init__(
        self: App,
        width: int,
        height: int,
        tracking_rate: float = 0.1,
        idle_rate: float = 3,
        initial_tracking_type: TrackingType = TrackingType.FACE,
        hsv: Hsv = Hsv(hue_low = 29, hue_high = 100, sat_low = 45, val_low = 6),
    ) -> None:
        self.width = width
        self.height = height
        self.tracking_rate = tracking_rate
        self.idle_rate = idle_rate
        self.tracking_type = initial_tracking_type

        self.ser = open_serial()
        self.cam = open_camera(width, height)

        self.hsv = hsv
        self.net = cv2.dnn.readNetFromCaffe('weights-prototxt.txt', 'res_ssd_300Dim.caffeModel')

        self.root = tk.Tk()
        self.frame = ttk.Frame(self.root, padding=10)
        self.frame.grid()
        self.label = ttk.Label(self.frame, text="Hello World!")
        self.label.grid(column = 0, row=0)

    def __enter__(self: App) -> App:
        return self

    def __exit__(self: App, exc_type, exc_value, traceback) -> None:
        if self.ser is not None:
            # Reset position on exit
            send_data(self.ser, (90, 50, 82, 90), (90, 50, 82, 90))

            self.ser.close()

        self.cam.release()
        cv2.destroyAllWindows()

    def process_frame(self: App) -> Optional[Frame]:
        ret, frame = self.cam.read()
        if not ret: return None

        now = time.time_ns()

        # Check if `tracking_rate` seconds have passed since `last_send_time`
        if now - self.last_send_time < 1_000_000_000 * self.tracking_rate:
            if self.last_r1 is not None:
                tl = self.last_r1.tl.round()
                br = self.last_r1.br.round()
                cv2.rectangle(frame, tl, br, (0, 255, 0), 2)
            if self.last_r2 is not None:
                tl = self.last_r2.tl.round()
                br = self.last_r2.br.round()
                cv2.rectangle(frame, tl, br, (0, 255, 0), 2)
            return frame

        match self.tracking_type:
            case TrackingType.BALL:
                r1 = track_ball(frame, self.hsv)
                r2 = r1
            case TrackingType.FACE:
                r1, r2 = track_face(frame, self.net, self.last_r1, self.last_r2)

        self.last_r1 = r1
        self.last_r2 = r2
        v1, v2 = get_servo_values(frame, r1, r2)

        if v1 is not None and v2 is not None: # Tracking function returned a point
            if self.tracking_state != TrackingState.TRACKING:
                print("\r\nTracking object...")
                self.tracking_state = TrackingState.TRACKING

            self.last_send_time = time.time_ns()
            send_data(self.ser, v1, v2)
        else:
            now = time.time_ns()

            # Check if `idle_rate` seconds have passed since `last_send_time`
            if now - self.last_send_time >= 1_000_000_000 * self.idle_rate:
                if self.tracking_state != TrackingState.IDLE:
                    print("\r\nIdling...")
                    self.tracking_state = TrackingState.IDLE

                x, y, z, w = new_idle_point(self.last_idle_point)
                self.last_send_time = now

                # Currently sends the same point to both lamps
                send_data(self.ser, (x, y, z, w), (x, y, z, w))

        return frame

    def show_frame(self: App, frame: Frame) -> None:
        photo = cv2_frame_to_tk_image(frame)

        # solution for bug in `PhotoImage`
        self.label.photo = photo

        # replace image in label
        self.label.configure(image=photo)  

    def show_frames(self: App) -> None:
        frame = self.process_frame()
        if frame is not None:
            self.show_frame(frame)

        self.root.after(1, App.show_frames, self)
    
    def run(self: App) -> None:
        self.show_frames()
        self.root.mainloop()

def cv2_frame_to_tk_image(frame: Frame) -> ImageTk.PhotoImage:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(frame)
    return ImageTk.PhotoImage(image = im)

def send_data(serial_file: Optional[serial.Serial], v1: ServoValues, v2: ServoValues) -> None:
    x1, y1, z1, w1 = v1
    x2, y2, z2, w2 = v2

    if serial_file is not None:
        try:
            serial_file.write(f'{x1},{y1},{z1},{w1}\n{x2},{y2},{z2},{w2}\n'.encode())

        except Exception as e:
            error(f"Error: {e}")

    print(f"Sent: {x1},{y1},{z1},{w1}    {x2},{y2},{z2},{w2}")

# Print to stderr
def error(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# Selects a random point from a predefined list of points until it is not `previous_result`,
# and returns it.
def new_idle_point(previous_result) -> ServoValues:
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

def get_distance(face_width) -> float:
    return 14.5 * 450 / face_width

def get_pixel_dist(face_width: float) -> float:
    pixels_per_cm = face_width / 14.5
    return get_distance(face_width) * pixels_per_cm

# Returns the servo values to send to a lamp for tracking a rectangle
def servo_values_from_rect(frame, rect: Rect, x_off: float) -> ServoValues:
    h, w = frame.shape[:2]
    x, y = (rect.tl.x, rect.tl.y)
    x1, y1 = (rect.br.x, rect.br.y)

    # Draw a green rectangle around the detected face
    cv2.rectangle(frame, (round(x), round(y)), (round(x1), round(y1)), (0, 255, 0), 2)

    # Distance in cm
    distance = get_distance(rect.width())
    pixel_dist = get_pixel_dist(rect.width())

    x = x + (x1 - x) / 2 # Center x coord in box
    y = y + (y1 - y) / 2 # Center y coord in box

    cam_center = w / 2 + x_off

    # The pixel offset from the center of the camera to the x coordinate
    offset = x - cam_center

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

    # diff = angle - 90
    # x = 90 + diff / 4 * 3
    # w = 90 - diff / 4
    x = angle

    # Subtract 120 from distance, with minimum of zero for calculations
    distance = max(0, distance - 120)
    return (
        round(x),
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
        # if distance < 60 or distance > 200: # TODO: Re-enable
        #     continue

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
) -> tuple[Optional[ServoValues], Optional[ServoValues]]:
    if r1 is None:
        assert r2 is None
        return None, None

    # Swap rectangles if the lamps would cross each other
    if r2 is not None and r2.mid().x < r1.mid().x:
        r3 = r2
        r2 = r1
        r1 = r3

    pixels_per_cm = r1.width() / 14.5
    distance_to_lamp1 = 24 * pixels_per_cm
    distance_to_lamp2 = 24 * pixels_per_cm
    v1 = servo_values_from_rect(frame, r1, -distance_to_lamp1)

    # Return v1 twice if we don't have a second face
    if r2 is not None:
        # We have a second face in frame
        v2 = servo_values_from_rect(frame, r2, distance_to_lamp2)
    else:
        v2 = servo_values_from_rect(frame, r1, distance_to_lamp2)

    return v1, v2

def track_ball(frame, hsv: Hsv) -> Optional[Rect]:
    height, width = frame.shape[:2]
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    frame_hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([ hsv.hue.low, hsv.sat.low, hsv.val.low ])
    upper_bound = np.array([ hsv.hue.high, hsv.sat.high, hsv.val.high ])
    mask = cv2.inRange(frame_hsv, lower_bound, upper_bound)

    # Erode/dilate passes to remove small artifacts
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

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

def open_camera(width: int, height: int) -> cv2.VideoCapture:
    if platform.system() == 'Windows':
        cam = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    else:
        # cam = cv2.VideoCapture(0)
        cam = cv2.VideoCapture('/dev/v4l/by-id/usb-WCM_USB_WEB_CAM-video-index0')

    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cam.set(cv2.CAP_PROP_FPS, 10)
    return cam


try:
    with App(640, 480, initial_tracking_type = TrackingType.FACE) as app:
        app.run()
except KeyboardInterrupt:
    pass
