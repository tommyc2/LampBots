#!/usr/bin/env python3
# Required for forward-references
from __future__ import annotations

import time
import random
import platform
import sys
import math

from enum import Enum
from typing import Optional, NamedTuple, Callable
import threading

import tkinter as tk
from PIL import Image, ImageTk
import ttkbootstrap as tb #type: ignore
from ttkbootstrap.constants import SUCCESS, DISABLED #type: ignore

import numpy as np
import cv2

import serial
import serial.tools.list_ports as list_ports
from face_detection.detector import Detector

CameraFrame = np.ndarray

class TrackingType(Enum):
    BALL = 1
    FACE = 2
    MANUAL = 3

    def __str__(self: TrackingType) -> str:
        match self:
            case TrackingType.BALL: ret = 'Ball'
            case TrackingType.FACE: ret = 'Face'
            case TrackingType.MANUAL: ret = 'Manual'
        return ret

    @staticmethod
    def from_str(s: str) -> TrackingType:
        match s:
            case 'Ball': ret = TrackingType.BALL
            case 'Face': ret = TrackingType.FACE
            case 'Manual': ret = TrackingType.MANUAL

        return ret

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

class ServoValues(NamedTuple):
    x: int
    y: int
    z: int
    w: int

    def bound(self: ServoValues) -> ServoValues:
        x = min(180, max(0, self.x))
        y = min(115, max(50, self.y))
        z = min(105, max(82, self.z))
        w = 90
        return ServoValues(x, y, z, w)

class Point(NamedTuple):
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

class Rect(NamedTuple):
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

class IntScale(tb.Scale):
    old_value: int

    def __init__(self, *args, **kwargs):
        self.chain = kwargs.pop('command', lambda *a: None)
        super(IntScale, self).__init__(*args, command=self._value_changed, **kwargs)
        self.old_value = self.get()

    def _value_changed(self, new_value):
        new_value = round(float(new_value))
        if new_value != self.old_value:
            self.old_value = new_value
            self.winfo_toplevel().globalsetvar(self.cget('variable'), (new_value))
            self.chain(new_value)

# Widget that contains the controls for a single lamp
class LampControlWidget(tb.LabelFrame):
    x_scale: IntScale
    y_scale: IntScale
    z_scale: IntScale

    def __init__(
        self: LampControlWidget,
        parent,
        name: str,
        x_cmd: Callable, y_cmd: Callable, z_cmd: Callable,
        x_release: Callable, y_release: Callable, z_release: Callable,
    ) -> None:
        super().__init__(
            parent,
            text = name,
            padding = (10, 10, 10, 10),
            bootstyle = (SUCCESS),
        )

        x_frame = tb.Frame(self)
        x_frame.grid(row = 0, column = 0, pady = 5)

        tb.Label(x_frame, text='X').grid(padx = 15)
        self.x_scale = IntScale(
            x_frame,
            from_ = 0,
            to = 180,
            value = 90,
            length = 300,
            command = x_cmd,
        )
        self.x_scale.grid(column = 1, row = 0)
        self.x_scale.bind('<ButtonRelease-1>', lambda ev: x_release(round(self.x_scale.get())))

        y_frame = tb.Frame(self)
        y_frame.grid(row = 1, column = 0, pady = 5)

        tb.Label(y_frame, text='Y').grid(padx = 15)
        self.y_scale = IntScale(
            y_frame,
            from_ = 50,
            to = 115,
            value = 70,
            length = 300,
            command = y_cmd,
        )
        self.y_scale.grid(column = 1, row = 0, sticky='nsew')
        self.y_scale.bind('<ButtonRelease-1>', lambda ev: y_release(round(self.y_scale.get())))

        z_frame = tb.Frame(self)
        z_frame.grid(row = 2, column = 0, pady = 5)

        tb.Label(z_frame, text='Z').grid(padx = 15)
        self.z_scale = IntScale(
            z_frame,
            from_ = 82,
            to = 105,
            value = 82,
            length = 300,
            precision = None,
            command = z_cmd,
        )
        self.z_scale.grid(row = 0, column = 1)
        self.z_scale.bind('<ButtonRelease-1>', lambda ev: z_release(round(self.z_scale.get())))

    def disable_scales(self: LampControlWidget) -> None:
        for scale in [self.x_scale, self.y_scale, self.z_scale]:
            scale.configure(state = DISABLED)

    def enable_scales(self: LampControlWidget) -> None:
        for scale in [self.x_scale, self.y_scale, self.z_scale]:
            scale.configure(state = '')

# Program-wide class to keep all application state in a single place, rather than passing
# tons of function params each time.
class App:
    SERVO_REST: ServoValues = ServoValues(90, 50, 82, 90)

    tracking_rate: float
    idle_rate: float
    tracking_type: TrackingType

    width: int
    height: int

    cam: cv2.VideoCapture
    ser: Optional[serial.Serial]

    hsv: Hsv

    tracking_state: Optional[TrackingState] = None
    last_idle_point: Optional[ServoValues] = None
    last_send_time: int = 0
    last_r1: Optional[Rect] = None
    last_r2: Optional[Rect] = None
    last_v1: ServoValues = SERVO_REST
    last_v2: ServoValues = SERVO_REST

    frame: CameraFrame

    root: tb.Window
    cam_label: tb.Label

    lamp1: LampControlWidget
    lamp2: LampControlWidget

    detector: Detector

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
        self.detector = Detector()

        self.hsv = hsv
        self.init_ui()

    def __enter__(self: App) -> App:
        return self

    def __exit__(self: App, exc_type, exc_value, traceback) -> None:
        if self.ser is not None:
            # Reset position on exit
            self.send_data(App.SERVO_REST, App.SERVO_REST)
            self.ser.close()

        self.cam.release()
        cv2.destroyAllWindows()

    def init_ui(self: App) -> None:
        self.root = tb.Window(themename = 'darkly')
        self.root.title('Lamp Control Centre')
        self.root.rowconfigure(0, weight = 1)
        self.root.columnconfigure(0, weight = 1)

        frame = tb.Frame(self.root, padding = (20, 20, 20, 20))
        frame.grid(row = 0, column = 0, sticky = 'nsew')
        
        cam_frame = tb.Frame(frame, width = 640, height = 480, border=2, bootstyle = SUCCESS)
        cam_frame.grid(row = 10, column = 10, sticky = 'nw')
        cam_frame.grid_propagate(False)

        cam_frame.grid_columnconfigure(0, weight = 1)
        cam_frame.grid_rowconfigure(0, weight = 1)

        cam_label = tb.Label(cam_frame, text='No camera detected')
        cam_label.grid(sticky = 'nsew')
        cam_label.configure(anchor = 'center')
        self.cam_label = cam_label

        lamp_frame = tb.Frame(frame)
        lamp_frame.grid(row = 10, column = 20, sticky = 'nw')

        input_picker = tb.Combobox(lamp_frame)
        input_picker['values'] = (TrackingType.BALL, TrackingType.FACE, TrackingType.MANUAL)
        input_picker.state(['readonly'])
        input_picker.set(TrackingType.MANUAL)
        input_picker.bind(
            '<<ComboboxSelected>>',
            lambda ev: self.set_tracking_type(TrackingType.from_str(input_picker.get()))
        )
        input_picker.grid(row = 0, column = 0, columnspan = 2)

        self.lamp1 = LampControlWidget(
            lamp_frame,
            'Lamp 1',
            lambda x: self.limited_move(x1 = x),
            lambda y: self.limited_move(y1 = y),
            lambda z: self.limited_move(z1 = z),
            lambda x: self.move(x1 = x),
            lambda y: self.move(y1 = y),
            lambda z: self.move(z1 = z),
        )
        self.lamp2 = LampControlWidget(
            lamp_frame,
            'Lamp 2',
            lambda x: self.limited_move(x2 = x),
            lambda y: self.limited_move(y2 = y),
            lambda z: self.limited_move(z2 = z),
            lambda x: self.move(x2 = x),
            lambda y: self.move(y2 = y),
            lambda z: self.move(z2 = z),
        )
        self.lamp1.grid(row = 1, column = 0, padx = 20)
        self.lamp2.grid(row = 1, column = 1, padx = 20, pady = 10)

    def set_tracking_type(self: App, tracking_type: TrackingType) -> None:
        if self.tracking_type == tracking_type: return

        if tracking_type == TrackingType.MANUAL:
            self.lamp1.enable_scales()
            self.lamp2.enable_scales()
            self.lamp1.x_scale.set(self.last_v1.x)
            self.lamp1.y_scale.set(self.last_v1.y)
            self.lamp1.z_scale.set(self.last_v1.z)
            self.lamp2.x_scale.set(self.last_v2.x)
            self.lamp2.y_scale.set(self.last_v2.y)
            self.lamp2.z_scale.set(self.last_v2.z)
            self.last_r1 = None
            self.last_r2 = None
        else:
            self.lamp1.disable_scales()
            self.lamp2.disable_scales()
               
        self.tracking_type = tracking_type

    # Same as `move`, but only sends data if it has been at least `self.tracking_rate` seconds
    # since the previous send. Used for sliders so they don't send data too fast.
    def limited_move(
        self: App,
        x1: Optional[int] = None,
        y1: Optional[int] = None,
        z1: Optional[int] = None,
        w1: Optional[int] = None,
        x2: Optional[int] = None,
        y2: Optional[int] = None,
        z2: Optional[int] = None,
        w2: Optional[int] = None,
    ) -> None:
        now = time.time_ns()
        if now - self.last_send_time >= 1_000_000_000 * self.tracking_rate:
            self.last_send_time = now
            self.move(x1, y1, z1, w1, x2, y2, z2, w2)

    # Moves the lamps, allowing for setting one or many servo values. Values set to `None`
    # will remain in the same position.
    def move(
        self: App,
        x1: Optional[int] = None,
        y1: Optional[int] = None,
        z1: Optional[int] = None,
        w1: Optional[int] = None,
        x2: Optional[int] = None,
        y2: Optional[int] = None,
        z2: Optional[int] = None,
        w2: Optional[int] = None,
    ) -> None:
        new_v1 = ServoValues(
            self.last_v1.x if x1 is None else x1,
            self.last_v1.y if y1 is None else y1,
            self.last_v1.z if z1 is None else z1,
            self.last_v1.w if w1 is None else w1,
        )
        new_v2 = ServoValues(
            self.last_v2.x if x2 is None else x2,
            self.last_v2.y if y2 is None else y2,
            self.last_v2.z if z2 is None else z2,
            self.last_v2.w if w2 is None else w2,
        )

        self.send_data(new_v1, new_v2)

    def send_data(self: App, v1: ServoValues, v2: ServoValues) -> None:
        if self.ser is not None:
            try:
                self.ser.write(f'{v1.x},{v1.y},{v1.z},{v1.w}\n{v2.x},{v2.y},{v2.z},{v2.w}\n'.encode())

            except Exception as e:
                error(f"Error: {e}")
                return

        self.last_v1 = v1
        self.last_v2 = v2

        print(f"Sent: {v1.x},{v1.y},{v1.z},{v1.w}    {v2.x},{v2.y},{v2.z},{v2.w}")

    def process_frame(self: App) -> Optional[CameraFrame]:
        ret, frame = self.cam.read()

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

        if ret:
            # Run tracking functions if we have a frame
            match self.tracking_type:
                case TrackingType.BALL:
                    r1 = track_ball(frame, self.hsv)
                    r2 = r1
                case TrackingType.FACE:
                    r1, r2 = self.track_face(frame)
                case TrackingType.MANUAL:
                    # Don't do any automatic tracking here
                    return frame
                case _:
                    assert False

            self.last_r1 = r1
            self.last_r2 = r2
            v1, v2 = get_servo_values(frame, r1, r2)
        elif self.tracking_type == TrackingType.MANUAL:
            return None
        else:
            # Don't have a frame, idle
            v1 = v2 = None

        if v1 is not None and v2 is not None: # Tracking function returned a point
            if self.tracking_state != TrackingState.TRACKING:
                print("\r\nTracking object...")
                self.tracking_state = TrackingState.TRACKING

            self.last_send_time = time.time_ns()
            self.send_data(v1, v2)
        else:
            now = time.time_ns()

            # Check if `idle_rate` seconds have passed since `last_send_time`
            if now - self.last_send_time >= 1_000_000_000 * self.idle_rate:
                if self.tracking_state != TrackingState.IDLE:
                    print("\r\nIdling...")
                    self.tracking_state = TrackingState.IDLE

                x, y, z, w = new_idle_value(self.last_idle_point)
                self.last_send_time = now

                # Currently sends the same point to both lamps
                self.send_data(ServoValues(x, y, z, w), ServoValues(x, y, z, w))

        return frame

    def show_frame(self: App) -> None:
        photo = cv2_frame_to_tk_image(self.frame)
        self.cam_label.photo = photo
        self.cam_label.configure(image = photo, text = '')  
    
    def run(self: App) -> None:
        self.root.bind('<<camera_frame>>', lambda ev: self.show_frame())

        thread = threading.Thread(target = self.run_camera)
        thread.daemon = True
        thread.start()

        self.root.mainloop()

        thread.join()

    # Run in a separate thread to avoid blocking tkinters event loop.
    def run_camera(self: App) -> None:
        while True:
            # No race condition because of GIL
            frame = self.process_frame()
            if frame is None: continue
            self.frame = frame

            try:
                self.root.event_generate('<<camera_frame>>', when='tail')
            except RuntimeError:
                return

    # Returns two rectangles bounding two faces in the given frame. Returns (r1, None) if only one
    # face is detected, or (None, None) if no faces are detected.
    def track_face(self: App, frame) -> tuple[Optional[Rect], Optional[Rect]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bboxes = self.detector.detect(frame)

        if len(bboxes) == 0:
            return None, None

        h, w = frame.shape[:2]

        last_p1 = self.last_r1.mid() if self.last_r1 is not None else Point(0, 0)
        last_p2 = self.last_r2.mid() if self.last_r2 is not None else Point(0, 0)

        r1: Optional[Rect] = None
        r1_closest_dist = w * h

        r2: Optional[Rect] = None
        second_r2: Optional[Rect] = None
        r2_closest_dist = w * h

        for bbox in bboxes:
            x1, y1, x2, y2, _ = bbox
            rect = Rect.from_points(x1, y1, x2, y2)
            d1 = rect.mid().dist(last_p1)
            d2 = rect.mid().dist(last_p2)
            if d1 < r1_closest_dist:
                r1 = rect
            if d2 < r2_closest_dist:
                second_r2 = r2
                r2 = rect

        if r1 == r2:
            r2 = second_r2

        return r1, r2

def cv2_frame_to_tk_image(frame: CameraFrame) -> ImageTk.PhotoImage:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(frame)
    return ImageTk.PhotoImage(image = im)

# Print to stderr
def error(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# Selects a random point from a predefined list of points until it is not `previous_result`,
# and returns it.
def new_idle_value(previous_value: Optional[ServoValues]) -> ServoValues:
    values = [
        ServoValues( 70, 70, 90, 90),
        ServoValues(113, 80, 95, 90),
        ServoValues(127, 90, 99, 90),
        ServoValues(135, 75, 85, 90),
        ServoValues( 98, 75, 82, 90),
        ServoValues( 84, 60, 90, 90),
        ServoValues( 70, 55, 82, 90),
    ]

    ret = random.choice(values)
    # Keep picking until the result is not the same as the previous one
    while ret == previous_value:
        ret = random.choice(values)
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
    return ServoValues(
        round(x),
        round(y / h * 65 + 50), # Map y value to 50-115
        min(105, round(distance / 6) + 82), # Map z value to 82 + distance/6, with max of 105
        90, # TODO: Get w servo working nicely
    )

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

def track_ball(frame: CameraFrame, hsv: Hsv) -> Optional[Rect]:
    height, width = frame.shape[:2]
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    frame_hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([ hsv.hue.low, hsv.sat.low, hsv.val.low ])
    upper_bound = np.array([ hsv.hue.high, hsv.sat.high, hsv.val.high ])
    mask = cv2.inRange(frame_hsv, lower_bound, upper_bound)

    # Erode/dilate passes to remove small artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

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
        cam = cv2.VideoCapture(0)
        # cam = cv2.VideoCapture('/dev/v4l/by-id/usb-WCM_USB_WEB_CAM-video-index0')

    if not cam.isOpened():
        error("Couldn't open camera")
        # TODO: Work without open camera
    else:
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) #type: ignore
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cam.set(cv2.CAP_PROP_FPS, 10)

    return cam


try:
    with App(640, 480, initial_tracking_type = TrackingType.MANUAL) as app:
        app.run()
except KeyboardInterrupt:
    pass
