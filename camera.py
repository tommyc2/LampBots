#!/usr/bin/env python3
# Required for forward references
from __future__ import annotations
from typing import Optional, NamedTuple, Callable, Any
from enum import Enum

from glob import glob
import time
import random
import sys
import math

import threading
from threading import Lock
import multiprocessing as mp
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Condition
import ctypes

from PIL import Image, ImageTk, ImageOps
import ttkbootstrap as tb
from ttkbootstrap.constants import SUCCESS, DISABLED, INFO
from tkVideoPlayer import TkinterVideo

import numpy as np
import cv2

from serial import Serial
import serial.tools.list_ports as list_ports
from face_detection.detector import Detector

CameraFrame = np.ndarray

class TrackingType(Enum):
    BALL = 1
    FACE = 2
    NONE = 3

    def __str__(self: TrackingType) -> str:
        match self:
            case TrackingType.BALL: ret = 'Ball'
            case TrackingType.FACE: ret = 'Face'
            case TrackingType.NONE: ret = 'Manual'
        return ret

    @staticmethod
    def from_str(s: str) -> TrackingType:
        match s:
            case 'Ball': ret = TrackingType.BALL
            case 'Face': ret = TrackingType.FACE
            case 'Manual': ret = TrackingType.NONE
            case _: raise ValueError(s)

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
        self.old_value = round(self.get())

    def _value_changed(self, new_value):
        new_value = round(float(new_value))
        if new_value != self.old_value:
            self.old_value = new_value
            self.winfo_toplevel().globalsetvar(self.cget('variable'), (new_value)) #type: ignore
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
        slider_width: int,
        x_cmd: Callable, y_cmd: Callable, z_cmd: Callable,
        x_release: Callable, y_release: Callable, z_release: Callable,
    ) -> None:
        super().__init__(
            parent,
            text = name,
            padding = (10, 10, 10, 10),
            bootstyle = SUCCESS, #type: ignore
        )

        x_frame = tb.Frame(self)
        x_frame.grid(row = 0, column = 0, pady = 5)

        tb.Label(x_frame, text='X').grid(padx = 15)
        self.x_scale = IntScale(
            x_frame,
            from_ = 0,
            to = 180,
            value = 90,
            length = slider_width,
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
            length = slider_width,
            command = y_cmd,
        )
        self.y_scale.grid(column = 1, row = 0)
        self.y_scale.bind('<ButtonRelease-1>', lambda ev: y_release(round(self.y_scale.get())))

        z_frame = tb.Frame(self)
        z_frame.grid(row = 2, column = 0, pady = 5)

        tb.Label(z_frame, text='Z').grid(padx = 15)
        self.z_scale = IntScale(
            z_frame,
            from_ = 82,
            to = 105,
            value = 82,
            length = slider_width,
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

    def resize_scales(self: LampControlWidget, width: int) -> None:
        self.x_scale.configure(length = width)
        self.y_scale.configure(length = width)
        self.z_scale.configure(length = width)

class Gui:
    root: tb.Window
    cam_frame: tb.Frame
    cam_label: tb.Label

    img_frame: tb.Frame
    img_label: tb.Label

    # Sending a tkinter event after window close will block indefinitely, so we need to make sure
    # we don't do that
    event_lock: Lock = Lock()

    closed: bool = False

    lamp_frame: tb.Frame
    lamp1: LampControlWidget
    lamp2: LampControlWidget

    prev_lamp_frame_width: int = 0

    logo_img: Image
    logo_frame: tb.Frame
    logo_label: tb.Label

    prev_logo_width: int = 0
    prev_logo_height: int = 0
    player: TkinterVideo

    app: App
    loading: bool = False

    def on_close(self: Gui) -> None:
        with self.event_lock:
            self.closed = True
            self.root.destroy()
        print('Closing...')

    def on_window_resize(self: Gui, event) -> None:
        if event.widget != self.logo_frame: return

        h, w = self.logo_frame.winfo_height(), self.logo_frame.winfo_width()
        if h < 20 or w < 20: return
        if w == self.prev_logo_width and h == self.prev_logo_height: return
        self.prev_logo_width = w
        self.prev_logo_height = h
        logo = ImageOps.pad(self.logo_img, (w - 20, h - 20))
        logo = ImageTk.PhotoImage(image = logo)
        self.logo_label.photo = logo
        self.logo_label.configure(text = '', image = logo)

    def on_lamp_frame_resize(self: Gui, event) -> None:
        if event.widget != self.lamp_frame: return

        w = self.lamp_frame.winfo_width()
        if w == self.prev_lamp_frame_width: return
        self.prev_lamp_frame_width = w

        self.lamp1.resize_scales(round(w / 5 * 1.75))
        self.lamp2.resize_scales(round(w / 5 * 1.75))

    def __init__(self: Gui, app: App) -> None:
        self.app = app
        self.root = tb.Window(themename = 'darkly')
        self.root.title('Lamp Control Centre')
        self.root.rowconfigure(0, weight = 1)
        self.root.columnconfigure(0, weight = 1)
        self.root.geometry('1280x720')
        self.root.minsize(640, 480)
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

        self.logo_img = Image.open('images/new-logo.png')

        frame = tb.Frame(self.root, padding = (0, 0, 0, 0))
        frame.grid(sticky = 'nsew')
        frame.grid_columnconfigure(10, weight = 2, uniform = 'col')
        frame.grid_columnconfigure(20, weight = 1, uniform = 'col')
        frame.grid_rowconfigure(10)
        frame.grid_rowconfigure(20, weight = 1)
        
        cam_frame = tb.Frame(frame, border = 2, bootstyle = SUCCESS) #type: ignore
        cam_frame.grid(row = 10, column = 10)
        self.cam_frame = cam_frame

        cam_label = tb.Label(cam_frame, text='No camera detected')
        cam_label.grid(row = 10, column = 10)
        self.cam_label = cam_label

        lamp_frame = tb.Frame(frame)
        lamp_frame.grid(row = 20, column = 10, sticky = 'new')
        lamp_frame.bind('<Configure>', self.on_lamp_frame_resize)
        lamp_frame.grid_rowconfigure(0, weight = 1)
        lamp_frame.grid_rowconfigure(1, weight = 1)
        lamp_frame.grid_columnconfigure(0, weight = 1)
        lamp_frame.grid_columnconfigure(1, weight = 1)
        self.lamp_frame = lamp_frame

        info_frame = tb.Frame(frame)
        info_frame.grid(row = 10, column = 20, rowspan = 200, sticky = 'nsew')
        info_frame.grid_rowconfigure(10, weight = 10, uniform = 'guh')
        info_frame.grid_rowconfigure(20, weight = 15, uniform = 'guh')
        info_frame.grid_columnconfigure(10, weight = 1)

        img_frame = tb.Frame(info_frame)
        img_frame.grid(row = 10, column = 10, sticky = 'nsew')
        img_frame.grid_rowconfigure(10, weight = 1)
        img_frame.grid_columnconfigure(10, weight = 1)
        self.img_frame = img_frame

        self.player = TkinterVideo(master = img_frame, scaled = True, consistant_frame_rate = True, keep_aspect = True)
        # self.player.grid(row = 10, column = 10, sticky = 'nsew')
        self.player.bind('<<Ended>>', lambda ev: app.next_image())

        self.img_label = tb.Label(img_frame, text = 'No slides found')
        self.img_label.grid(row = 10, column = 10)

        logo_frame = tb.Frame(info_frame)
        logo_frame.grid(row = 20, column = 10, sticky = 'nsew')
        logo_frame.bind('<Configure>', self.on_window_resize)
        self.logo_frame = logo_frame

        logo_label = tb.Label(logo_frame, text = 'No image')
        logo_label.grid(row = 10, column = 10)
        self.logo_label = logo_label

        input_picker = tb.Combobox(lamp_frame)
        input_picker['values'] = (TrackingType.BALL, TrackingType.FACE, TrackingType.NONE)
        input_picker.state(['readonly'])
        input_picker.set(TrackingType.NONE)
        input_picker.bind(
            '<<ComboboxSelected>>',
            lambda ev: self.app.set_tracking_type(TrackingType.from_str(input_picker.get()))
        )
        input_picker.grid(row = 0, column = 0, columnspan = 2)

        self.lamp1 = LampControlWidget(
            lamp_frame,
            'Lamp 1',
            300,
            lambda x: self.app.limited_move(x1 = x),
            lambda y: self.app.limited_move(y1 = y),
            lambda z: self.app.limited_move(z1 = z),
            lambda x: self.app.move(x1 = x),
            lambda y: self.app.move(y1 = y),
            lambda z: self.app.move(z1 = z),
        )
        self.lamp2 = LampControlWidget(
            lamp_frame,
            'Lamp 2',
            300,
            lambda x: self.app.limited_move(x2 = x),
            lambda y: self.app.limited_move(y2 = y),
            lambda z: self.app.limited_move(z2 = z),
            lambda x: self.app.move(x2 = x),
            lambda y: self.app.move(y2 = y),
            lambda z: self.app.move(z2 = z),
        )

        self.lamp1.grid(row = 1, column = 0, padx = 20)
        self.lamp2.grid(row = 1, column = 1, padx = 20, pady = 10)

        self.root.bind('<<camera_frame>>', lambda _: self.show_frame())

    def set_tracking_type(self: Gui, tracking_type: TrackingType) -> None:
        if tracking_type == tracking_type.NONE:
            self.lamp1.enable_scales()
            self.lamp2.enable_scales()
            self.lamp1.x_scale.set(self.app.last_v1.x)
            self.lamp1.y_scale.set(self.app.last_v1.y)
            self.lamp1.z_scale.set(self.app.last_v1.z)
            self.lamp2.x_scale.set(self.app.last_v2.x)
            self.lamp2.y_scale.set(self.app.last_v2.y)
            self.lamp2.z_scale.set(self.app.last_v2.z)
        else:
            self.lamp1.disable_scales()
            self.lamp2.disable_scales()

    def set_frame(self: Gui, frame: CameraFrame) -> None:
        self.frame = frame
        with self.event_lock:
            if self.closed: return
            self.root.event_generate('<<camera_frame>>')

    def show_frame(self: Gui) -> None:
        h, w = self.frame.shape[:2];
        aspect_ratio = w / h

        win_w = self.root.winfo_width();
        new_width = win_w / 3 * 2 - 10

        frame = cv2.resize(self.frame, (round(new_width), round(new_width / w * h)))
        photo = cv2_frame_to_tk_image(frame)
        self.cam_label.photo = photo # type: ignore
        self.cam_label.configure(text = '', image = photo)  
        # self.cam_frame.configure(width = w, height = h)

    def mainloop(self: Gui) -> None:
        self.root.mainloop()

# Program-wide class to keep all application state in a single place, rather than passing
# tons of function params each time.
class App:
    SERVO_REST: ServoValues = ServoValues(90, 50, 82, 90)
    tracker: Tracker
    pipe: Connection
    running: Any

    tracking_state: Optional[TrackingState] = None
    tracking_rate: float
    idle_rate: float

    ser: Optional[Serial]

    last_idle_point: Optional[ServoValues] = None
    last_send_time: int = 0
    last_v1: ServoValues = SERVO_REST
    last_v2: ServoValues = SERVO_REST

    gui: Gui

    images: list[Image]
    image_index: int = 0

    def next_image(self: App) -> None:
        # For some reason, loading a video triggers a <<Ended>> event, so we need to guard
        # against that
        if self.gui.loading: return

        item = self.images[self.image_index]
        self.image_index = (self.image_index + 1) % len(self.images)

        if isinstance(item, str):
            self.gui.player.grid(row = 10, column = 10, sticky = 'nsew')
            self.gui.img_label.grid_remove()
            self.gui.loading = True
            self.gui.player.load(item)
            self.gui.loading = False
            self.gui.player.play()
            return

        img = item.copy()

        h, w = self.gui.img_frame.winfo_height(), self.gui.img_frame.winfo_width()
        img = ImageOps.pad(img, (w - 5, h - 5))
        img = ImageTk.PhotoImage(image = img)
        self.gui.img_label.photo = img
        self.gui.img_label.configure(text = '', image = img)
        self.gui.player.grid_remove()
        self.gui.img_label.grid(row = 10, column = 10, sticky = 'nsew')
        self.gui.root.after(7500, self.next_image)

    def __init__(
        self: App,
        width: int,
        height: int,
        tracking_rate: float = 0.1,
        idle_rate: float = 3,
        hsv: Hsv = Hsv(hue_low = 29, hue_high = 100, sat_low = 45, val_low = 6),
    ) -> None:
        self.idle_rate = idle_rate
        self.pipe, tracker_pipe = mp.Pipe()
        self.running = mp.Value(ctypes.c_bool, True)
        self.tracker = Tracker(
            tracker_pipe,
            width,
            height,
            self.running,
            hsv = hsv,
            tracking_rate = tracking_rate,
        )

        self.ser = open_serial()
        self.gui = Gui(self)

        paths = glob('slides/*')
        self.images = []
        for f in paths:
            if f.endswith('.jpg') or f.endswith('.png'):
                img = Image.open(f)
                self.images.append(img)
            elif f.endswith('.webm') or f.endswith('.mp4'):
                self.images.append(f)
        if len(self.images) > 0:
            self.gui.root.after(100, self.next_image)

    def __enter__(self: App) -> App:
        return self

    def __exit__(self: App, exc_type, exc_value, traceback) -> None:
        if self.ser is not None:
            # Reset position on exit
            self.send_data(App.SERVO_REST, App.SERVO_REST)
            self.ser.close()

    # Update the tracking type for the Tracker process
    def set_tracking_type(self: App, tracking_type: TrackingType) -> None:
        self.pipe.send(tracking_type)
        self.gui.set_tracking_type(tracking_type)

    # Same as `move`, but only sends data if it has been at least 100 miliseconds
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
        if now - self.last_send_time >= 1e8:
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

    def wait_for_frame(self: App) -> None:
        """
        Waits for frames to come in and generates a tkinter event to display them. Also waits
        for rectangles for tracked objects and sends the appropriate servo values to the lamps.

        Runs on a separate thread. Waits for frames from the camera to be ready using a condition
        variable. When the condition variable gets notified it reads the frame data from shared
        memory, which is written to by `Tracker`. Then checks if any rectangles were sent over
        `self.pipe` from `Tracker`.

        Rects will always be sent right after a frame if the tracking type is FACE or BALL.
        """
        with self.tracker.condition:
            # Wait for notify from `Tracker` process
            self.tracker.condition.wait()

            frame = self.tracker.get_shared_frame().copy()

            has_rects = self.pipe.poll()
            if has_rects:
                r1, r2 = self.pipe.recv()
            else:
                r1 = r2 = None

        # Generate event to display frame.
        # tkinter calls must be made from the main thread, so we need to generate an event rather
        # than doing it directly
        try:
            self.gui.set_frame(frame)
        except RuntimeError:
            return

        match has_rects, get_servo_values(frame, r1, r2):
            case True, None: # Idle
                now = time.time_ns()

                # Check if `idle_rate` seconds have passed since `last_send_time`
                if now - self.last_send_time >= 1e9 * self.idle_rate:
                    if self.tracking_state != TrackingState.IDLE:
                        print("\r\nIdling...")
                        self.tracking_state = TrackingState.IDLE

                    v = new_idle_value(self.last_idle_point)
                    self.last_send_time = now

                    # Currently sends the same point to both lamps
                    self.send_data(v, v)
            case _, (v1, v2):
                if self.tracking_state != TrackingState.TRACKING:
                    print("\r\nTracking object...")
                    self.tracking_state = TrackingState.TRACKING

                self.last_send_time = time.time_ns()
                self.send_data(v1, v2)
    
    def run(self: App) -> None:
        self.tracker.start()

        thread = threading.Thread(target = self.run_camera)
        thread.daemon = True
        thread.start()

        try:
            self.gui.mainloop()
        except KeyboardInterrupt:
            pass
        finally:
            self.running.value = False

            self.tracker.join()
            self.tracker.close()

            thread.join()

    # Run in a separate thread to avoid blocking tkinters event loop.
    def run_camera(self: App) -> None:
        while self.running.value:
            self.wait_for_frame()

class Tracker(mp.Process):
    cap: cv2.VideoCapture
    condition: Condition = mp.Condition()
    tracking_type: TrackingType = TrackingType.NONE
    frame_array: Any
    pipe: Connection
    running: Any

    cam_width: int
    cam_height: int

    last_update_time: int = 0
    last_r1: Optional[Rect] = None
    last_r2: Optional[Rect] = None

    hsv: Hsv
    face_detector: Detector

    tracking_rate: float

    def __init__(
        self: Tracker,
        pipe: Connection,
        cam_width: int,
        cam_height: int,
        running: Any,
        hsv: Hsv,
        tracking_rate,
    ) -> None:
        super(Tracker, self).__init__()

        self.hsv = hsv
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.tracking_rate = tracking_rate
        self.pipe = pipe
        self.running = running
        
        self.frame_array = mp.Array(
            ctypes.c_uint8,
            cam_width * cam_height * 3,
            lock = False,
        )
        self.cap = open_camera(cam_width, cam_height)

    def run(self: Tracker) -> None:
        try:
            # Detector must be initialised in the process that it is used in, so init it here.
            self.face_detector = Detector()
            frame_count: int = 0
            start = time.time_ns()
            fps: float = -1

            while self.running.value:
                success, frame = self.cap.read()
                if not success: continue

                # Check if we received a new tracking type
                if self.pipe.poll():
                    self.tracking_type = self.pipe.recv()
                    if self.tracking_type == TrackingType.NONE:
                        self.last_r1 = self.last_r2 = None

                frame_count += 1
                if frame_count >= 30:
                    end = time.time_ns()
                    fps = 1e9 * frame_count / (end - start)
                    frame_count = 0
                    start = time.time_ns()

                if fps > 0:
                    fps_label = "FPS: %.2f" % fps
                    cv2.putText(
                        frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                    )

                now = time.time_ns()
                update = self.tracking_type != TrackingType.NONE and\
                    now - self.last_update_time >= 1e9 * self.tracking_rate

                if update:
                    self.last_update_time = now

                    r1, r2 = self.track_objects(frame)
                    self.last_r1 = r1
                    self.last_r2 = r2

                # Draw rectangles on frame
                if self.last_r1 is not None: draw_rect(frame, self.last_r1)
                if self.last_r2 is not None: draw_rect(frame, self.last_r2)

                shared_frame = self.get_shared_frame()
                with self.condition:
                    # Write frame to shared memory
                    np.copyto(shared_frame, frame)

                    # Send rectangles to main process if required
                    if update:
                        self.pipe.send((r1, r2)) # type: ignore

                    # Notify that a frame is available
                    self.condition.notify_all()
        except KeyboardInterrupt:
            pass
        finally:
            self.cap.release()
            self.pipe.close()

            # Wake up the consumer if it's waiting
            with self.condition:
                self.condition.notify_all()

    def track_objects(self: Tracker, frame: CameraFrame) -> tuple[Optional[Rect], Optional[Rect]]:
        match self.tracking_type:
            case TrackingType.BALL: r1, r2 = self.track_ball(frame)
            case TrackingType.FACE: r1, r2 = self.track_face(frame)
            case TrackingType.NONE: r1 = r2 = None

        return r1, r2

    def get_shared_frame(self: Tracker) -> CameraFrame:
        buf = np.frombuffer(self.frame_array, np.uint8)
        return buf.reshape(self.cam_height, self.cam_width, 3)

    # Returns two rectangles bounding two faces in the given frame. Returns (r1, None) if only one
    # face is detected, or (None, None) if no faces are detected.
    def track_face(self: Tracker, frame: CameraFrame) -> tuple[Optional[Rect], Optional[Rect]]:
        """
        Find up to two faces in `frame`.

        :return: A tuple of two rectangles, each of which may be `None`
        """
        bboxes: Any = self.face_detector.detect(frame)

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

    def track_ball(self: Tracker, frame: CameraFrame) -> tuple[Optional[Rect], Optional[Rect]]:
        """
        Find round objects matching `self.hue` in `frame`.
        
        :return: A tuple of two rectangles, each which may be `None`
        """
        hsv = self.hsv
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

            return Rect(Point(x, y), Point(x + w, y + h)), None
    
        return None, None

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

def draw_rect(frame: CameraFrame, rect: Rect, color: tuple[int, int, int] = (0, 255, 0)) -> None:
    (x1, y1), (x2, y2) = rect
    cv2.rectangle(frame, (round(x1), round(y1)), (round(x2), round(y2)), color, 2)

# Returns the servo values to send to a lamp for tracking a rectangle
def servo_values_from_rect(frame: CameraFrame, rect: Rect, x_off: float) -> ServoValues:
    h, w = frame.shape[:2]
    (x, y), (x1, y1) = rect

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
) -> Optional[tuple[ServoValues, ServoValues]]:
    match r1, r2:
        case None, None:
            return None
        case [r1 as r2, None] | [None, r2 as r1] | [r1, r2]:
            assert r1 is not None and r2 is not None # Purely to satisfy the type checker :\

            if r2.mid().x < r1.mid().x:
                r1, r2 = r2, r1

            pixels_per_cm = r1.width() / 14.5
            distance_to_lamp = 24 * pixels_per_cm
            v1 = servo_values_from_rect(frame, r1, -distance_to_lamp)
            v2 = servo_values_from_rect(frame, r2, +distance_to_lamp)

            return v1, v2

def open_serial() -> Optional[Serial]:
    # See https://support.microbit.org/support/solutions/articles/19000035697-what-are-the-usb-vid-pid-numbers-for-micro-bit
    MICROBIT_PID = 0x0204
    MICROBIT_VID = 0x0d28
    BAUD_RATE = 115200  # Default baud rate for micro:bit

    serial_file = None
    for p in list_ports.comports():
        if p.vid != MICROBIT_VID or p.pid != MICROBIT_PID:
            continue

        try:
            serial_file = Serial(p.device, BAUD_RATE, timeout=1)
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
    cam = cv2.VideoCapture('/dev/v4l/by-id/usb-WCM_USB_WEB_CAM-video-index0')

    if not cam.isOpened():
        error("Couldn't open camera")
        # TODO: Work without open camera
    else:
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) #type: ignore
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cam.set(cv2.CAP_PROP_FPS, 30)

    return cam


with App(1920, 1080) as app:
    app.run()
