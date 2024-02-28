import serial
import serial.tools.list_ports as list_ports
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import sys

def error(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
def send_data(file, x1, y1, z1, w1, x2, y2, z2, w2):
    if file is not None:
        try:
            file.write(f"{x1},{y1},{z1},{w1},{x2},{y2},{z2},{w2}\r\n".encode())

        except Exception as e:
            error(f"Error: {e}")

    print(f"Data sent: {x1},{y1},{z1},{w1}    {x2},{y2},{z2},{w2}")


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
                    sep='',
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


def parse_arguments():
    """Parse command line arguments."""
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the (optional) video file")
    ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
    return vars(ap.parse_args())

def get_video_stream(args):
    """Initialize the video stream."""
    if not args.get("video", False):
        return VideoStream(src=0).start()
    else:
        return cv2.VideoCapture(args["video"])

def process_frame(frame, greenLower, greenUpper, args):
    """Process a single frame for ball tracking."""
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    return frame, center



def main():
    lastSendTime = time.time_ns()  # Corrected indentation
    args = parse_arguments()
    greenLower = (29, 45, 6)
    greenUpper = (100, 255, 255)
    pts = deque(maxlen=args["buffer"])

    ser = open_serial()
    vs = get_video_stream(args)
    time.sleep(2.0)

    while True:
        frame = vs.read()
        frame = frame[1] if args.get("video", False) else frame

        if frame is None:
            break

        frame, center = process_frame(frame, greenLower, greenUpper, args)
        now = time.time_ns()
        if center is not None and (now - lastSendTime >= 1_000_000_000):
            x = center[0]
            y = center[1]
            send_data(ser, x, y, 0, 0, x, y, 0, 0)
            lastSendTime = now  # Update lastSendTime after sending data

        pts.appendleft(center)

        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    if not args.get("video", False):
        vs.stop()
    else:
        vs.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
