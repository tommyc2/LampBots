y2 = 0
x2 = 0
list2: List[str] = []
radio.set_group(23)
serial.redirect(SerialPin.USB_TX, SerialPin.USB_RX, BaudRate.BAUD_RATE115200)

def on_forever():
    global list2, x2, y2
    list2 = serial.read_line().split(",")
    x2 = Math.round(Math.map(parse_float(list2[0]), 0, 640, 0, 180))
    y2 = Math.round(Math.map(parse_float(list2[1]), 0, 480, 0, 180))
    radio.send_value("x2", x2)
    radio.send_value("y2", y2)
basic.forever(on_forever)