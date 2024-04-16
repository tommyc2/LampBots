w1 = 0
z1 = 0
y1 = 0
x1 = 0
w = 0
z = 0
y = 0
x = 0
radio.set_group(23)
serial.redirect(SerialPin.USB_TX, SerialPin.USB_RX, BaudRate.BAUD_RATE115200)

def on_forever():
    global x, y, z, w, x1, y1, z1, w1
    list2 = serial.read_line().split(",")
    x = Math.round(parse_float(list2[0]))
    y = Math.round(parse_float(list2[1]))
    z = Math.round(parse_float(list2[2]))
    w = Math.round(parse_float(list2[3]))
    list2 = serial.read_line().split(",")
    x1 = Math.round(parse_float(list2[0]))
    y1 = Math.round(parse_float(list2[1]))
    z1 = Math.round(parse_float(list2[2]))
    w1 = Math.round(parse_float(list2[3]))
    radio.send_value("x1", x1)
    radio.send_value("y1", y1)
    radio.send_value("z1", z1)
    radio.send_value("w1", w1)
    radio.send_value("x", x)
    radio.send_value("y", y)
    radio.send_value("z", z)
    radio.send_value("w", w)
basic.forever(on_forever)
