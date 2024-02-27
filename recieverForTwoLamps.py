def on_received_value(name, value):
    global z, w, x, y, x1, y1, z1, w1
    if name == "z":
        z = value
    elif name == "w":
        w = value
    elif name == "x":
        x = value
    elif name == "y":
        y = value
    elif name == "x1":
        x1 = value
    elif name == "y1":
        y1 = value
    elif name == "z1":
        z1 = value
    elif name == "w1":
        w1 = value
radio.on_received_value(on_received_value)

w1 = 0
z1 = 0
y1 = 0
x1 = 0
y = 0
x = 0
w = 0
z = 0
serial.redirect(SerialPin.P13, SerialPin.P1, BaudRate.BAUD_RATE9600)
radio.set_group(23)
EyeX = 3
EyeY = 3
EyeX1 = 3
EyeY1 = 3

def on_forever():
    global EyeY, EyeY1, EyeX, EyeX1
    Kitronik_Robotics_Board.servo_write(Kitronik_Robotics_Board.Servos.SERVO6, z1)
    Kitronik_Robotics_Board.servo_write(Kitronik_Robotics_Board.Servos.SERVO5, w1)
    Kitronik_Robotics_Board.servo_write(Kitronik_Robotics_Board.Servos.SERVO4, x)
    Kitronik_Robotics_Board.servo_write(Kitronik_Robotics_Board.Servos.SERVO3, y)
    Kitronik_Robotics_Board.servo_write(Kitronik_Robotics_Board.Servos.SERVO2, w)
    Kitronik_Robotics_Board.servo_write(Kitronik_Robotics_Board.Servos.SERVO1, z)
    Kitronik_Robotics_Board.servo_write(Kitronik_Robotics_Board.Servos.SERVO8, x1)
    Kitronik_Robotics_Board.servo_write(Kitronik_Robotics_Board.Servos.SERVO7, y1)
    if y > 36:
        EyeY = 5
    elif y > 72:
        EyeY = 4
    elif y > 108:
        EyeY = 3
    elif y > 144:
        EyeY = 2
    else:
        EyeY = 1
    if y1 > 36:
        EyeY1 = 5
    elif y1 > 72:
        EyeY1 = 4
    elif y1 > 108:
        EyeY1 = 3
    elif y1 > 144:
        EyeY1 = 2
    else:
        EyeY1 = 1
    serial.write_number(8)
    serial.write_number(2)
    serial.write_number(EyeY)
    serial.write_number(EyeX)
    if x > 36:
        EyeX = 5
    elif x > 72:
        EyeX = 4
    elif x > 108:
        EyeX = 3
    elif x > 144:
        EyeX = 2
    else:
        EyeX = 1
    if x1 > 36:
        EyeX1 = 5
    elif x1 > 72:
        EyeX1 = 4
    elif x1 > 108:
        EyeX1 = 3
    elif x1 > 144:
        EyeX1 = 2
    else:
        EyeX1 = 1
    serial.write_number(EyeY1)
    serial.write_number(EyeX1)
basic.forever(on_forever)
