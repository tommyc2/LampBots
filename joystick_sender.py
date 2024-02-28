y2 = 0
x2 = 0
radio.set_group(23)
basic.show_leds("""
    # . . . #
    # . . . #
    # # # # #
    # . . . #
    # . . . #
    """)

def on_forever():
    global x2, y2
    x2 = Math.round(Math.map(input.acceleration(Dimension.X), 0, 1024, 55, 125))
    y2 = Math.round(Math.map(input.acceleration(Dimension.Y), 0, 1024, 45, 90))
    radio.send_value("x", x2)
    radio.send_value("x1", x2)
    radio.send_value("y", y2)
    radio.send_value("y1", y2)
    serial.write_numbers([x2, y2])
basic.forever(on_forever)
