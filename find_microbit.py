import serial

try:
    PATH = '/dev/edit_here'
    baud_rate = 115200
    # Change directory for Linux, macOS etc.
    with serial.Serial('PATH', baud_rate, timeout=1) as ser:
        print("Microbit found at: ", PATH)
        print("Baud Rate: ", baud_rate)
        # Uncomment here --> exec(open("main.py").read())

except serial.SerialException as e:
    print(f"Error: {e}. Microbit not found or unable to establish serial connection.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")