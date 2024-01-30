import serial

try:
    PATH = '/dev/edit_here'
    # Change directory for Linux, macOS etc.
    with serial.Serial('PATH', 115200, timeout=1) as ser:
        print("Microbit found and ready for serial communications.")
        # Uncomment here --> exec(open("main.py").read())

except serial.SerialException as e:
    print(f"Error: {e}. Microbit not found or unable to establish serial connection.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")