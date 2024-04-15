# Lamp Bot Project

Project iteration on a servo controlled lamp using Python and OpenCV.

# Note:
To minimise/avoid any issues, please run this in a **Linux Environment** either using a VM or dual-boot etc. Popular examples include Ubuntu, Linux Mint etc.

# Usage

Requires at least Python 3.10.

Install dependencies:

```bash
pip3 install -r requirements.txt
```

To manually run application, execute following command:

```bash
run-lamp
```

# Info

## Servo Functions

- **Head Servo:** Moves the lamp's head side to side along the x-axis.
- **Middle Servo:** Adjusts the lamp's head up and down along the y-axis.
- **Bottom Servo:** Moves the lamp back and forwards based on the distance of objects or people, controlled along the z-axis.
- **Servo mounted to plank:** Moves the lamp's body side to side along an imaginary "w" axis.

## Data Transmission
- Data from the OpenCV script is sent from a laptop to a microbit via serial communication.
- The receiving microbit maps coordinates to the servo motors' movement capabilities.

## Dual Lamp Functionality
- Another lamp tracks a separate person when they enter the camera frame, based on a confidence variable.

## Explanation of Servo Operation
- Servos work by receiving PWM signals, adjusting the position of the motor shaft.
- They are DC actuators, converting electrical energy into mechanical movement.
