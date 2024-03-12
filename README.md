# Lamp Bot Project:

### Note: Requires atleast Python 3.10. Use pip3 to install requirements in 'requirements.txt'

- Project Iteration on a servo controlled lamp using OpenCV library & Python.

## Servo Functions:

- <b>Head Servo: </b>Moves the lamp's head side to side along the x-axis.
- <b> Middle Servo:</b>Adjusts the lamp's head up and down along the y-axis.
- <b> Bottom Servo: </b>Moves the lamp back and forwards based on the distance of objects or people, controlled along the z-axis.
- <b>Servo mounted to plank: </b> Moves the lamp's body side to side along an imaginary "w" axis.

## Data Transmission:
- Data from the OpenCV script is sent from a laptop to a microbit via serial communication.
- The receiving microbit maps coordinates to the servo motors' movement capabilities.

## Dual Lamp Functionality: 
- Another lamp tracks a separate person when they enter the camera frame, based on a confidence variable.

## Explanation of Servo Operation:
- Servos work by receiving PWM signals, adjusting the position of the motor shaft.
- They are DC actuators, converting electrical energy into mechanical movement.

## Conclusion:
- The Robotic Lamp project combines robotics, computer vision, and microcontroller technology to create an interactive and engaging experience.
