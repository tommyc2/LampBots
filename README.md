<img class="img-fluid" src="https://github.com/tommyc2/LAMP-Bot-Project/assets/114081733/4ee7a6b1-cdb1-430e-ae1f-a18cac9e826c" alt="logo-lampbots" width=300 height=300>

# Lamp Bot Project

### Important Note: It is recommended to clone instead of fork this repo. This will allow SETU robotics/IoT/electronic students to enhance the model every year :)


Project iteration on a servo controlled lamp using Python and OpenCV.

# Note:
To minimise/avoid any issues, please run this in a **Linux Environment** either using a VM or dual-boot etc. Popular examples include Ubuntu, Linux Mint etc.

** NB: You should NEVER publish any API key to a public repo. When you generate your OpenAI (ChatGPT) key, put the key in a file (e.g. .env file) assigned to a variable. This file will always stay on your machine (not github). That way, you don't have to manually remove your API key from your Python script before pushing to GitHub.

# Usage

Requires at least Python 3.10.

Install dependencies:

```bash
pip3 install -r requirements.txt
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

If in doubt with any of the above, feel free to reach out directly to Tommy, Igor, Evan or Jeff :)
