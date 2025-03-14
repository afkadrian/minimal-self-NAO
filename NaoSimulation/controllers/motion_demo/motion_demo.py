from controller import Robot, Keyboard, Motion
from PIL import Image
import numpy as np
import io
import os
from flask import Flask, request, jsonify
import threading
import ast
import time # needed for waiting after setting axes
import random

# Flask server for communication
app = Flask(__name__)
#timestemp = datetime.datetime.now()
image_dir = "/Users/adrianschiller/Documents/IT/Minimal_Self_Seminar/Images/"
run_id = 0

IGNORE = False
JOINT_IDS = {
            1 : "HeadYaw",
            2 : "HeadPitch",
            3 : "LShoulderPitch",
            4 : "LShoulderRoll",
            5 : "LElbowYaw",
            6 : "LElbowRoll",
            7 : "LWristYaw",
            8 : "RShoulderPitch",
            9 : "RShoulderRoll",
            10: "RElbowYaw",
            11: "RElbowRoll",
            12: "RWristYaw",
            13: "LHipYawPitch",
            14: "LHipRoll",
            15: "LHipPitch",
            16: "LKneePitch",
            17: "LAnklePitch",
            18: "LAnkleRoll",
            19: "RHipYawPitch",
            20: "RHipRoll",
            21: "RHipPitch",
            22: "RKneePitch",
            23: "RAnklePitch",
            24: "RAnkleRoll"
            }

class Nao(Robot):
    def __init__(self):
        Robot.__init__(self)
        self.currentlyPlaying = False
        self.command = None
        self.motors = {}  # Dictionary to store joint devices
        self.joint_positions = {1: 127, 2: 144, 3: 213, 4: 49,
                                5: 127, 6: 255, 7: 127, 8: 213,
                                9: 206, 10: 128, 11: 0, 12: 128, 
                                13: 155, 14: 90, 15: 200, 16: 11,
                                17: 143, 18: 87, 19: 155, 20: 151,
                                21: 200, 22: 11, 23: 143, 24: 168
                                }  # Dictionary to store target joint positions
        self.joint_translate = {}
        # Initialize stuff
        self.findAndEnableDevices()

    def catpureImage(self, camera, save_path):
        width = camera.getWidth()
        height = camera.getHeight()
        print(save_path)
        image_bytes = camera.getImage()

        image_array = np.frombuffer(image_bytes, dtype=np.uint8).reshape((height, width, 4))
        image_array = image_array[:, :, [2, 1, 0, 3]]
        image_rgb = image_array[:, :, :3]
        img = Image.fromarray(image_rgb, 'RGB')  # RGB-Modus für Pillow
        img.save(os.path.expanduser(save_path))
        print(f"Image saved as {save_path}")
        return True

    def findAndEnableDevices(self):
        # get the time step of the current world.
        self.timeStep = int(self.getBasicTimeStep())
        #n = self.getNumberOfDevices()
        #for i in range(n):
        #    device = self.getDeviceByIndex(i)
        #    print(device.getName())
        # Initialize all joint devices
        self.motors = {
            "HeadYaw": self.getDevice("HeadYaw"),
            "HeadPitch": self.getDevice("HeadPitch"),
            "LShoulderPitch": self.getDevice("LShoulderPitch"),
            "LShoulderRoll": self.getDevice("LShoulderRoll"),
            "LElbowYaw": self.getDevice("LElbowYaw"),
            "LElbowRoll": self.getDevice("LElbowRoll"),
            "LWristYaw": self.getDevice("LWristYaw"),
            "RShoulderPitch": self.getDevice("RShoulderPitch"),
            "RShoulderRoll": self.getDevice("RShoulderRoll"),
            "RElbowYaw": self.getDevice("RElbowYaw"),
            "RElbowRoll": self.getDevice("RElbowRoll"),
            "RWristYaw": self.getDevice("RWristYaw"),
            "LHipYawPitch": self.getDevice("LHipYawPitch"),
            "LHipRoll": self.getDevice("LHipRoll"),
            "LHipPitch": self.getDevice("LHipPitch"),
            "LKneePitch": self.getDevice("LKneePitch"),
            "LAnklePitch": self.getDevice("LAnklePitch"),
            "LAnkleRoll": self.getDevice("LAnkleRoll"),
            "RHipYawPitch": self.getDevice("RHipYawPitch"),
            "RHipRoll": self.getDevice("RHipRoll"),
            "RHipPitch": self.getDevice("RHipPitch"),
            "RKneePitch": self.getDevice("RKneePitch"),
            "RAnklePitch": self.getDevice("RAnklePitch"),
            "RAnkleRoll": self.getDevice("RAnkleRoll")
        }

        for i in range(1, 25):
            self.joint_translate[i] = (((self.motors[JOINT_IDS[i]].getMaxPosition()-self.motors[JOINT_IDS[i]].getMinPosition())/255), self.motors[JOINT_IDS[i]].getMinPosition())

        #for i in range(1, 25):
        #    print("*"*100)
        #    print(JOINT_IDS[i])
        #    print((self.motors[JOINT_IDS[i]].getTargetPosition()-self.motors[JOINT_IDS[i]].getMinPosition())*(255/(self.motors[JOINT_IDS[i]].getMaxPosition()-self.motors[JOINT_IDS[i]].getMinPosition())))
        #    print(f"- Range: {self.motors[JOINT_IDS[i]].getMinPosition()} to {self.motors[JOINT_IDS[i]].getMaxPosition()}\n\t- Default: {self.motors[JOINT_IDS[i]].getTargetPosition()}")

        # Initialize other devices (cameras, sensors, etc.)
        self.cameraTop = self.getDevice("CameraTop")
        self.cameraBottom = self.getDevice("CameraBottom")
        self.cameraTop.enable(4 * self.timeStep)
        self.cameraBottom.enable(4 * self.timeStep)

        self.accelerometer = self.getDevice('accelerometer')
        self.accelerometer.enable(4 * self.timeStep)

        self.gyro = self.getDevice('gyro')
        self.gyro.enable(4 * self.timeStep)

        self.gps = self.getDevice('gps')
        self.gps.enable(4 * self.timeStep)

        self.inertialUnit = self.getDevice('inertial unit')
        self.inertialUnit.enable(self.timeStep)

        self.us = []
        usNames = ['Sonar/Left', 'Sonar/Right']
        for i in range(0, len(usNames)):
            self.us.append(self.getDevice(usNames[i]))
            self.us[i].enable(self.timeStep)

        self.fsr = []
        fsrNames = ['LFsr', 'RFsr']
        for i in range(0, len(fsrNames)):
            self.fsr.append(self.getDevice(fsrNames[i]))
            self.fsr[i].enable(self.timeStep)

        self.lfootlbumper = self.getDevice('LFoot/Bumper/Left')
        self.lfootrbumper = self.getDevice('LFoot/Bumper/Right')
        self.rfootlbumper = self.getDevice('RFoot/Bumper/Left')
        self.rfootrbumper = self.getDevice('RFoot/Bumper/Right')
        self.lfootlbumper.enable(self.timeStep)
        self.lfootrbumper.enable(self.timeStep)
        self.rfootlbumper.enable(self.timeStep)
        self.rfootrbumper.enable(self.timeStep)

        self.leds = []
        self.leds.append(self.getDevice('ChestBoard/Led'))
        self.leds.append(self.getDevice('RFoot/Led'))
        self.leds.append(self.getDevice('LFoot/Led'))
        self.leds.append(self.getDevice('Face/Led/Right'))
        self.leds.append(self.getDevice('Face/Led/Left'))
        self.leds.append(self.getDevice('Ears/Led/Right'))
        self.leds.append(self.getDevice('Ears/Led/Left'))

        self.keyboard = self.getKeyboard()
        self.keyboard.enable(10 * self.timeStep)

    def set_joint(self, joint, angle):
        if angle == 255:
            rad = self.motors[JOINT_IDS[joint]].getMaxPosition()
        elif angle == 0:
            rad = self.motors[JOINT_IDS[joint]].getMinPosition()
        else:
            rad = angle*self.joint_translate[joint][0] + self.joint_translate[joint][1]
        self.motors[JOINT_IDS[joint]].setPosition(rad)
        self.joint_positions[joint] = angle
        return angle
    
    def set_axes(self, axes, angles):
        for ax, angle in zip(axes, angles):
            self.set_joint(ax, angle)

    def run(self):
        """
        Main loop of the controller.
        """
        print("Starting NAO Motion Control Demo.")
        while self.step(self.timeStep) != -1:
            # Apply the target joint positions
            for joint_id, angle in self.joint_positions.items():
                self.set_joint(joint_id, angle)

@app.route('/run_id', methods=['POST'])
def set_run_id():
    data = request.json
    global run_id
    run_id = data.get("id")
    try: os.mkdir(f"{image_dir}run_{run_id}/")
    except Exception as e:
        return jsonify({"status": "failure", "message": f"{e}\nRun ID {run_id} has been used. Please enter another ID"})
    return jsonify({"status": "success", "message": f"Run ID {run_id} has been set"})

@app.route('/set_joints', methods=['POST'])
def set_joints():
    data = request.json
    print("*"*10)
    print(data)
    print("*"*10)
    #code = ast.literal_eval(str(data.get("code")))
    code = data.get("code")
    print(f"Method call: {code}")
    exec(f"{code}")
    return jsonify({"status": "success", "message": "Joint positions updated", "positions": f"{nao.joint_positions}", "code": code})

@app.route('/capture_image', methods=['POST'])
def capture_image():
    data = request.json
    id = data.get("id")
    head_img = f"head_cam{id}.jpeg"
    body_img = f"body_cam{id}.jpeg"
    run_dir = f"{image_dir}run_{run_id}/"
    nao.catpureImage(camera=nao.cameraTop, save_path = run_dir+head_img)
    nao.catpureImage(camera=nao.cameraBottom, save_path = run_dir+body_img)
    return jsonify({"status": "success", "message": "Image saved.",
                    "image_id": id})

def run_flask_server():
    """Run the Flask server in a separate thread."""
    app.run(host='0.0.0.0', port=5000)

# Start the Flask server in a separate thread
flask_thread = threading.Thread(target=run_flask_server)
#flask_thread.daemon = True  # Daemonize thread to exit when the main program exits
flask_thread.start()

# Create the Robot instance and run main loop
nao = Nao()
nao.run()