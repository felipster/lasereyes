import board
import busio
import adafruit_pca9685
from adafruit_servokit import ServoKit
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


i2c = busio.I2C(board.SCL, board.SDA)
pca = adafruit_pca9685.PCA9685(i2c)
servo_channels = pca.channels[0:7]
kit =ServoKit(channels=16)
# THESE ARE THE LIMITS FOUND, DO NOT CHANGE THE SERVO ARMS' POSITION RELATIVE TO THE SERVO, 
# AND KEEP THE POINT OF WHERE THE HINGES ARE CONNECTED CONSTANT OR YOU'LL HAVE TO REPEAT THIS ASS PROCESS
servo_limits = np.array([[14,115], #  0 left elevation
                        [0,88], #     1 right azimuth
                        [70,155], #   2 left azimuth
                        [145, 180], # 3 bottom eyelids
                        [65,170], #   4 right elevation
                        [75, 130]]) # 5 top eyelids
print(servo_limits)

# set initial values or right eye and eyelids
kit.servo[1].angle = (servo_limits[1,1] - servo_limits[1,0] )/2 + servo_limits[1,0] + 20
kit.servo[3].angle = servo_limits[3,1]
kit.servo[4].angle = (servo_limits[4,1] - servo_limits[4,0] )/2 + servo_limits[4,0]
kit.servo[5].angle = servo_limits[5,1]

# switching logic
going_up = True

# loop through left eyes positions
for az in range(servo_limits[1,0],servo_limits[1,1]):
    kit.servo[1].angle = az
    for el in range(servo_limits[4,0], servo_limits[4,1]):
        if going_up:
            print("el: " + str(el) + ", az: " + str(az))
            #kit.servo[4].angle = el
        else: # going_down
            el_flip = servo_limits[4,1] - el + servo_limits[4,0]
            print("el: " + str(el_flip) + ", az: " + str(az))
            #kit.servo[4].angle = el_flip
    if going_up:
        going_up = False
    else:
        going_up = True
