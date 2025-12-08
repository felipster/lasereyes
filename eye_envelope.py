import board
import busio
import adafruit_pca9685
from adafruit_servokit import ServoKit
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

# THESE ARE THE LIMITS FOUND, DO NOT CHANGE THE SERVO ARMS' POSITION RELATIVE TO THE SERVO, 
# AND KEEP THE POINT OF WHERE THE HINGES ARE CONNECTED CONSTANT OR YOU'LL HAVE TO REPEAT THIS ASS PROCESS
servo_limits = np.array([[34,80], #  0 left elevation
                        [64,132], #  1 left azimuth
                        [0,43], #  2 right elevation
                        [32,122], #  3 right azimuth
                        [60, 90], # 4 top eyelids
                        [120, 150]]) # 5 bottom eyelids
print(servo_limits)




def loop_thru_azEl(az_limits, el_limits, az_chan, el_chan, kit):
    # switching logic
    going_up = True
    lag = 1e-3

    for az in range(az_limits[0],az_limits[1]):
        kit.servo[az_chan].angle = az
        for el in range(el_limits[0], el_limits[1]):
            if going_up:
                #print("el: " + str(el) + ", az: " + str(az))
                kit.servo[el_chan].angle = el
                time.sleep(lag)
            else: # going_down
                el_flip = el_limits[1] - el + el_limits[0]
                #print("el: " + str(el_flip) + ", az: " + str(az))
                kit.servo[el_chan].angle = el_flip
                time.sleep(lag)
        if going_up:
            going_up = False
        else:
            going_up = True

def set_initials(servo_limits, kit):
    # set initial values of eyes and eyelids
    kit.servo[0].angle = (servo_limits[0,1] - servo_limits[0,0] )/2 + servo_limits[0,0] # left el
    kit.servo[1].angle = (servo_limits[1,1] - servo_limits[1,0] )/2 + servo_limits[1,0]  # right az
    kit.servo[2].angle = (servo_limits[2,1] - servo_limits[2,0] )/2 + servo_limits[2,0] -5# left az
    kit.servo[3].angle = (servo_limits[3,1] - servo_limits[3,0] )/2 + servo_limits[3,0] +5# right az
    kit.servo[4].angle = servo_limits[4,1]
    kit.servo[5].angle = servo_limits[5,1]

if __name__ == '__main__':
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = adafruit_pca9685.PCA9685(i2c)
    pca.frequency = 30
    kit =ServoKit(channels=16)


    # set initial values of eyes and eyelids
    set_initials(servo_limits, kit)

    # right eye
    az_chan_r = 3
    el_chan_r = 2
    loop_thru_azEl(servo_limits[az_chan_r,:], servo_limits[el_chan_r,:], az_chan_r, el_chan_r, kit)

    # left eye
    az_chan_l = 1
    el_chan_l = 0
    loop_thru_azEl(servo_limits[az_chan_l,:], servo_limits[el_chan_l,:], az_chan_l, el_chan_l, kit)

    set_initials(servo_limits,kit)