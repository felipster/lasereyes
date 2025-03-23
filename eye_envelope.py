import board
import busio
import adafruit_pca9685
from adafruit_servokit import ServoKit
import numpy as np
i2c = busio.I2C(board.SCL, board.SDA)
pca = adafruit_pca9685.PCA9685(i2c)
servo_channels = pca.channels[0:7]
kit =ServoKit(channels=16)
servo_limits = np.array([14,115], # left elevation
                        [0,88], # right azimuth
                        [70,155], # left azimuth
                        [145, 180], # bottom eyelids
                        [60,180], # right elevation
                        [130,75]) # top eyelids

