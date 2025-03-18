# lasereyes
# This project uses a raspberry pi 5 with a camera module v3, the rpi ai kit with the M.2 Hat+, and an adafruit PCA9685 16 Channel servo driver, which drives the 6 servos and (maybe) two HiLetgo 5V 650nm 5mW red diode laser beams. 
# Here is a final setup of the hardware:
# insert foto
# Setup for the Raspberry pi 5, camera module v3, and rpi ai kit can all be found at https://www.raspberrypi.com/documentation/
# Setup up for the Servo driver can be found at https://learn.adafruit.com/adafruit-16-channel-servo-driver-with-raspberry-pi. The script blinkatest.py is used to make sure your rpi is ready for use with the servo driver, which tests that I2C is accesible through adafruit_blinka CircuitPython solution. 
# the red lasers are to operate at less than 20 mA, so must connect a resistor (~ 50 ohms) to PWM output of one of the servo channels which is running at maximum of 25 mA (each channel has 220 ohms added on to its 5V+ supply). max 5V / (220 + 50 Ohms) = 18.5 mA Data from https://learn.adafruit.com/16-channel-pwm-servo-driver/pinouts 
# turn on LED with adafruit servo driver:
need to pip3 install adafruit-circuitpython-pca9685 and adafruit-circuitpython-servokit
import board
import busio
import adafruit_pca9685
i2c = busio.I2C(board.SCL, board.SDA)
pca = adafruit_pca9685.PCA9685(i2c)
pca.frequency = 60
led_channel = pca.channels[0]
led_channel.duty_cycle = 0xffff, full brightnss
# Implementing Extended Kalman Filter for orientation correction, optimal guidance for target tracking, and train yolov7 for laser dot recognition? Will use azimuth elevetion state (or potentially pitch yaw euler angles?) for each eye, where they are mirror images of eachother -- take advantage of this