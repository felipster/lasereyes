# lasereyes bitch
# This project uses a raspberry pi 5 with a camera module v3, the rpi ai kit with the M.2 Hat+, and an adafruit PCA9685 16 Channel servo driver, which drives the 6 servos and two HiLetgo 5V 650nm 5mW red diode laser beams. 
# Here is a final setup of the hardware:
# insert foto
# Setup for the Raspberry pi 5, camera module v3, and rpi ai kit can all be found at https://www.raspberrypi.com/documentation/
# Reference https://discuss.luxonis.com/d/29-ssh-connecting-macbook-pro-to-raspberry-pi-over-direct-ethernet for connecting mac to rpi over ethernet. Need to enable internet sharing to dongle ( AX99179A ) - or whatever rpi is directly connected to computer - and make sure its  connected in networks. Then connect to rpi directly through ssh with the following command:
ssh -X felipster16@raspberrypi.local
# installed XQuartz to enable X11 forwarding to be able to open windows of rpicam video stream over ssh. This was a pain in the ass. need to have command be for stream to work, and also some other settings, and a lot of rpi-reboots and a mac restart lmao:
rpicam-hello -t 10s --qt-preview
# Setup up for the Servo driver can be found at https://learn.adafruit.com/adafruit-16-channel-servo-driver-with-raspberry-pi. The script blinkatest.py is used to make sure your rpi is ready for use with the servo driver, which tests that I2C is accesible through adafruit_blinka CircuitPython solution. installing circuit python can be found here: https://learn.adafruit.com/circuitpython-on-raspberrypi-linux/installing-circuitpython-on-raspberry-pi
# the red lasers are to operate at less than 20 mA, so must connect a resistor (~ 50 ohms) to PWM output of one of the servo channels which is running at maximum of 25 mA (each channel has 220 ohms added on to its 5V+ supply). max 5V / (220 + 50 Ohms) = 18.5 mA Data from https://learn.adafruit.com/16-channel-pwm-servo-driver/pinouts 
# turn on LED with adafruit servo driver:
# need to pip3 install adafruit-circuitpython-pca9685 and adafruit-circuitpython-servokit
import board
import busio
import adafruit_pca9685
i2c = busio.I2C(board.SCL, board.SDA)
pca = adafruit_pca9685.PCA9685(i2c)
pca.frequency = 60
led_channel = pca.channels[0]
led_channel.duty_cycle = 0xffff, full brightnss
# check i2c connections: sudo i2cdetect -y 1
# Implementing Extended Kalman Filter for orientation correction, optimal guidance for target tracking, and train yolov7 for laser dot recognition? Will use azimuth elevetion state (or potentially pitch yaw euler angles?) for each eye, where they are mirror images of eachother -- take advantage of this
# lets find limits of eye motion:
