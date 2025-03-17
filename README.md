# lasereyes
# This project uses a raspberry pi 5 with a camera module v3, the rpi ai kit with the M.2 Hat+, and an adafruit PCA9685 16 Channel servo driver, which drives the 6 servos and (maybe) two HiLetgo 5V 650nm 5mW red diode laser beams. 
# Here is a final setup of the hardware:
# insert foto
# Setup for the Raspberry pi 5, camera module v3, and rpi ai kit can all be found at https://www.raspberrypi.com/documentation/
# Setup up for the Servo driver can be found at https://learn.adafruit.com/adafruit-16-channel-servo-driver-with-raspberry-pi. The script blinkatest.py is used to make sure your rpi is ready for use with the servo driver, which tests that I2C is accesible through adafruit_blinka CircuitPython solution. 