"""
ServoController: Hardware interface to PCA9685 servo driver.
Manages servo angle commands with safety limits.
"""

import numpy as np
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_servokit import ServoKit


class ServoController:
    """
    Hardware interface to PCA9685 servo driver.
    Manages servo angle commands with safety limits.
    """
    
    # Class-level constants (servo channel mapping)
    LEFT_ELEVATION = 0
    LEFT_AZIMUTH = 1
    RIGHT_ELEVATION = 2
    RIGHT_AZIMUTH = 3
    TOP_EYELID = 4
    BOTTOM_EYELID = 5
    
    def __init__(self, servo_limits: np.ndarray, pca_frequency: int = 30):
        """
        Initialize servo controller.
        
        Args:
            servo_limits: 6x2 numpy array, each row [min_angle, max_angle]
            pca_frequency: PWM frequency for PCA9685 (Hz)
        """
        self.servo_limits = servo_limits
        self.kit = self._init_hardware(pca_frequency)
        self.current_angles = np.zeros(6)
        
    def _init_hardware(self, pca_frequency: int):
        """Initialize I2C communication and ServoKit."""
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c)
        pca.frequency = pca_frequency
        return ServoKit(channels=16)
    
    def set_angle(self, channel: int, angle: float) -> bool:
        """
        Set servo angle with safety bounds checking.
        
        Args:
            channel: Servo channel (0-5)
            angle: Desired angle in degrees
            
        Returns:
            True if successful, False if out of bounds
        """
        min_angle, max_angle = self.servo_limits[channel]
        
        if not (min_angle <= angle <= max_angle):
            print(f"Warning: Angle {angle} out of bounds [{min_angle}, {max_angle}]")
            angle = np.clip(angle, min_angle, max_angle)
        
        self.kit.servo[channel].angle = angle
        self.current_angles[channel] = angle
        return True
    
    def set_eye_angles(self, left_az: float, left_el: float, 
                       right_az: float, right_el: float) -> bool:
        """
        Convenience method: set both eyes at once.
        
        Args:
            left_az, left_el: Left eye azimuth/elevation
            right_az, right_el: Right eye azimuth/elevation
        """
        self.set_angle(self.LEFT_AZIMUTH, left_az)
        self.set_angle(self.LEFT_ELEVATION, left_el)
        self.set_angle(self.RIGHT_AZIMUTH, right_az)
        self.set_angle(self.RIGHT_ELEVATION, right_el)
        return True
    
    def get_current_angles(self) -> np.ndarray:
        """Return current servo angles."""
        return self.current_angles.copy()
    
    def center_eyes(self):
        """Move eyes to center of motion range."""
        for ch in [self.LEFT_ELEVATION, self.LEFT_AZIMUTH, 
                   self.RIGHT_ELEVATION, self.RIGHT_AZIMUTH]:
            min_ang, max_ang = self.servo_limits[ch]
            center = (min_ang + max_ang) / 2
            self.set_angle(ch, center)
    
    def emergency_stop(self):
        """Center all eyes (safe position)."""
        self.center_eyes()
        print("Emergency stop: eyes centered")
