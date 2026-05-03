"""
ServoController: Hardware interface to PCA9685 servo driver.
Manages servo angle commands with safety limits.
"""

import numpy as np
from adafruit_extended_bus import ExtendedI2C as I2C
from adafruit_pca9685 import PCA9685


# Standard servo pulse widths (µs) — matches Adafruit ServoKit defaults.
_SERVO_MIN_PULSE_US = 750
_SERVO_MAX_PULSE_US = 2250
_SERVO_ACTUATION_RANGE = 180.0


class ServoController:
    """
    Hardware interface to PCA9685 servo driver.
    Manages servo angle commands with safety limits.

    Exposes self.pca so that other subsystems (e.g. LaserDetector) can share
    the same PCA9685 instance rather than creating a conflicting second one.
    """

    LEFT_ELEVATION = 0
    LEFT_AZIMUTH   = 1
    RIGHT_ELEVATION = 2
    RIGHT_AZIMUTH   = 3
    TOP_EYELID      = 4
    BOTTOM_EYELID   = 5

    def __init__(self, servo_limits: np.ndarray,
                 i2c_bus: int = 2,
                 address: int = 0x40,
                 frequency: int = 50):
        """
        Args:
            servo_limits: 6×2 array of [min_angle, max_angle] per channel
            i2c_bus:   Linux I2C bus number (2 = /dev/i2c-2)
            address:   PCA9685 I2C address
            frequency: PWM frequency in Hz (50 Hz is standard for servos)
        """
        self.servo_limits = servo_limits
        self.current_angles = np.zeros(6)

        i2c = I2C(i2c_bus)
        self.pca = PCA9685(i2c, address=address)
        self.pca.frequency = frequency

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _angle_to_duty_cycle(self, angle: float) -> int:
        """Convert a servo angle (degrees) to a 16-bit PCA9685 duty cycle."""
        period_us = 1_000_000 / self.pca.frequency
        pulse_us = (_SERVO_MIN_PULSE_US
                    + (angle / _SERVO_ACTUATION_RANGE)
                    * (_SERVO_MAX_PULSE_US - _SERVO_MIN_PULSE_US))
        return int(pulse_us / period_us * 0xFFFF)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_angle(self, channel: int, angle: float) -> bool:
        """
        Set servo angle with safety bounds checking.

        Returns True if successful; angle is clamped to limits if out of range.
        """
        min_angle, max_angle = self.servo_limits[channel]

        if not (min_angle <= angle <= max_angle):
            print(f"Warning: Angle {angle} out of bounds [{min_angle}, {max_angle}]")
            angle = np.clip(angle, min_angle, max_angle)

        self.pca.channels[channel].duty_cycle = self._angle_to_duty_cycle(angle)
        self.current_angles[channel] = angle
        return True

    def set_eye_angles(self, left_az: float, left_el: float,
                       right_az: float, right_el: float) -> bool:
        """Convenience: set both eyes at once."""
        self.set_angle(self.LEFT_AZIMUTH,    left_az)
        self.set_angle(self.LEFT_ELEVATION,  left_el)
        self.set_angle(self.RIGHT_AZIMUTH,   right_az)
        self.set_angle(self.RIGHT_ELEVATION, right_el)
        return True

    def get_current_angles(self) -> np.ndarray:
        """Return a copy of the current servo angles."""
        return self.current_angles.copy()

    def center_eyes(self):
        """Move all eye servos to the midpoint of their motion range."""
        for ch in [self.LEFT_ELEVATION, self.LEFT_AZIMUTH,
                   self.RIGHT_ELEVATION, self.RIGHT_AZIMUTH]:
            min_ang, max_ang = self.servo_limits[ch]
            self.set_angle(ch, (min_ang + max_ang) / 2)

    def emergency_stop(self):
        """Center all eyes (safe position)."""
        self.center_eyes()
        print("Emergency stop: eyes centered")
