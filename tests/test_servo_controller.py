"""
test_servo_controller.py: Unit tests for ServoController
"""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch


class TestServoController(unittest.TestCase):
    """Test cases for ServoController class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock servo limits
        self.servo_limits = np.array([
            [-22.5, 22.5],   # LEFT_ELEVATION
            [-30.0, 30.0],   # LEFT_AZIMUTH
            [-22.5, 22.5],   # RIGHT_ELEVATION
            [-30.0, 30.0],   # RIGHT_AZIMUTH
            [0.0, 90.0],     # TOP_EYELID
            [0.0, 90.0]      # BOTTOM_EYELID
        ], dtype=np.float32)
    
    @patch('src.servo_controller.busio.I2C')
    @patch('src.servo_controller.PCA9685')
    @patch('src.servo_controller.ServoKit')
    def test_servo_controller_init(self, mock_kit, mock_pca, mock_i2c):
        """Test ServoController initialization."""
        from src.servo_controller import ServoController
        
        controller = ServoController(self.servo_limits)
        
        # Verify initialization
        self.assertTrue(np.array_equal(controller.servo_limits, self.servo_limits))
        self.assertEqual(len(controller.current_angles), 6)
        self.assertTrue(np.allclose(controller.current_angles, np.zeros(6)))
    
    @patch('src.servo_controller.busio.I2C')
    @patch('src.servo_controller.PCA9685')
    @patch('src.servo_controller.ServoKit')
    def test_angle_bounds_checking(self, mock_kit, mock_pca, mock_i2c):
        """Test that angles are clipped to bounds."""
        from src.servo_controller import ServoController
        
        controller = ServoController(self.servo_limits)
        mock_servo = MagicMock()
        controller.kit.servo = [mock_servo] * 6
        
        # Try to set out-of-bounds angle
        result = controller.set_angle(0, 50.0)  # Should clip to 22.5
        
        # Verify angle was clipped
        self.assertEqual(mock_servo.angle, 22.5)
    
    @patch('src.servo_controller.busio.I2C')
    @patch('src.servo_controller.PCA9685')
    @patch('src.servo_controller.ServoKit')
    def test_set_eye_angles(self, mock_kit, mock_pca, mock_i2c):
        """Test set_eye_angles convenience method."""
        from src.servo_controller import ServoController
        
        controller = ServoController(self.servo_limits)
        mock_servo = MagicMock()
        controller.kit.servo = [mock_servo] * 6
        
        # Set eye angles
        controller.set_eye_angles(
            left_az=10.0, left_el=5.0,
            right_az=-10.0, right_el=-5.0
        )
        
        # Verify current angles were updated
        self.assertEqual(controller.current_angles[1], 10.0)   # LEFT_AZIMUTH
        self.assertEqual(controller.current_angles[0], 5.0)    # LEFT_ELEVATION
        self.assertEqual(controller.current_angles[3], -10.0)  # RIGHT_AZIMUTH
        self.assertEqual(controller.current_angles[2], -5.0)   # RIGHT_ELEVATION


if __name__ == '__main__':
    unittest.main()
