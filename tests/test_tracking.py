"""
test_tracking.py: Integration tests for TrackingController
"""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch


class TestTrackingController(unittest.TestCase):
    """Test cases for TrackingController class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.frame_shape = (480, 640)
    
    @patch('src.tracking_controller.cv2.KalmanFilter')
    def test_tracking_controller_init(self, mock_kf):
        """Test TrackingController initialization."""
        from src.tracking_controller import TrackingController
        
        controller = TrackingController()
        
        # Verify PID controllers are initialized
        self.assertIsNotNone(controller.pid_left_az)
        self.assertIsNotNone(controller.pid_left_el)
        self.assertIsNotNone(controller.pid_right_az)
        self.assertIsNotNone(controller.pid_right_el)
        
        # Verify state
        self.assertIsNone(controller.last_target_eyes)
        self.assertIsNone(controller.last_laser_dots)
        self.assertEqual(len(controller.error_history), 0)
    
    @patch('src.tracking_controller.cv2.KalmanFilter')
    def test_associate_lasers_simple(self, mock_kf):
        """Test laser association to left/right eyes."""
        from src.tracking_controller import TrackingController
        
        controller = TrackingController()
        
        # Mock laser detections
        laser_detections = [
            {'x': 100.0, 'y': 240.0},  # Left of center
            {'x': 500.0, 'y': 240.0}   # Right of center
        ]
        
        laser_left, laser_right = controller._associate_lasers(
            laser_detections, self.frame_shape
        )
        
        # Verify association
        self.assertAlmostEqual(laser_left[0], 100.0)
        self.assertAlmostEqual(laser_right[0], 500.0)
    
    @patch('src.tracking_controller.cv2.KalmanFilter')
    def test_associate_lasers_empty(self, mock_kf):
        """Test laser association with no detections."""
        from src.tracking_controller import TrackingController
        
        controller = TrackingController()
        
        laser_left, laser_right = controller._associate_lasers(
            [], self.frame_shape
        )
        
        # Both should be None
        self.assertIsNone(laser_left)
        self.assertIsNone(laser_right)
    
    @patch('src.tracking_controller.cv2.KalmanFilter')
    def test_compute_errors_simple(self, mock_kf):
        """Test error computation."""
        from src.tracking_controller import TrackingController
        
        controller = TrackingController()
        
        # Target eyes at origin
        target_eyes = {
            'left': (0.0, 0.0),
            'right': (0.0, 0.0)
        }
        
        # Lasers at frame center (should correspond to 0 error)
        laser_left = (320.0, 240.0)
        laser_right = (320.0, 240.0)
        
        error_left, error_right = controller._compute_errors(
            target_eyes, laser_left, laser_right
        )
        
        # Errors should be small
        self.assertAlmostEqual(error_left[0], 0.0, places=1)
        self.assertAlmostEqual(error_left[1], 0.0, places=1)


if __name__ == '__main__':
    unittest.main()
