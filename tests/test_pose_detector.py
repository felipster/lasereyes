"""
test_pose_detector.py: Unit tests for PoseDetector
"""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch


class TestPoseDetector(unittest.TestCase):
    """Test cases for PoseDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Default RPi Camera v3 calibration
        self.default_K = np.array([
            [800.0,   0.0, 320.0],
            [  0.0, 800.0, 240.0],
            [  0.0,   0.0,   1.0]
        ], dtype=np.float32)
    
    @patch('src.pose_detector.YOLO')
    def test_pose_detector_init_default_calibration(self, mock_yolo):
        """Test PoseDetector initialization with default calibration."""
        from src.pose_detector import PoseDetector
        
        detector = PoseDetector("dummy_model.pt")
        
        # Verify default K matrix is set
        self.assertTrue(np.allclose(detector.K, self.default_K))
        self.assertEqual(detector.fx, 800.0)
        self.assertEqual(detector.fy, 800.0)
        self.assertEqual(detector.cx, 320.0)
        self.assertEqual(detector.cy, 240.0)
    
    @patch('src.pose_detector.YOLO')
    def test_pose_detector_init_custom_calibration(self, mock_yolo):
        """Test PoseDetector initialization with custom calibration."""
        from src.pose_detector import PoseDetector
        
        custom_K = np.array([
            [1000.0,    0.0, 400.0],
            [   0.0, 1000.0, 300.0],
            [   0.0,    0.0,   1.0]
        ], dtype=np.float32)
        
        detector = PoseDetector("dummy_model.pt", camera_matrix=custom_K)
        
        # Verify custom K matrix is set
        self.assertTrue(np.allclose(detector.K, custom_K))
    
    @patch('src.pose_detector.YOLO')
    def test_pixel_to_normalized_3d(self, mock_yolo):
        """Test pixel to 3D conversion."""
        from src.pose_detector import PoseDetector
        
        detector = PoseDetector("dummy_model.pt")
        
        # Test center pixel (should give approximately [0, 0, 1] normalized)
        ray = detector.pixel_to_normalized_3d(320.0, 240.0)
        expected = np.array([0, 0, 1])
        expected = expected / np.linalg.norm(expected)
        
        self.assertTrue(np.allclose(ray, expected, atol=1e-5))
    
    @patch('src.pose_detector.YOLO')
    def test_gaze_angles_from_3d(self, mock_yolo):
        """Test 3D to angle conversion."""
        from src.pose_detector import PoseDetector
        
        detector = PoseDetector("dummy_model.pt")
        
        # Test forward gaze [0, 0, 1]
        gaze_3d = np.array([0, 0, 1])
        az, el = detector.get_gaze_angles_from_3d(gaze_3d)
        self.assertAlmostEqual(az, 0.0, places=5)
        self.assertAlmostEqual(el, 0.0, places=5)
        
        # Test rightward gaze [1, 0, 1]
        gaze_3d = np.array([1, 0, 1]) / np.sqrt(2)
        az, el = detector.get_gaze_angles_from_3d(gaze_3d)
        self.assertGreater(az, 0)  # Should be positive
        self.assertAlmostEqual(el, 0.0, places=5)


if __name__ == '__main__':
    unittest.main()
