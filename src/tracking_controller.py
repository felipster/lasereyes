"""
TrackingController: Multi-target tracking and command generation.
Tracks target eyes (desired) and achieved laser dots.
Computes servo commands via Kalman filtering and PID control.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from .pid_controller import PIDController


class TrackingController:
    """
    Multi-target tracking and command generation.
    Tracks target eyes (desired) and achieved laser dots.
    Computes servo commands via Kalman filtering and PID control.
    """
    
    def __init__(self, process_noise: float = 0.1, 
                 measurement_noise: float = 1.0,
                 kp: float = 0.5, ki: float = 0.1, kd: float = 0.2):
        """
        Initialize tracking controller.
        
        Args:
            process_noise: Kalman filter process noise
            measurement_noise: Kalman filter measurement noise
            kp, ki, kd: PID controller gains
        """
        # Kalman filters (one for left eye, one for right eye)
        self.kf_left = self._init_kalman_filter(process_noise, measurement_noise)
        self.kf_right = self._init_kalman_filter(process_noise, measurement_noise)
        
        # PID controllers (one for each servo axis)
        self.pid_left_az = PIDController(kp, ki, kd)
        self.pid_left_el = PIDController(kp, ki, kd)
        self.pid_right_az = PIDController(kp, ki, kd)
        self.pid_right_el = PIDController(kp, ki, kd)
        
        # State tracking
        self.last_target_eyes = None
        self.last_laser_dots = None
        self.error_history = []
        
    def _init_kalman_filter(self, Q: float, R: float):
        """Initialize Kalman filter for position tracking."""
        # Simple 2D position tracking: state = [x, y, vx, vy]
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * Q
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * R
        return kf
    
    def update(self, pose_detection: Dict, laser_detections: List[Dict], 
               dt: float, frame_shape: Tuple[int, int],
               pose_detector: 'PoseDetector') -> Dict:
        """
        Update tracking state and compute servo commands.
        
        Args:
            pose_detection: From PoseDetector.detect()
            laser_detections: From LaserDetector.detect()
            dt: Time since last update (seconds)
            frame_shape: (height, width) of frame
            pose_detector: PoseDetector instance (for 3D gaze calculation)
            
        Returns:
            {
                'left_az': float (degrees),
                'left_el': float (degrees),
                'right_az': float (degrees),
                'right_el': float (degrees),
                'error_left': (az_error, el_error),
                'error_right': (az_error, el_error),
                'tracking_confidence': float
            }
        """
        commands = {
            'left_az': 0, 'left_el': 0,
            'right_az': 0, 'right_el': 0,
            'error_left': (0, 0), 'error_right': (0, 0),
            'tracking_confidence': 0.0
        }
        
        # 1. Extract target eye positions using camera-calibrated 3D gaze
        if not pose_detection['detected']:
            return commands
        
        target_eyes = self._extract_target_positions(pose_detection, frame_shape, pose_detector)
        
        # 2. Associate laser dots to eyes (simple: closest match)
        laser_left, laser_right = self._associate_lasers(laser_detections, frame_shape)
        
        # 3. Compute errors
        error_left, error_right = self._compute_errors(target_eyes, laser_left, laser_right)
        
        # 4. Update Kalman filters with measurements
        if laser_left:
            self.kf_left.correct(np.array([[laser_left[0]], [laser_left[1]]], dtype=np.float32))
        if laser_right:
            self.kf_right.correct(np.array([[laser_right[0]], [laser_right[1]]], dtype=np.float32))
        
        # 5. Generate PID commands
        cmd_left_az = self.pid_left_az.update(error_left[0], dt)
        cmd_left_el = self.pid_left_el.update(error_left[1], dt)
        cmd_right_az = self.pid_right_az.update(error_right[0], dt)
        cmd_right_el = self.pid_right_el.update(error_right[1], dt)
        
        commands['left_az'] = cmd_left_az
        commands['left_el'] = cmd_left_el
        commands['right_az'] = cmd_right_az
        commands['right_el'] = cmd_right_el
        commands['error_left'] = error_left
        commands['error_right'] = error_right
        commands['tracking_confidence'] = pose_detection['confidence']
        
        self.last_target_eyes = target_eyes
        self.last_laser_dots = [laser_left, laser_right]
        self.error_history.append((error_left, error_right))
        
        return commands
    
    def _extract_target_positions(self, pose_detection: Dict, frame_shape: Tuple[int, int],
                                  pose_detector: 'PoseDetector') -> Dict:
        """
        Extract eye positions from pose detection using proper 3D gaze calculation.
        
        Now uses camera-calibrated 3D gaze direction instead of simplified 2D projection.
        """
        # Get 3D gaze vector (proper calculation using camera intrinsics)
        gaze_3d = pose_detector.get_gaze_direction_3d(pose_detection)
        
        if gaze_3d is None:
            return {'left': (0, 0), 'right': (0, 0)}
        
        # Convert 3D gaze vector to spherical angles
        az, el = pose_detector.get_gaze_angles_from_3d(gaze_3d)
        
        # Clamp to reasonable servo ranges
        az = np.clip(az, -30, 30)
        el = np.clip(el, -22.5, 22.5)
        
        # Both eyes track the same target (assuming single person)
        return {
            'left': (az, el),
            'right': (az, el)
        }
    
    def _associate_lasers(self, laser_detections: List[Dict], 
                          frame_shape: Tuple[int, int]) -> Tuple[Optional[Tuple], Optional[Tuple]]:
        """
        Associate detected laser dots to left/right eyes.
        Simple heuristic: assign to closest eye.
        """
        if not laser_detections:
            return None, None
        
        h, w = frame_shape
        center_x = w / 2
        
        laser_left = None
        laser_right = None
        
        for laser in laser_detections:
            x = laser['x']
            # Assign to left if x < center, right if x > center
            if x < center_x and (laser_left is None or 
                                 abs(x - center_x) < abs(laser_left[0] - center_x)):
                laser_left = (laser['x'], laser['y'])
            elif x >= center_x and (laser_right is None or 
                                    abs(x - center_x) < abs(laser_right[0] - center_x)):
                laser_right = (laser['x'], laser['y'])
        
        return laser_left, laser_right
    
    def _compute_errors(self, target_eyes: Dict, laser_left: Optional[Tuple], 
                        laser_right: Optional[Tuple]) -> Tuple[Tuple, Tuple]:
        """
        Compute error between target and achieved laser positions.
        
        Returns: ((left_az_err, left_el_err), (right_az_err, right_el_err))
        """
        h, w = 480, 640  # Assume RPi camera resolution (customize as needed)
        
        # Convert laser pixel positions back to angles
        def pixel_to_angle(px, py):
            az = (px - w/2) / w * 60
            el = (h/2 - py) / h * 45
            return (az, el)
        
        target_left_az, target_left_el = target_eyes['left']
        target_right_az, target_right_el = target_eyes['right']
        
        if laser_left:
            achieved_left_az, achieved_left_el = pixel_to_angle(laser_left[0], laser_left[1])
            error_left = (target_left_az - achieved_left_az, target_left_el - achieved_left_el)
        else:
            error_left = (target_left_az, target_left_el)
        
        if laser_right:
            achieved_right_az, achieved_right_el = pixel_to_angle(laser_right[0], laser_right[1])
            error_right = (target_right_az - achieved_right_az, target_right_el - achieved_right_el)
        else:
            error_right = (target_right_az, target_right_el)
        
        return error_left, error_right
