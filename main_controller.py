"""
LaserEyeController: Main orchestrator of the closed-loop laser eye tracking system.
Coordinates all subsystems (servo, pose detection, laser detection, tracking).
"""

from typing import Optional
import time
import numpy as np
import cv2
from .servo_controller import ServoController
from .pose_detector import PoseDetector
from .laser_detector import LaserDetector
from .tracking_controller import TrackingController


class LaserEyeController:
    """
    Main orchestrator: closed-loop laser eye tracking system.
    Coordinates all subsystems (servo, pose detection, laser detection, tracking).
    """
    
    def __init__(self, 
                 servo_limits: np.ndarray,
                 pose_model_path: str,
                 laser_model_path: str,
                 camera_matrix: Optional[np.ndarray] = None,
                 loop_rate_hz: float = 30.0,
                 verbose: bool = False):
        """
        Initialize main controller.
        
        Args:
            servo_limits: 6x2 array of servo angle limits
            pose_model_path: Path to YOLO11-pose weights
            laser_model_path: Path to laser detector weights
            camera_matrix: 3x3 camera intrinsics (if None, uses RPi Camera v3 defaults)
            loop_rate_hz: Target update frequency
            verbose: Print debug info
        """
        self.servo_controller = ServoController(servo_limits)
        self.pose_detector = PoseDetector(pose_model_path, camera_matrix=camera_matrix)
        self.laser_detector = LaserDetector(laser_model_path)
        self.tracking_controller = TrackingController()
        
        self.loop_rate_hz = loop_rate_hz
        self.loop_period = 1.0 / loop_rate_hz
        self.verbose = verbose
        
        self.running = False
        self.frame_count = 0
        
    def run(self, camera_source: int = 0, max_frames: Optional[int] = None):
        """
        Main execution loop.
        
        Args:
            camera_source: OpenCV camera ID (usually 0 for RPi)
            max_frames: Max frames to process (None = infinite)
        """
        cap = cv2.VideoCapture(camera_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.running = True
        loop_start = time.time()
        
        try:
            while self.running:
                frame_start = time.time()
                
                # 1. Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                # 2. Detect target (human eyes)
                pose_detection = self.pose_detector.detect(frame)
                
                # 3. Detect achieved laser position
                laser_detections = self.laser_detector.detect(frame)
                
                # 4. Update tracking and compute commands
                dt = time.time() - frame_start
                commands = self.tracking_controller.update(
                    pose_detection, 
                    laser_detections,
                    dt,
                    frame.shape[:2],
                    self.pose_detector  # Pass for 3D gaze calculation
                )
                
                # 5. Execute servo commands
                self.servo_controller.set_eye_angles(
                    left_az=commands['left_az'],
                    left_el=commands['left_el'],
                    right_az=commands['right_az'],
                    right_el=commands['right_el']
                )
                
                # 6. Logging/visualization (optional)
                if self.verbose:
                    print(f"Frame {self.frame_count}: "
                          f"Target err L: {commands['error_left']}, "
                          f"Target err R: {commands['error_right']}")
                
                self.frame_count += 1
                
                # 7. Rate limiting
                elapsed = time.time() - frame_start
                if elapsed < self.loop_period:
                    time.sleep(self.loop_period - elapsed)
                
                if max_frames and self.frame_count >= max_frames:
                    break
        
        except KeyboardInterrupt:
            print("\nShutdown signal received")
        finally:
            self.shutdown(cap)
    
    def shutdown(self, cap: cv2.VideoCapture):
        """Clean shutdown."""
        self.running = False
        self.servo_controller.emergency_stop()
        cap.release()
        print("Shutdown complete")
