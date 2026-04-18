"""
test_hardware_discrete.py: Discrete hardware subsystem tests.

Tests individual subsystems in isolation:
- Servo controller (6 channels)
- Camera capture
- Laser detector (HSV thresholding)
- Pose detector
- Laser pulsing (PCA9685 PWM)

Usage:
    python3 test_hardware_discrete.py --test servo
    python3 test_hardware_discrete.py --test camera
    python3 test_hardware_discrete.py --test laser
    python3 test_hardware_discrete.py --test pose
    python3 test_hardware_discrete.py --test all
"""

import sys
import argparse
import time
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.servo_controller import ServoController
from src.laser_detector import LaserController, LaserDetector
from src.pose_detector import PoseDetector


class HardwareTests:
    """Discrete hardware subsystem tests."""
    
    def __init__(self, verbose: bool = True):
        """Initialize test suite."""
        self.verbose = verbose
        self.results = {}
    
    def log(self, msg: str, level: str = "INFO"):
        """Log message."""
        if self.verbose:
            prefix = f"[{level}]"
            print(f"{prefix} {msg}")
    
    # ========================================================================
    # SERVO CONTROLLER TESTS
    # ========================================================================
    
    def test_servo_controller(self):
        """Test servo controller basic functionality."""
        self.log("Testing ServoController...")
        
        try:
            # Initialize with test servo limits
            servo_limits = np.array([
                [-22.5, 22.5],   # LEFT_ELEVATION
                [-30.0, 30.0],   # LEFT_AZIMUTH
                [-22.5, 22.5],   # RIGHT_ELEVATION
                [-30.0, 30.0],   # RIGHT_AZIMUTH
                [0.0, 90.0],     # TOP_EYELID
                [0.0, 90.0]      # BOTTOM_EYELID
            ], dtype=np.float32)
            
            servo = ServoController(servo_limits)
            self.log("✓ ServoController initialized")
            
            # Test center eyes
            servo.center_eyes()
            self.log("✓ Centered eyes")
            
            # Test setting individual angles
            for i in range(6):
                mid_angle = (servo_limits[i, 0] + servo_limits[i, 1]) / 2
                servo.set_angle(i, mid_angle)
                self.log(f"✓ Set servo {i} to {mid_angle:.1f}°")
            
            # Test safe angle limiting
            servo.set_angle(0, 50.0)  # Should be limited
            self.log("✓ Tested angle limiting (hardware would enforce)")
            
            # Test emergency stop
            servo.emergency_stop()
            self.log("✓ Emergency stop executed")
            
            self.results['servo'] = True
            return True
            
        except Exception as e:
            self.log(f"✗ ServoController test failed: {e}", "ERROR")
            self.results['servo'] = False
            return False
    
    # ========================================================================
    # CAMERA CAPTURE TESTS
    # ========================================================================
    
    def test_camera_capture(self, camera_id: int = 0, num_frames: int = 10):
        """Test camera capture and frame statistics."""
        self.log(f"Testing camera capture (device {camera_id})...")
        
        try:
            cap = cv2.VideoCapture(camera_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            if not cap.isOpened():
                self.log("✗ Camera not available", "ERROR")
                self.results['camera'] = False
                return False
            
            self.log(f"✓ Camera opened (640x480 @ 30fps)")
            
            # Capture frames and gather statistics
            frames = []
            frame_times = []
            
            for i in range(num_frames):
                start = time.time()
                ret, frame = cap.read()
                frame_times.append(time.time() - start)
                
                if not ret:
                    self.log(f"✗ Failed to capture frame {i}", "WARN")
                    break
                
                frames.append(frame)
                self.log(f"  Frame {i+1}/{num_frames}: "
                        f"{frame.shape} {frame.dtype} in {frame_times[-1]*1000:.1f}ms")
            
            # Statistics
            if frames:
                avg_read_time = np.mean(frame_times) * 1000
                avg_fps = 1.0 / np.mean(frame_times)
                
                self.log(f"✓ Captured {len(frames)} frames successfully")
                self.log(f"  Avg read time: {avg_read_time:.1f}ms")
                self.log(f"  Avg FPS: {avg_fps:.1f}")
                
                # Check frame characteristics
                frame = frames[0]
                self.log(f"  Frame shape: {frame.shape}")
                self.log(f"  Frame range: [{frame.min()}, {frame.max()}]")
            
            cap.release()
            self.results['camera'] = True
            return True
            
        except Exception as e:
            self.log(f"✗ Camera test failed: {e}", "ERROR")
            self.results['camera'] = False
            return False
    
    # ========================================================================
    # LASER DETECTOR TESTS (HSV Thresholding)
    # ========================================================================
    
    def test_laser_detector_hsv(self, camera_id: int = 0, num_frames: int = 5):
        """Test laser detector with HSV thresholding."""
        self.log("Testing LaserDetector (HSV method)...")
        
        try:
            # Initialize detector
            detector = LaserDetector(
                method="hsv",
                conf_threshold=0.3,
                hsv_h_ranges=[(0, 10), (170, 180)],
                hsv_s_min=80,
                hsv_v_min=100
            )
            self.log("✓ LaserDetector (HSV) initialized")
            
            # Capture test frames
            cap = cv2.VideoCapture(camera_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if not cap.isOpened():
                self.log("✗ Camera not available", "ERROR")
                self.results['laser_hsv'] = False
                return False
            
            detection_times = []
            total_detections = 0
            
            for i in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection
                start = time.time()
                detections, debug_info = detector.detect(frame)
                det_time = time.time() - start
                detection_times.append(det_time)
                total_detections += len(detections)
                
                self.log(f"  Frame {i+1}: {len(detections)} detections in {det_time*1000:.1f}ms")
                
                if debug_info.get('hsv_mask') is not None:
                    mask = debug_info['hsv_mask']
                    mask_pixels = np.count_nonzero(mask)
                    self.log(f"    Mask pixels: {mask_pixels}")
            
            cap.release()
            
            if detection_times:
                avg_time = np.mean(detection_times) * 1000
                self.log(f"✓ HSV detection working")
                self.log(f"  Avg detection time: {avg_time:.1f}ms")
                self.log(f"  Total detections: {total_detections}")
            
            self.results['laser_hsv'] = True
            return True
            
        except Exception as e:
            self.log(f"✗ Laser HSV test failed: {e}", "ERROR")
            self.results['laser_hsv'] = False
            return False
    
    def test_laser_detector_adaptive(self, camera_id: int = 0, num_frames: int = 5):
        """Test laser detector with adaptive thresholding."""
        self.log("Testing LaserDetector (Adaptive method)...")
        
        try:
            # Initialize detector
            detector = LaserDetector(method="adaptive", conf_threshold=0.3)
            self.log("✓ LaserDetector (Adaptive) initialized")
            
            # Capture test frames
            cap = cv2.VideoCapture(camera_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if not cap.isOpened():
                self.log("✗ Camera not available", "ERROR")
                self.results['laser_adaptive'] = False
                return False
            
            detection_times = []
            
            for i in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                start = time.time()
                detections, debug_info = detector.detect(frame)
                det_time = time.time() - start
                detection_times.append(det_time)
                
                self.log(f"  Frame {i+1}: {len(detections)} detections in {det_time*1000:.1f}ms")
            
            cap.release()
            
            if detection_times:
                avg_time = np.mean(detection_times) * 1000
                self.log(f"✓ Adaptive detection working")
                self.log(f"  Avg detection time: {avg_time:.1f}ms")
            
            self.results['laser_adaptive'] = True
            return True
            
        except Exception as e:
            self.log(f"✗ Laser Adaptive test failed: {e}", "ERROR")
            self.results['laser_adaptive'] = False
            return False
    
    # ========================================================================
    # LASER CONTROLLER TEST (PCA9685 Pulsing)
    # ========================================================================
    
    def test_laser_controller(self):
        """Test laser controller PWM pulsing."""
        self.log("Testing LaserController (PCA9685 PWM)...")
        
        try:
            controller = LaserController(pca_channel=7)
            self.log("✓ LaserController initialized on channel 7")
            
            # Test on/off
            controller.on()
            self.log("  Laser ON (5V)")
            time.sleep(0.5)
            
            controller.off()
            self.log("  Laser OFF (0V)")
            time.sleep(0.5)
            
            # Test brightness control
            for brightness in [0.25, 0.5, 0.75, 1.0]:
                controller.set_brightness(brightness)
                self.log(f"  Laser brightness: {brightness*100:.0f}%")
                time.sleep(0.2)
            
            controller.off()
            self.log("✓ LaserController working")
            self.results['laser_pwm'] = True
            return True
            
        except Exception as e:
            self.log(f"✗ LaserController test failed: {e}", "WARN")
            self.log("  (Expected if hardware not available)")
            self.results['laser_pwm'] = False
            return False
    
    # ========================================================================
    # POSE DETECTOR TEST
    # ========================================================================
    
    def test_pose_detector(self, model_path: str, camera_id: int = 0, num_frames: int = 5):
        """Test pose detector."""
        self.log(f"Testing PoseDetector (model: {model_path})...")
        
        try:
            if not Path(model_path).exists():
                self.log(f"✗ Model file not found: {model_path}", "WARN")
                self.results['pose'] = False
                return False
            
            detector = PoseDetector(model_path, conf_threshold=0.5)
            self.log("✓ PoseDetector initialized")
            
            cap = cv2.VideoCapture(camera_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if not cap.isOpened():
                self.log("✗ Camera not available", "ERROR")
                self.results['pose'] = False
                return False
            
            detection_times = []
            total_detections = 0
            
            for i in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                start = time.time()
                pose = detector.detect(frame)
                det_time = time.time() - start
                detection_times.append(det_time)
                
                if pose:
                    total_detections += 1
                    self.log(f"  Frame {i+1}: Face detected in {det_time*1000:.1f}ms")
                else:
                    self.log(f"  Frame {i+1}: No face detected in {det_time*1000:.1f}ms")
            
            cap.release()
            
            if detection_times:
                avg_time = np.mean(detection_times) * 1000
                self.log(f"✓ PoseDetector working")
                self.log(f"  Avg detection time: {avg_time:.1f}ms")
                self.log(f"  Faces detected: {total_detections}/{num_frames}")
            
            self.results['pose'] = True
            return True
            
        except Exception as e:
            self.log(f"✗ PoseDetector test failed: {e}", "ERROR")
            self.results['pose'] = False
            return False
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("HARDWARE TEST SUMMARY")
        print("="*60)
        
        for test_name, passed in self.results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status}: {test_name}")
        
        total = len(self.results)
        passed = sum(1 for v in self.results.values() if v)
        
        print("="*60)
        print(f"Total: {passed}/{total} tests passed")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Hardware Subsystem Tests')
    parser.add_argument('--test', type=str, default='all',
                        choices=['servo', 'camera', 'laser', 'laser-hsv', 'laser-adaptive',
                                 'laser-pwm', 'pose', 'all'],
                        help='Test to run')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--frames', type=int, default=10,
                        help='Number of frames to test')
    parser.add_argument('--pose-model', type=str, default='yolo/yolo11n-pose.pt',
                        help='Path to pose model')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose output')
    
    args = parser.parse_args()
    
    print("="*60)
    print("LASER EYES HARDWARE TEST SUITE")
    print("="*60 + "\n")
    
    tests = HardwareTests(verbose=args.verbose)
    
    try:
        if args.test in ['servo', 'all']:
            tests.test_servo_controller()
        
        if args.test in ['camera', 'all']:
            tests.test_camera_capture(camera_id=args.camera, num_frames=args.frames)
        
        if args.test in ['laser', 'laser-hsv', 'all']:
            tests.test_laser_detector_hsv(camera_id=args.camera, num_frames=args.frames)
        
        if args.test in ['laser', 'laser-adaptive', 'all']:
            tests.test_laser_detector_adaptive(camera_id=args.camera, num_frames=args.frames)
        
        if args.test in ['laser', 'laser-pwm', 'all']:
            tests.test_laser_controller()
        
        if args.test in ['pose', 'all']:
            tests.test_pose_detector(args.pose_model, camera_id=args.camera, 
                                     num_frames=args.frames)
    
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Tests interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    tests.print_summary()


if __name__ == '__main__':
    main()
