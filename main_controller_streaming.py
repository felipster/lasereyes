"""
LaserEyeController with streaming visualization: Main orchestrator with live camera streaming.
Visualizes camera feed and algorithm outputs for debugging.
"""

from typing import Optional, Dict
import time
import numpy as np
import cv2
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.servo_controller import ServoController
from src.pose_detector import PoseDetector
from src.laser_detector import LaserDetector
from src.tracking_controller import TrackingController
from src.camera_capture import CameraCapture


class LaserEyeControllerStreaming:
    """
    Main orchestrator with live camera streaming and debug visualization.
    Displays camera feed with detection overlays and algorithm diagnostics.
    """
    
    def __init__(self,
                 servo_limits: np.ndarray,
                 pose_model_path: str,
                 laser_config: Optional[Dict] = None,
                 pose_config: Optional[Dict] = None,
                 camera_matrix: Optional[np.ndarray] = None,
                 loop_rate_hz: float = 30.0,
                 verbose: bool = False,
                 enable_visualization: bool = True):
        """
        Initialize main controller with streaming.

        Args:
            servo_limits: 6x2 array of servo angle limits
            pose_model_path: Path to YOLO11-pose weights
            laser_config: Dict of laser detector settings from config.yaml
            camera_matrix: 3x3 camera intrinsics (if None, uses RPi Camera v3 defaults)
            loop_rate_hz: Target update frequency
            verbose: Print debug info
            enable_visualization: Show camera stream with overlays
        """
        self.servo_controller = ServoController(servo_limits)
        pose_cfg = pose_config or {}
        self.pose_detector = PoseDetector(
            pose_model_path,
            camera_matrix=camera_matrix,
            device=pose_cfg.get('device', 'cpu'),
        )

        cfg = laser_config or {}
        h_ranges_raw = cfg.get('hsv_h_ranges', [[0, 8], [172, 180]])
        self.laser_detector = LaserDetector(
            method=cfg.get('method', 'hsv'),
            conf_threshold=cfg.get('confidence_threshold', 0.5),
            pca_channel1=cfg.get('pca_channel1', 6),
            pca_channel2=cfg.get('pca_channel2', 7),
            hsv_h_ranges=[tuple(r) for r in h_ranges_raw],
            hsv_s_min=cfg.get('hsv_s_min', 150),
            hsv_v_min=cfg.get('hsv_v_min', 150),
            pca=self.servo_controller.pca,  # share the single PCA9685 instance
        )
        
        self.tracking_controller = TrackingController()
        
        self.loop_rate_hz = loop_rate_hz
        self.loop_period = 1.0 / loop_rate_hz
        self.verbose = verbose
        self.enable_visualization = enable_visualization
        
        self.running = False
        self.frame_count = 0
        self.pause_on_detection = False
        
        # Performance tracking
        self.frame_times = []
        self.laser_times = []
        
    def run(self, camera_source: int = 0, max_frames: Optional[int] = None):
        """
        Main execution loop with streaming visualization.
        
        Args:
            camera_source: OpenCV camera ID (usually 0 for RPi)
            max_frames: Max frames to process (None = infinite)
        """
        # Use new camera wrapper instead of cv2.VideoCapture
        try:
            cap = CameraCapture(camera_id=camera_source, width=640, height=480, fps=30)
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            return

        # Wire camera into detector (required for temporal method's on/off captures;
        # also used by all methods for exposure locking via lock_exposure()).
        self.laser_detector.set_camera_capture(cap)

        # Lasers on for all methods; temporal turns them off internally per frame.
        if self.laser_detector.laser_controller1:
            self.laser_detector.laser_controller1.on()
        if self.laser_detector.laser_controller2:
            self.laser_detector.laser_controller2.on()

        self.running = True
        loop_start = time.time()
        
        print("[STREAMING] Starting camera stream...")
        print("[STREAMING] Controls: 'p' = pause, 's' = save frame, 'q' = quit")
        
        try:
            while self.running:
                frame_start = time.time()
                
                # 1. Capture frame using Picamera2 or fallback
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[ERROR] Failed to read frame")
                    break
                
                # 2. Detect target (human eyes)
                pose_detection = self.pose_detector.detect(frame)
                
                # 3. Detect laser position using classical signal processing
                laser_start = time.time()
                laser_detections, laser_debug = self.laser_detector.detect(frame)
                laser_time = time.time() - laser_start
                
                # 4. Create visualization
                viz_frame = self._create_visualization(
                    frame, pose_detection, laser_detections, laser_debug, laser_time
                )
                
                # 5. Display stream
                if self.enable_visualization:
                    cv2.imshow('Laser Eye Tracking - Debug View', viz_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        self.pause_on_detection = not self.pause_on_detection
                    elif key == ord('s'):
                        frame_path = f"debug_frame_{self.frame_count:04d}.png"
                        cv2.imwrite(frame_path, viz_frame)
                        print(f"[SAVED] {frame_path}")
                
                # 6. Execute servo commands (if not paused)
                if not self.pause_on_detection:
                    dt = time.time() - frame_start
                    commands = self.tracking_controller.update(
                        pose_detection, 
                        laser_detections,
                        dt,
                        frame.shape[:2],
                        self.pose_detector
                    )
                    
                    self.servo_controller.set_eye_angles(
                        left_az=commands['left_az'],
                        left_el=commands['left_el'],
                        right_az=commands['right_az'],
                        right_el=commands['right_el']
                    )
                
                # 7. Performance tracking
                frame_time = time.time() - frame_start
                self.frame_times.append(frame_time)
                self.laser_times.append(laser_time)
                
                if self.verbose and self.frame_count % 30 == 0:
                    avg_frame_time = np.mean(self.frame_times[-30:]) * 1000
                    avg_laser_time = np.mean(self.laser_times[-30:]) * 1000
                    print(f"[PERF] Frame {self.frame_count}: "
                          f"Total: {avg_frame_time:.1f}ms, "
                          f"Laser: {avg_laser_time:.1f}ms, "
                          f"Detections: {len(laser_detections)}")
                
                self.frame_count += 1
                
                # 8. Rate limiting
                elapsed = time.time() - frame_start
                if elapsed < self.loop_period:
                    time.sleep(self.loop_period - elapsed)
                
                if max_frames and self.frame_count >= max_frames:
                    break
        
        except KeyboardInterrupt:
            print("\n[SHUTDOWN] Keyboard interrupt received")
        except Exception as e:
            print(f"[ERROR] Exception: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.shutdown(cap)
    
    def _create_visualization(self, frame: np.ndarray, 
                              pose_detection: Dict,
                              laser_detections: list,
                              laser_debug: Dict,
                              laser_time: float) -> np.ndarray:
        """Create visualization overlay on frame."""
        viz = frame.copy()
        height, width = frame.shape[:2]
        
        # 1. Draw laser detections
        for det in laser_detections:
            x, y = int(det['x']), int(det['y'])
            conf = det['confidence']
            
            # Draw circle at detection
            color = (0, 255, 0) if conf > 0.7 else (0, 165, 255)
            cv2.circle(viz, (x, y), 15, color, 2)
            cv2.circle(viz, (x, y), 3, color, -1)
            
            # Draw confidence text
            cv2.putText(viz, f"{conf:.2f}", (x+20, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 2. Draw pose keypoints (if detected)
        if pose_detection and 'keypoints' in pose_detection:
            for kpt in pose_detection['keypoints']:
                if len(kpt) >= 3:
                    x, y, conf = kpt[0], kpt[1], kpt[2]
                    if conf > 0.5:
                        cv2.circle(viz, (int(x), int(y)), 5, (255, 0, 0), -1)
        
        # 3. Draw debug info panel
        panel_height = 120
        panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        panel.fill(30)
        
        # Add text info
        y_offset = 20
        info_lines = [
            f"Frame: {self.frame_count}",
            f"Method: {laser_debug.get('method', 'unknown')} | "
            f"Detections: {laser_debug.get('num_detections', 0)} | "
            f"Time: {laser_time*1000:.1f}ms",
            f"Pose: {'✓' if pose_detection else '✗'} | "
            f"Laser: {len(laser_detections)} dots"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(panel, line, (10, y_offset + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 4. Combine visualization
        viz_final = np.vstack([viz, panel])
        
        # 5. Add HSV mask if available
        if 'hsv_mask' in laser_debug and self.enable_visualization:
            mask = laser_debug['hsv_mask']
            mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            # Scale masks to fit
            h, w = mask.shape[:2]
            scale = min(200 / h, 200 / w)
            mask_scaled = cv2.resize(mask_color, (int(w*scale), int(h*scale)))
            
            # Place mask in corner
            viz_final[:mask_scaled.shape[0], -mask_scaled.shape[1]:] = mask_scaled
        
        return viz_final
    
    def shutdown(self, cap: cv2.VideoCapture):
        """Clean shutdown."""
        self.running = False
        
        # Print performance stats
        if self.frame_times:
            avg_time = np.mean(self.frame_times) * 1000
            max_time = np.max(self.frame_times) * 1000
            print(f"\n[STATS] Processed {self.frame_count} frames")
            print(f"[STATS] Avg frame time: {avg_time:.1f}ms")
            print(f"[STATS] Max frame time: {max_time:.1f}ms")
            print(f"[STATS] Avg FPS: {1/np.mean(self.frame_times):.1f}")
        
        if self.laser_detector.laser_controller1:
            self.laser_detector.laser_controller1.off()
        if self.laser_detector.laser_controller2:
            self.laser_detector.laser_controller2.off()
        self.servo_controller.emergency_stop()
        cap.release()
        cv2.destroyAllWindows()
        print("[SHUTDOWN] Complete")
