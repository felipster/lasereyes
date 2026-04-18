#!/usr/bin/env python3
"""
visualize_laser_detection.py: Real-time visualization of laser detection algorithms.

Displays:
- Camera feed with detection overlays
- HSV mask and detection contours
- Algorithm performance metrics
- Real-time plots of detection history

Usage:
    python3 visualize_laser_detection.py --camera 0 --method hsv
    python3 visualize_laser_detection.py --camera 0 --method adaptive
"""

import sys
import argparse
import time
import numpy as np
import cv2
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent))

from src.laser_detector import LaserDetector
from src.camera_capture import CameraCapture


class LaserDetectionVisualizer:
    """Real-time visualization of laser detection."""
    
    def __init__(self, method: str = "hsv", max_history: int = 100):
        """
        Initialize visualizer.
        
        Args:
            method: Detection method
            max_history: Max frames to keep in history
        """
        self.method = method
        self.detector = LaserDetector(method=method, conf_threshold=0.3)
        
        # History for plots
        self.max_history = max_history
        self.frame_history = deque(maxlen=max_history)
        self.detection_history = deque(maxlen=max_history)
        self.confidence_history = deque(maxlen=max_history)
        self.time_history = deque(maxlen=max_history)
        
        self.frame_count = 0
        self.total_detections = 0
    
    def run(self, camera_id: int = 0, max_frames: int = None):
        """Run real-time visualization."""
        try:
            cap = CameraCapture(camera_id=camera_id, width=640, height=480, fps=30)
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            return
        
        print(f"[START] Laser detection visualization ({self.method})")
        print("[CTRL] 'q'=quit, 's'=save frame, 'p'=pause")
        
        paused = False
        frame_to_show = None
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_to_show = frame.copy()
                    
                    # Run detection
                    start_time = time.time()
                    detections, debug_info = self.detector.detect(frame)
                    det_time = time.time() - start_time
                    
                    # Record history
                    self.frame_count += 1
                    self.frame_history.append(self.frame_count)
                    self.detection_history.append(len(detections))
                    self.time_history.append(det_time * 1000)  # ms
                    
                    avg_conf = np.mean([d['confidence'] for d in detections]) if detections else 0
                    self.confidence_history.append(avg_conf)
                    self.total_detections += len(detections)
                    
                    # Create composite visualization
                    viz = self._create_composite(frame, detections, debug_info, det_time)
                else:
                    viz = frame_to_show if frame_to_show is not None else np.zeros((480, 640, 3), np.uint8)
                
                # Display
                cv2.imshow(f'Laser Detection - {self.method.upper()}', viz)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print(f"[PAUSE] Video {'paused' if paused else 'resumed'}")
                elif key == ord('s'):
                    filename = f"laser_detection_{self.method}_{self.frame_count:04d}.png"
                    cv2.imwrite(filename, viz)
                    print(f"[SAVED] {filename}")
                
                if max_frames and self.frame_count >= max_frames:
                    break
        
        except KeyboardInterrupt:
            print("\n[INTERRUPTED]")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._print_summary()
    
    def _create_composite(self, frame: np.ndarray, detections: list, 
                         debug_info: dict, det_time: float) -> np.ndarray:
        """Create composite visualization with multiple panels."""
        height, width = frame.shape[:2]
        
        # Create canvas for composition
        canvas_width = width * 2
        canvas_height = height + 200
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # 1. Draw detections on original frame
        frame_with_det = frame.copy()
        for det in detections:
            x, y = int(det['x']), int(det['y'])
            conf = det['confidence']
            
            color = (0, 255, 0) if conf > 0.7 else (0, 165, 255)
            cv2.circle(frame_with_det, (x, y), 15, color, 2)
            cv2.circle(frame_with_det, (x, y), 3, color, -1)
            cv2.putText(frame_with_det, f"{conf:.2f}", (x+15, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Place in canvas
        canvas[:height, :width] = frame_with_det
        
        # 2. Draw mask or debug image
        if 'hsv_mask' in debug_info:
            mask = debug_info['hsv_mask']
            mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            canvas[:height, width:] = mask_color
        elif 'adaptive_mask' in debug_info:
            mask = debug_info['adaptive_mask']
            mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            canvas[:height, width:] = mask_color
        
        # 3. Info panel
        panel_y = height
        self._draw_info_panel(canvas, 0, panel_y, width, 200, detections, debug_info, det_time)
        self._draw_plots_panel(canvas, width, panel_y, width, 200)
        
        return canvas
    
    def _draw_info_panel(self, canvas: np.ndarray, x: int, y: int, 
                        w: int, h: int, detections: list, debug_info: dict, 
                        det_time: float):
        """Draw information panel."""
        # Background
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (40, 40, 40), -1)
        
        lines = [
            f"Frame: {self.frame_count} | Detections: {len(detections)} | Time: {det_time*1000:.1f}ms",
            f"Method: {debug_info.get('method', 'unknown')} | "
            f"Total: {self.total_detections} | "
            f"Avg conf: {np.mean([d['confidence'] for d in detections]) if detections else 0:.2f}",
            f"Processing time: {np.mean(self.time_history) if self.time_history else 0:.1f}ms avg"
        ]
        
        text_y = y + 20
        for line in lines:
            cv2.putText(canvas, line, (x+10, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            text_y += 25
    
    def _draw_plots_panel(self, canvas: np.ndarray, x: int, y: int, 
                         w: int, h: int):
        """Draw performance plots."""
        # Background
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (40, 40, 40), -1)
        
        if len(self.time_history) < 2:
            cv2.putText(canvas, "Collecting data...", (x+10, y+50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            return
        
        # Plot processing time
        self._draw_graph(canvas, x, y, w//2, h, self.time_history, 
                        title="Time (ms)", color=(0, 255, 0))
        
        # Plot detection count
        self._draw_graph(canvas, x+w//2, y, w//2, h, self.detection_history,
                        title="Detections", color=(0, 165, 255))
    
    def _draw_graph(self, canvas: np.ndarray, x: int, y: int, w: int, h: int,
                   data: deque, title: str = "", color: tuple = (0, 255, 0)):
        """Draw a simple line graph on the canvas."""
        if len(data) < 2:
            return
        
        # Normalize data
        data_array = np.array(data, dtype=np.float32)
        if data_array.max() > 0:
            normalized = (data_array / data_array.max()) * (h - 40)
        else:
            normalized = np.zeros_like(data_array)
        
        # Draw title
        cv2.putText(canvas, title, (x+5, y+15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Draw graph
        offset_y = y + h - 25
        offset_x = x + 5
        
        for i in range(1, len(normalized)):
            x1 = offset_x + (i-1) * w // len(normalized)
            y1 = int(offset_y - normalized[i-1])
            x2 = offset_x + i * w // len(normalized)
            y2 = int(offset_y - normalized[i])
            
            cv2.line(canvas, (x1, y1), (x2, y2), color, 1)
    
    def _print_summary(self):
        """Print summary statistics."""
        print("\n" + "="*60)
        print("LASER DETECTION SUMMARY")
        print("="*60)
        print(f"Method: {self.method}")
        print(f"Total frames: {self.frame_count}")
        print(f"Total detections: {self.total_detections}")
        
        if self.frame_count > 0:
            print(f"Avg detections per frame: {self.total_detections / self.frame_count:.2f}")
        
        if self.time_history:
            print(f"Avg processing time: {np.mean(self.time_history):.2f}ms")
            print(f"Min/Max processing time: {np.min(self.time_history):.2f}ms / "
                  f"{np.max(self.time_history):.2f}ms")
        
        if self.confidence_history:
            avg_conf = np.mean([c for c in self.confidence_history if c > 0])
            print(f"Avg confidence (when detected): {avg_conf:.2f}")
        
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Real-time laser detection visualization'
    )
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--method', type=str, default='hsv',
                        choices=['hsv', 'adaptive', 'temporal', 'hybrid'],
                        help='Detection method')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum frames to process')
    
    args = parser.parse_args()
    
    visualizer = LaserDetectionVisualizer(method=args.method)
    visualizer.run(camera_id=args.camera, max_frames=args.max_frames)


if __name__ == '__main__':
    main()
