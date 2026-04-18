#!/usr/bin/env python3
"""
tune_hsv_interactive.py: Interactive HSV threshold tuning with live camera feed.

Displays trackbars to adjust HSV thresholds in real-time.
Perfect for finding optimal detection parameters for your specific lighting.

Usage:
    python3 tune_hsv_interactive.py --camera 0
"""

import sys
import argparse
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.camera_capture import CameraCapture


class HSVTuner:
    """Interactive HSV threshold tuner."""
    
    def __init__(self):
        """Initialize tuner with default red values."""
        # Default red laser values
        self.h_min_1 = 0
        self.h_max_1 = 10
        self.h_min_2 = 170
        self.h_max_2 = 180
        self.s_min = 80
        self.v_min = 100
    
    def create_trackbars(self, window_name: str):
        """Create OpenCV trackbars for parameter adjustment."""
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        # Red range 1 (lower)
        cv2.createTrackbar('H_Min_1', window_name, self.h_min_1, 10,
                          lambda x: self._update('h_min_1', x))
        cv2.createTrackbar('H_Max_1', window_name, self.h_max_1, 10,
                          lambda x: self._update('h_max_1', x))
        
        # Red range 2 (upper)
        cv2.createTrackbar('H_Min_2', window_name, self.h_min_2, 180,
                          lambda x: self._update('h_min_2', x))
        cv2.createTrackbar('H_Max_2', window_name, self.h_max_2, 180,
                          lambda x: self._update('h_max_2', x))
        
        # Saturation and Value
        cv2.createTrackbar('S_Min', window_name, self.s_min, 255,
                          lambda x: self._update('s_min', x))
        cv2.createTrackbar('V_Min', window_name, self.v_min, 255,
                          lambda x: self._update('v_min', x))
    
    def _update(self, param: str, value: int):
        """Update parameter value."""
        setattr(self, param, value)
    
    def get_mask(self, frame: np.ndarray) -> np.ndarray:
        """Get HSV mask for current thresholds."""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for red (two ranges)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        # Red range 1 (0-10)
        lower1 = np.array([self.h_min_1, self.s_min, self.v_min])
        upper1 = np.array([self.h_max_1, 255, 255])
        mask |= cv2.inRange(hsv, lower1, upper1)
        
        # Red range 2 (170-180)
        lower2 = np.array([self.h_min_2, self.s_min, self.v_min])
        upper2 = np.array([self.h_max_2, 255, 255])
        mask |= cv2.inRange(hsv, lower2, upper2)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def count_detections(self, mask: np.ndarray) -> int:
        """Count connected components in mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area
        count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5 < area < 500:  # Typical laser dot size
                count += 1
        
        return count
    
    def run(self, camera_id: int = 0):
        """Run interactive tuner."""
        try:
            cap = CameraCapture(camera_id=camera_id, width=640, height=480, fps=30)
        except Exception as e:
            print(f"[ERROR] Camera initialization failed: {e}")
            return
        
        window_name = "HSV Threshold Tuner"
        self.create_trackbars(window_name)
        
        print("="*60)
        print("HSV THRESHOLD TUNER")
        print("="*60)
        print("\nControls:")
        print("  Adjust trackbars to tune HSV thresholds")
        print("  'q' - quit and print final values")
        print("  's' - save screenshot")
        print("  'r' - reset to defaults")
        print("\nRed laser optimal ranges:")
        print("  Hue: 0-10 and 170-180")
        print("  Sat: 80-255")
        print("  Val: 100-255")
        print("="*60 + "\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get mask
                mask = self.get_mask(frame)
                
                # Get detections
                detections = self.count_detections(mask)
                
                # Create visualization
                mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                
                # Highlight detected regions
                frame_with_dets = frame.copy()
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 5 < area < 500:
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.circle(frame_with_dets, (x+w//2, y+h//2), 10, (0, 255, 0), 2)
                
                # Combine images side by side
                h_space = np.ones((frame.shape[0], 20, 3), dtype=np.uint8) * 100
                combined = np.hstack([frame_with_dets, h_space, mask_color])
                
                # Add info text
                info_text = f"Detections: {detections} | "
                info_text += f"H1:[{self.h_min_1}-{self.h_max_1}] "
                info_text += f"H2:[{self.h_min_2}-{self.h_max_2}] "
                info_text += f"S:{self.s_min} V:{self.v_min}"
                
                cv2.putText(combined, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display
                cv2.imshow(window_name, combined)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset
                    self.h_min_1 = 0
                    self.h_max_1 = 10
                    self.h_min_2 = 170
                    self.h_max_2 = 180
                    self.s_min = 80
                    self.v_min = 100
                    print("[RESET] Reverted to default values")
                elif key == ord('s'):
                    filename = f"hsv_tuning_{self.h_min_1}_{self.h_max_1}_"
                    filename += f"{self.h_min_2}_{self.h_max_2}_{self.s_min}_{self.v_min}.png"
                    cv2.imwrite(filename, combined)
                    print(f"[SAVED] {filename}")
        
        except KeyboardInterrupt:
            print("\n[INTERRUPTED]")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.print_summary()
    
    def print_summary(self):
        """Print final values for config."""
        print("\n" + "="*60)
        print("FINAL HSV THRESHOLDS")
        print("="*60)
        print("\nAdd to config.yaml:")
        print(f"""detectors:
  laser:
    hsv_h_ranges: [[{self.h_min_1}, {self.h_max_1}], [{self.h_min_2}, {self.h_max_2}]]
    hsv_s_min: {self.s_min}
    hsv_v_min: {self.v_min}""")
        print("\nPython code:")
        print(f"""LaserDetector(
    method="hsv",
    hsv_h_ranges=[({self.h_min_1}, {self.h_max_1}), ({self.h_min_2}, {self.h_max_2})],
    hsv_s_min={self.s_min},
    hsv_v_min={self.v_min}
)""")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Interactive HSV threshold tuner')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID')
    
    args = parser.parse_args()
    
    tuner = HSVTuner()
    tuner.run(camera_id=args.camera)


if __name__ == '__main__':
    main()
