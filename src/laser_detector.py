"""
LaserDetector: Detects red laser dots in image using YOLO11-detect model.
Optimized for small red objects (laser dots ~10-50 pixels).
"""

from typing import List, Dict, Optional
import numpy as np
import cv2
from ultralytics import YOLO


class LaserDetector:
    """
    Detects red laser dots in image using YOLO11-detect model.
    Optimized for small red objects (laser dots ~10-50 pixels).
    """
    
    def __init__(self, model_path: str = "path/to/laser_detector.pt", 
                 conf_threshold: float = 0.5,
                 use_hsv_filter: bool = True):
        """
        Initialize laser detector.
        
        Args:
            model_path: Path to trained laser detection model
            conf_threshold: Confidence threshold
            use_hsv_filter: Pre-filter with HSV red detection
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.use_hsv_filter = use_hsv_filter
        self.last_detections = []
        
    def detect(self, frame: np.ndarray, device: int = 0) -> List[Dict]:
        """
        Detect laser dots in frame.
        
        Args:
            frame: Input image (BGR)
            
        Returns:
            List of detections:
            [
                {
                    'x': float (pixel),
                    'y': float (pixel),
                    'width': float (pixels),
                    'height': float (pixels),
                    'confidence': float,
                    'class': int (always 0 for laser dot)
                },
                ...
            ]
        """
        detections = []
        
        # Optional HSV pre-filtering (speed optimization)
        if self.use_hsv_filter:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Red in HSV: H ~0-10 or 170-180, S >100, V >100
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
            # Apply mask
            frame_filtered = cv2.bitwise_and(frame, frame, mask=mask)
        else:
            frame_filtered = frame
        
        # Run YOLO detection
        results = self.model(frame_filtered, device=device, verbose=False)
        
        if len(results) == 0:
            self.last_detections = detections
            return detections
        
        # Extract detections
        for box, conf in zip(results[0].boxes.xyxy, results[0].boxes.conf):
            if conf > self.conf_threshold:
                x_min, y_min, x_max, y_max = box
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min
                
                detections.append({
                    'x': float(x_center),
                    'y': float(y_center),
                    'width': float(width),
                    'height': float(height),
                    'confidence': float(conf),
                    'class': 0  # laser dot
                })
        
        self.last_detections = detections
        return detections
