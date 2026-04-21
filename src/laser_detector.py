"""
LaserDetector: Detects red laser dots using HSV thresholding + optional PCA9685 pulsing.
Provides multiple detection methods: HSV, Adaptive, Temporal (with pulsing), and Hybrid.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
import time
import pdb

from adafruit_servokit import ServoKit
import board
import busio
import adafruit_pca9685
PCA9685_AVAILABLE = True


class LaserController:
    """Control laser on/off via PCA9685 PWM."""
    
    def __init__(self, pca_channel: int = 7):
        """
        Initialize laser controller on given PCA9685 channel.
        
        Args:
            pca_channel: Channel number (0-15)
        """
        if not PCA9685_AVAILABLE:
            raise RuntimeError("adafruit_pca9685 not available")

        self.i2c =  busio.I2C(board.SCL, board.SDA)
        self.pca = adafruit_pca9685.PCA9685(self.i2c)
        self.pca.frequency = 60  # Set frequency to 60Hz for LED control
        self.led_channel = self.pca.channels[pca_channel]
        self.is_on = False
    
    def on(self):
        """Turn laser ON (5V)."""
        self.led_channel.duty_cycle = 0xffff
        self.is_on = True
    
    def off(self):
        """Turn laser OFF (0V)."""
        self.led_channel.duty_cycle = 0x0000
        self.is_on = False
    
    def set_brightness(self, brightness: float):
        """
        Set laser brightness (0.0 = off, 0xffff = full brightness).
        
        Args:
            brightness: 0.0 to 0xffff
        """

        self.led_channel = brightness
        self.is_on = brightness > 0.1


class LaserDetector:
    """
    Classical signal processing laser detection using HSV thresholding.
    Can optionally use PCA9685 PWM pulsing for temporal detection.
    """
    
    def __init__(self, 
                 method: str = "hsv",
                 conf_threshold: float = 0.5,
                 use_pulsing: bool = False,
                 pca_channel1: int = 6,
                 pca_channel2: int = 7,
                 hsv_h_ranges: Optional[List[Tuple[int, int]]] = None,
                 hsv_s_min: int = 100,
                 hsv_v_min: int = 100):
        """
        Initialize laser detector with HSV thresholding.
        
        Args:
            method: Detection method - "hsv", "adaptive", "temporal", "hybrid"
            conf_threshold: Confidence threshold for detections
            use_pulsing: Enable PCA9685 laser pulsing for temporal detection
            pca_channel: PCA9685 channel for laser control (default: 7)
            hsv_h_ranges: List of (h_min, h_max) tuples for red hue ranges
            hsv_s_min: Saturation minimum threshold (0-255)
            hsv_v_min: Value minimum threshold (0-255)
        """
        self.method = method
        self.conf_threshold = conf_threshold
        pdb.set_trace()
        self.use_pulsing = use_pulsing
        self.pca_channel1 = pca_channel1
        self.pca_channel2 = pca_channel2
        # HSV thresholds for red laser (650nm)
        # Red wraps around in HSV: [0-10] and [170-180]
        self.hsv_h_ranges = hsv_h_ranges or [(0, 10), (170, 180)]
        self.hsv_s_min = hsv_s_min
        self.hsv_v_min = hsv_v_min
        
        # Initialize laser controller if pulsing enabled
        self.laser_controller1 = None
        self.laser_controller2 = None
        if self.use_pulsing and PCA9685_AVAILABLE:
            try:
                self.laser_controller1 = LaserController(pca_channel1)
                self.laser_controller2 = LaserController(pca_channel2)
            except Exception as e:
                print(f"[WARNING] Could not initialize laser controller: {e}")
                self.use_pulsing = False
        
        self.last_detections = []
        self.last_frame_on = None
        self.frame_count = 0
        self.debug_info = {}
        
    def detect(self, frame: np.ndarray) -> Tuple[List[Dict], Dict]:
        """
        Detect laser dots in frame using specified method.
        
        Args:
            frame: Input image (BGR)
            
        Returns:
            Tuple of (detections, debug_info)
            detections: List of detection dicts with keys:
                {
                    'x': float (pixel),
                    'y': float (pixel),
                    'width': float (pixels),
                    'height': float (pixels),
                    'confidence': float (0-1),
                    'class': int (always 0 for laser dot),
                    'method': str (detection method used)
                }
            debug_info: Dictionary with debug data for visualization
        """
        self.frame_count += 1
        debug_info = {'method': self.method, 'frame_count': self.frame_count}
        
        # Select detection method
        if self.method == "hsv":
            detections, debug_info = self._detect_hsv(frame, debug_info)
        elif self.method == "adaptive":
            detections, debug_info = self._detect_adaptive(frame, debug_info)
        elif self.method == "temporal":
            if self.use_pulsing and self.laser_controller1 and self.laser_controller2:
                detections, debug_info = self._detect_temporal(frame, debug_info)
            else:
                print("[WARNING] Temporal requires pulsing - falling back to HSV")
                detections, debug_info = self._detect_hsv(frame, debug_info)
        elif self.method == "hybrid":
            detections, debug_info = self._detect_hybrid(frame, debug_info)
        else:
            detections, debug_info = self._detect_hsv(frame, debug_info)
        
        # Filter by confidence threshold
        detections = [d for d in detections if d['confidence'] >= self.conf_threshold]
        
        self.last_detections = detections
        self.debug_info = debug_info
        
        return detections, debug_info
    
    def _detect_hsv(self, frame: np.ndarray, debug_info: Dict) -> Tuple[List[Dict], Dict]:
        """HSV red thresholding + morphology."""
        start_time = time.time()
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for red (two ranges due to HSV wrapping)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for h_min, h_max in self.hsv_h_ranges:
            lower = np.array([h_min, self.hsv_s_min, self.hsv_v_min])
            upper = np.array([h_max, 255, 255])
            mask |= cv2.inRange(hsv, lower, upper)
        
        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        detections = self._contours_to_detections(mask, frame)
        
        debug_info['hsv_mask'] = mask
        debug_info['processing_time_ms'] = (time.time() - start_time) * 1000
        debug_info['num_detections'] = len(detections)
        
        return detections, debug_info
    
    def _detect_adaptive(self, frame: np.ndarray, debug_info: Dict) -> Tuple[List[Dict], Dict]:
        """Adaptive thresholding for variable lighting."""
        start_time = time.time()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Adaptive threshold
        mask = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )
        
        # Invert (we want bright dots)
        mask = cv2.bitwise_not(mask)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        detections = self._contours_to_detections(mask, frame)
        
        debug_info['adaptive_mask'] = mask
        debug_info['processing_time_ms'] = (time.time() - start_time) * 1000
        debug_info['num_detections'] = len(detections)
        
        return detections, debug_info
    
    def _detect_temporal(self, frame: np.ndarray, debug_info: Dict) -> Tuple[List[Dict], Dict]:
        """Temporal frame differencing with laser pulsing."""
        start_time = time.time()
        
        # Pulse laser ON, capture frame
        self.laser_controller1.on()
        self.laser_controller2.on()
        time.sleep(0.01)  # Wait for LED to fully turn on
        frame_on = frame.copy()
        
        # Pulse laser OFF, capture frame
        self.laser_controller1.off()
        self.laser_controller2.off()
        time.sleep(0.01)  # Wait for LED to fully turn off
        frame_off = frame.copy()
        
        # Compute difference
        if self.last_frame_on is not None:
            diff = cv2.absdiff(frame_on, frame_off)
            
            # Convert to grayscale and threshold
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_diff, 50, 255, cv2.THRESH_BINARY)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            detections = self._contours_to_detections(mask, frame)
        else:
            detections = []
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        self.last_frame_on = frame_on
        
        debug_info['temporal_mask'] = mask
        debug_info['processing_time_ms'] = (time.time() - start_time) * 1000
        debug_info['num_detections'] = len(detections)
        
        return detections, debug_info
    
    def _detect_hybrid(self, frame: np.ndarray, debug_info: Dict) -> Tuple[List[Dict], Dict]:
        """Hybrid: Try HSV first (fast), fall back to adaptive if needed."""
        # First, try HSV (fast path)
        detections_hsv, debug_hsv = self._detect_hsv(frame, debug_info)
        
        # If we got 2+ detections with good confidence, return HSV results
        confident_dets = [d for d in detections_hsv if d['confidence'] >= self.conf_threshold]
        if len(confident_dets) >= 2:
            debug_info['hybrid_used_method'] = 'hsv'
            return detections_hsv, debug_info
        
        # Otherwise, try adaptive for more robustness
        detections_adaptive, debug_adaptive = self._detect_adaptive(frame, debug_info)
        debug_info['hybrid_used_method'] = 'adaptive'
        return detections_adaptive, debug_info
    
    def _contours_to_detections(self, mask: np.ndarray, frame: np.ndarray) -> List[Dict]:
        """Convert contours to detection format."""
        detections = []
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter by area (laser dots are typically 10-50 pixels)
            area = cv2.contourArea(contour)
            if area < 5 or area > 500:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Get centroid
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
            else:
                cx, cy = x + w / 2, y + h / 2
            
            # Compute confidence based on circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                confidence = min(1.0, circularity)  # More circular = more confident
            else:
                confidence = 0.5
            
            # Only keep high-circularity detections
            if confidence >= 0.3:
                detections.append({
                    'x': float(cx),
                    'y': float(cy),
                    'width': float(w),
                    'height': float(h),
                    'confidence': float(confidence),
                    'class': 0,
                    'area': float(area)
                })
        
        # Sort by confidence (descending)
        detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)
        
        return detections
