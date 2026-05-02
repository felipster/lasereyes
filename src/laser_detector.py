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
from adafruit_extended_bus import ExtendedI2C as I2C
from adafruit_pca9685 import PCA9685
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

        self.i2c = I2C(2)
        self.pca = PCA9685(self.i2c, address=0x40)
        self.pca.frequency = 1000  # Set frequency to 1000Hz for LED control
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
                 conf_threshold: float = 0.8,
                 pca_channel1: int = 6,
                 pca_channel2: int = 7,
                 camera_capture: Optional['CameraCapture'] = None,
                 hsv_h_ranges: Optional[List[Tuple[int, int]]] = None,
                 hsv_s_min: int = 50,
                 hsv_v_min: int = 50):
        """
        Initialize laser detector with HSV thresholding.
        
        Args:
            method: Detection method - "hsv", "adaptive", "temporal", "hybrid"
            conf_threshold: Confidence threshold for detections
            pca_channel1: PCA9685 channel for first laser control (default: 6)
            pca_channel2: PCA9685 channel for second laser control (default: 7)
            camera_capture: CameraCapture instance for temporal method (optional)
            hsv_h_ranges: List of (h_min, h_max) tuples for red hue ranges
            hsv_s_min: Saturation minimum threshold (0-255)
            hsv_v_min: Value minimum threshold (0-255)
        """
        self.method = method
        self.conf_threshold = conf_threshold
        self.pca_channel1 = pca_channel1
        self.pca_channel2 = pca_channel2
        self.camera_capture = camera_capture
        # HSV thresholds for red laser (650nm)
        # Red wraps around in HSV: [0-10] and [170-180]
        self.hsv_h_ranges = hsv_h_ranges or [(0, 10), (170, 180)]
        self.hsv_s_min = hsv_s_min
        self.hsv_v_min = hsv_v_min
        
        # Initialize laser controller if pulsing enabled
        self.laser_controller1 = None
        self.laser_controller2 = None
        if PCA9685_AVAILABLE:
            try:
                self.laser_controller1 = LaserController(pca_channel=self.pca_channel1)
                self.laser_controller2 = LaserController(pca_channel=self.pca_channel2)
            except Exception as e:
                print(f"[WARNING] Could not initialize laser controller: {e}")
        
        self.last_detections = []
        self.last_frame_on = None
        self.frame_count = 0
        self.debug_info = {}
    
    def set_camera_capture(self, camera_capture: 'CameraCapture'):
        """Set the camera capture instance for temporal detection."""
        self.camera_capture = camera_capture
        
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
        elif self.method == "bgr":
            detections, debug_info = self._detect_bgr(frame, debug_info)
        elif self.method == "temporal":
            detections, debug_info = self._detect_temporal(frame, debug_info) 
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
        
        # Debug: sample HSV and BGR values from center of frame to diagnose color issues
        if self.frame_count % 30 == 0:  # Print every 30 frames
            h, w = frame.shape[:2]
            center_bgr = frame[h//2, w//2]
            center_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[h//2, w//2]
            print(f"[DEBUG] Center pixel BGR={center_bgr} HSV={center_hsv} | Frame: {self.frame_count}")
        
        # Create mask for red (two ranges due to HSV wrapping)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for h_min, h_max in self.hsv_h_ranges:
            lower = np.array([h_min, self.hsv_s_min, self.hsv_v_min])
            upper = np.array([h_max, 255, 255])
            mask |= cv2.inRange(hsv, lower, upper)
        
        # Debug: show mask statistics
        mask_pixels = np.count_nonzero(mask)
        debug_info['mask_pixel_count'] = mask_pixels
        
        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        detections = self._contours_to_detections(mask, frame, hsv, debug_info)
        
        debug_info['hsv_mask'] = mask
        debug_info['processing_time_ms'] = (time.time() - start_time) * 1000
        debug_info['num_detections'] = len(detections)
        
        return detections, debug_info
    
    def _detect_adaptive(self, frame: np.ndarray, debug_info: Dict) -> Tuple[List[Dict], Dict]:
        """Adaptive thresholding for variable lighting."""
        start_time = time.time()
        
        # Convert to grayscale and HSV (for debug)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
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
        detections = self._contours_to_detections(mask, frame, hsv, debug_info)
        
        debug_info['adaptive_mask'] = mask
        debug_info['processing_time_ms'] = (time.time() - start_time) * 1000
        debug_info['num_detections'] = len(detections)
        
        return detections, debug_info
    
    def _detect_bgr(self, frame: np.ndarray, debug_info: Dict) -> Tuple[List[Dict], Dict]:
        """BGR red thresholding - detect red in BGR color space directly."""
        start_time = time.time()
        
        # Create mask for red in BGR space
        # Red channel should be high, blue and green should be lower
        # Also check that red is significantly higher than blue and green
        b, g, r = cv2.split(frame)
        
        # Conditions for red detection in BGR:
        # 1. Red channel is above threshold
        # 2. Red is significantly higher than blue (r > b + threshold)
        # 3. Red is significantly higher than green (r > g + threshold)
        r_thresh = 100  # Minimum red value
        diff_thresh = 30  # Red must be at least this much higher than B and G
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask = ((r > r_thresh) & 
                (r > b + diff_thresh) & 
                (r > g + diff_thresh)).astype(np.uint8) * 255
        
        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        detections = self._contours_to_detections(mask, frame, None, debug_info)
        
        debug_info['bgr_mask'] = mask
        debug_info['processing_time_ms'] = (time.time() - start_time) * 1000
        debug_info['num_detections'] = len(detections)
        
        return detections, debug_info
    
    def _detect_temporal(self, frame: np.ndarray, debug_info: Dict) -> Tuple[List[Dict], Dict]:
        """Temporal frame differencing with laser pulsing."""
        start_time = time.time()
        
        if self.camera_capture is None:
            raise RuntimeError("CameraCapture required for temporal detection method")
        
        # capture frame with lasers on, triggered by main loop
        ret_on, frame_on = self.camera_capture.read()
        if not ret_on:
            frame_on = frame.copy()  # Fallback to passed frame
        
        # Pulse laser OFF, capture frame
        self.laser_controller1.off()
        self.laser_controller2.off()
        time.sleep(0.001)  # Wait for LED to fully turn off
        ret_off, frame_off = self.camera_capture.read()
        if not ret_off:
            frame_off = frame.copy()  # Fallback
        
        # Compute difference
        if frame_on is not None and frame_off is not None:
            diff = cv2.absdiff(frame_on, frame_off)
            
            # Convert to grayscale and threshold
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_diff, 50, 255, cv2.THRESH_BINARY)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Get HSV for debug
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            detections = self._contours_to_detections(mask, frame, hsv, debug_info)
            debug_info['temporal_diff'] = diff
        else:
            detections = []
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            debug_info['temporal_diff'] = np.zeros_like(frame)
        
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
    
    def _contours_to_detections(self, mask: np.ndarray, frame: np.ndarray, hsv: np.ndarray = None, debug_info: Optional[Dict] = None) -> List[Dict]:
        """Convert contours to detection format."""
        detections = []
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if debug_info is not None:
            debug_info['contours'] = contours
        
        for contour in contours:
            # Filter by area (laser dots are typically 10-50 pixels)
            area = cv2.contourArea(contour)
            # Tightened: laser dots are small, reject large areas that are likely not lasers
            if area < 1 or area > 200:
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
            
            # Only keep high-circularity detections (laser dots should be fairly circular)
            if confidence >= 0.5:
                # Additional BGR verification: ensure it's actually red, not just HSV-matched
                cx_int, cy_int = int(cx), int(cy)
                if 0 <= cx_int < frame.shape[1] and 0 <= cy_int < frame.shape[0]:
                    b, g, r = frame[cy_int, cx_int]
                    # Red in BGR: R should be significantly higher than B and G
                    is_red_bgr = (r > b + 30) and (r > g + 30) and (r > 100)
                    
                    # Debug: show HSV and BGR at detection centroid
                    if self.frame_count % 30 == 0 and len(detections) < 5:
                        det_hsv = hsv[cy_int, cx_int] if hsv is not None else [0, 0, 0]
                        print(f"[DEBUG] Detection at ({cx_int},{cy_int}): HSV={det_hsv}, BGR=({b},{g},{r}), is_red_bgr={is_red_bgr}, area={area:.0f}, conf={confidence:.2f}")
                    
                    if not is_red_bgr:
                        continue  # Skip this detection - not actually red in BGR
                
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
