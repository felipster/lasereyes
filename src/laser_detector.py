"""
LaserDetector: Detects red laser dots using HSV thresholding + optional PCA9685 pulsing.
Provides multiple detection methods: HSV, Adaptive, Temporal (with pulsing), and Hybrid.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
import time
import pdb

from adafruit_extended_bus import ExtendedI2C as I2C
from adafruit_pca9685 import PCA9685
PCA9685_AVAILABLE = True


class LaserController:
    """Control laser on/off via a PCA9685 PWM channel."""

    def __init__(self, pca: 'PCA9685', pca_channel: int = 7):
        """
        Args:
            pca:         Shared PCA9685 instance (owned by ServoController or
                         created by LaserDetector when used standalone).
            pca_channel: Channel number (0-15) for this laser's LED.
        """
        self.led_channel = pca.channels[pca_channel]
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
                 hsv_s_min: int = 150,
                 hsv_v_min: int = 150,
                 pca: Optional['PCA9685'] = None):
        """
        Args:
            method:         Detection method — "hsv" | "bgr" | "brightness" | "temporal" | "hybrid"
            conf_threshold: Minimum confidence to report a detection
            pca_channel1:   PCA9685 channel for laser 1
            pca_channel2:   PCA9685 channel for laser 2
            camera_capture: Required for temporal method; also used for lock_exposure()
            hsv_h_ranges:   List of (h_min, h_max) hue bands for red detection
            hsv_s_min:      HSV saturation minimum (0-255)
            hsv_v_min:      HSV value minimum (0-255)
            pca:            Shared PCA9685 instance. When None (standalone use, e.g.
                            visualiser) a new one is created on I2C-2 at 50 Hz.
        """
        self.method = method
        self.conf_threshold = conf_threshold
        self.pca_channel1 = pca_channel1
        self.pca_channel2 = pca_channel2
        self.camera_capture = camera_capture
        # HSV thresholds for a 650 nm red laser with fixed ColourGains=(1.8, 0.8).
        # Tight hue bands around the red wrap-point; high S and V reject background
        # objects that are merely reddish or dimly lit.
        self.hsv_h_ranges = hsv_h_ranges or [(0, 8), (172, 180)]
        self.hsv_s_min = hsv_s_min
        self.hsv_v_min = hsv_v_min
        
        # Use the shared PCA9685 if provided; otherwise create a standalone one.
        # A shared instance avoids the frequency conflict that occurs when
        # ServoController (50 Hz) and LaserController (formerly 1000 Hz) each
        # instantiate their own PCA9685 at address 0x40.  LEDs only need
        # duty_cycle 0x0000/0xFFFF so 50 Hz is perfectly adequate.
        self.laser_controller1 = None
        self.laser_controller2 = None
        if PCA9685_AVAILABLE:
            try:
                if pca is None:
                    _i2c = I2C(2)
                    pca = PCA9685(_i2c, address=0x40)
                    pca.frequency = 50
                self.laser_controller1 = LaserController(pca=pca, pca_channel=self.pca_channel1)
                self.laser_controller2 = LaserController(pca=pca, pca_channel=self.pca_channel2)
            except Exception as e:
                print(f"[WARNING] Could not initialize laser controller: {e}")
        
        self.last_detections = []
        self.last_frame_on = None
        self.frame_count = 0
        self.debug_info = {}
        self._exposure_locked = False
    
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
        elif self.method == "brightness":
            detections, debug_info = self._detect_brightness_excess(frame, debug_info)
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
        """HSV red thresholding + morphology with fixed camera exposure."""
        start_time = time.time()

        if not self._exposure_locked and self.camera_capture is not None:
            self.camera_capture.lock_exposure()
            self._exposure_locked = True
            time.sleep(0.1)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for h_min, h_max in self.hsv_h_ranges:
            lower = np.array([h_min, self.hsv_s_min, self.hsv_v_min])
            upper = np.array([h_max, 255, 255])
            mask |= cv2.inRange(hsv, lower, upper)

        debug_info['mask_pixel_count'] = np.count_nonzero(mask)

        # (3,3) open removes 1-px noise without erasing 2-3 px laser dots.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

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
        """BGR red thresholding with fixed camera exposure."""
        start_time = time.time()

        if not self._exposure_locked and self.camera_capture is not None:
            self.camera_capture.lock_exposure()
            self._exposure_locked = True
            time.sleep(0.1)

        b, g, r = cv2.split(frame)

        r_thresh = 150   # minimum absolute red value
        diff_thresh = 30  # red must exceed both B and G by this margin

        mask = ((r > r_thresh) &
                (r > b + diff_thresh) &
                (r > g + diff_thresh)).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        detections = self._contours_to_detections(mask, frame, None, debug_info)

        debug_info['bgr_mask'] = mask
        debug_info['processing_time_ms'] = (time.time() - start_time) * 1000
        debug_info['num_detections'] = len(detections)

        return detections, debug_info
    
    def _detect_temporal(self, frame: np.ndarray, debug_info: Dict) -> Tuple[List[Dict], Dict]:
        """Temporal frame differencing with laser pulsing (Strategy A+C).

        Strategy A: lock camera exposure/gain before capture pairs so AGC doesn't
        corrupt the diff with full-frame brightness shifts.
        Strategy C: AND the temporal diff mask with a local red-channel excess mask
        to handle saturated laser dots that lose their colour information.
        """
        start_time = time.time()

        if self.camera_capture is None:
            raise RuntimeError("CameraCapture required for temporal detection method")

        # Strategy A: lock exposure on first call so AGC cannot shift between frames
        if not self._exposure_locked:
            self.camera_capture.lock_exposure()
            self._exposure_locked = True
            time.sleep(0.1)  # let camera settle to new fixed settings

        # Capture laser-ON frame (lasers already on from main loop)
        ret_on, frame_on = self.camera_capture.read()
        if not ret_on:
            frame_on = frame.copy()

        # Turn lasers off and wait long enough for full extinguish + one camera frame
        if self.laser_controller1:
            self.laser_controller1.off()
        if self.laser_controller2:
            self.laser_controller2.off()
        time.sleep(0.020)  # 20 ms: ~1 PCA9685 cycle at 50 Hz + camera settle

        ret_off, frame_off = self.camera_capture.read()
        if not ret_off:
            frame_off = frame.copy()

        # Restore lasers for next iteration
        if self.laser_controller1:
            self.laser_controller1.on()
        if self.laser_controller2:
            self.laser_controller2.on()

        if frame_on is not None and frame_off is not None:
            diff = cv2.absdiff(frame_on, frame_off)

            # Use the red channel of the diff — red laser changes red channel most
            diff_red = diff[:, :, 2]
            _, mask_temporal = cv2.threshold(diff_red, 10, 255, cv2.THRESH_BINARY)

            # Strategy C: local red-channel brightness excess on the laser-on frame.
            # Computed at half resolution (4× faster); (15,15) at 320×240 is
            # equivalent to (30,30) at 640×480 for background estimation.
            fh, fw = frame_on.shape[:2]
            r_small = cv2.resize(frame_on[:, :, 2], (fw // 2, fh // 2)).astype(np.float32)
            r_bg_small = cv2.GaussianBlur(r_small, (15, 15), 0)
            r_excess_small = np.clip(r_small - r_bg_small, 0, 255).astype(np.uint8)
            r_excess = cv2.resize(r_excess_small, (fw, fh))
            _, mask_excess = cv2.threshold(r_excess, 20, 255, cv2.THRESH_BINARY)

            # Require both signals: temporal change AND local brightness peak
            mask = cv2.bitwise_and(mask_temporal, mask_excess)

            # (3,3) open only: removes isolated 1-px ADC noise but preserves
            # 2-3 px laser dots. A (5,5) kernel would erase them entirely.
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            hsv = cv2.cvtColor(frame_on, cv2.COLOR_BGR2HSV)
            detections = self._contours_to_detections(mask, frame_on, hsv, debug_info)

            debug_info['temporal_diff'] = diff
            debug_info['brightness_excess'] = cv2.cvtColor(r_excess, cv2.COLOR_GRAY2BGR)
        else:
            detections = []
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            debug_info['temporal_diff'] = np.zeros_like(frame)
            debug_info['brightness_excess'] = np.zeros_like(frame)

        self.last_frame_on = frame_on

        debug_info['temporal_mask'] = mask
        debug_info['processing_time_ms'] = (time.time() - start_time) * 1000
        debug_info['num_detections'] = len(detections)

        return detections, debug_info
    
    def _detect_brightness_excess(self, frame: np.ndarray, debug_info: Dict) -> Tuple[List[Dict], Dict]:
        """Strategy C: local red-channel brightness peak detection.

        Finds pixels where the red channel is significantly brighter than their
        local neighbourhood. This is background-independent and works even when
        the laser dot is fully saturated (appears white/pink in BGR).
        """
        start_time = time.time()

        # Compute background at half resolution: 4× faster blur with equivalent
        # spatial scale (15×15 at 320×240 ≈ 30×30 at 640×480).
        h, w = frame.shape[:2]
        r_full = frame[:, :, 2]
        r_small = cv2.resize(r_full, (w // 2, h // 2)).astype(np.float32)
        r_bg_small = cv2.GaussianBlur(r_small, (15, 15), 0)
        r_excess_small = np.clip(r_small - r_bg_small, 0, 255).astype(np.uint8)
        r_excess = cv2.resize(r_excess_small, (w, h))

        _, mask = cv2.threshold(r_excess, 25, 255, cv2.THRESH_BINARY)

        # (3,3) open removes 1-px noise without erasing 2-3 px laser dots.
        # No close: it expands blobs, creating large contours that slow findContours.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detections = self._contours_to_detections(mask, frame, hsv, debug_info)

        debug_info['brightness_mask'] = mask
        debug_info['brightness_excess'] = cv2.cvtColor(r_excess, cv2.COLOR_GRAY2BGR)
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
        """Convert contours to detection format using multi-factor confidence scoring."""
        detections = []
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if debug_info is not None:
            debug_info['contours'] = contours
        
        for contour in contours:
            # Filter by effective radius from bounding box, NOT by cv2.contourArea.
            # Reason: for a ring (annulus) contour, RETR_EXTERNAL returns only the outer
            # boundary. cv2.contourArea on that boundary uses the shoelace formula and
            # returns π*R² (the full disc area including the hole), which easily exceeds
            # a pixel-count-based limit and causes all ring detections to be rejected.
            # Bounding-box radius is immune to this and works for both filled circles
            # and ring outer contours.
            x, y, w, h = cv2.boundingRect(contour)
            r_eff = (w + h) / 4.0  # mean of half-width and half-height
            if r_eff < 1.0 or r_eff > 35.0:  # ~2–70 px outer diameter
                continue
            area = cv2.contourArea(contour)
            
            # Get centroid
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
            else:
                cx, cy = x + w / 2, y + h / 2
            
            cx_int, cy_int = int(cx), int(cy)
            
            # Compute three confidence components
            circularity_score = self._compute_circularity(contour, area)
            red_saturation_score = self._compute_red_saturation_score(contour, frame, hsv, mask)
            
            # Weighted confidence: red saturation (50%), circularity (50%)
            confidence = (
                0.50 * red_saturation_score +
                0.50 * circularity_score
            )
            
            # Only keep detections with meaningful confidence
            if confidence >= 0.45:
                # Log HSV and BGR values at/around the contour for debugging
                if 0 <= cx_int < frame.shape[1] and 0 <= cy_int < frame.shape[0]:
                    b, g, r = frame[cy_int, cx_int]
                    
                    # Debug: show all scoring components and pixel values
                    if self.frame_count % 30 == 0 and len(detections) < 5:
                        det_hsv = hsv[cy_int, cx_int] if hsv is not None else [0, 0, 0]
                        print(f"[DEBUG] Detection at ({cx_int},{cy_int}): HSV={det_hsv}, BGR=({b},{g},{r})")
                        print(f"  Scores: red_sat={red_saturation_score:.2f}, circ={circularity_score:.2f}, final={confidence:.2f}")
                
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
    
    def _compute_circularity(self, contour: np.ndarray, area: float) -> float:
        """Circularity score via ISO quotient and bounding-box aspect ratio.

        For a ring (annulus) contour: RETR_EXTERNAL returns the outer boundary,
        a circle of radius R. cv2.contourArea (shoelace) = π*R², perimeter = 2π*R,
        so ISO = 4π*(π*R²)/(2π*R)² = 1.0 — exactly what we want.

        Solidity (area/hull_area) was previously used but is a no-op here: for any
        approximately circular contour (filled disc OR ring outer boundary) both
        areas equal π*R², so solidity is always ~1.0 and contributes nothing.

        Range: [0, 1]
        """
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            # 4π*A/P² = 1.0 for a perfect circle, <1 for any deviation
            iso = min(1.0, (4 * np.pi * area) / (perimeter ** 2))
        else:
            iso = 0.0

        x, y, w, h = cv2.boundingRect(contour)
        aspect = min(float(w) / h, float(h) / w) if h > 0 else 1.0  # 1.0 = square/circular

        # ISO is the primary discriminator; aspect penalises elongated noise blobs
        return 0.75 * iso + 0.25 * aspect
    
    def _compute_red_saturation_score(self, contour: np.ndarray, frame: np.ndarray, 
                                     hsv: np.ndarray = None, mask: np.ndarray = None) -> float:
        """
        Compute red saturation score: percentage of pixels within contour that are "red".
        
        Most direct measure of laser presence. Handles:
        - Donut shapes (red ring detected)
        - Ellipses (distributed red pixels)
        - Noise (only counts truly red pixels)
        
        Range: [0, 1]
        """
        if mask is None:
            return 0.5
        
        # Create a mask for just this contour
        contour_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], 0, 255, -1)
        
        # Count pixels in contour
        total_pixels = np.sum(contour_mask > 0)
        if total_pixels == 0:
            return 0.5
        
        # Count red pixels within contour.
        # Two complementary checks are OR'd together:
        #   HSV check: works for non-saturated red (high S)
        #   Normalised-RGB check: works for saturated/near-white laser dots where
        #     S drops to ~0, by checking that R dominates the sum R+G+B.
        b_f = frame[:, :, 0].astype(np.float32)
        g_f = frame[:, :, 1].astype(np.float32)
        r_f = frame[:, :, 2].astype(np.float32)
        r_norm = r_f / (r_f + g_f + b_f + 1.0)
        norm_red_mask = (r_norm > 0.38) & (r_f > 80)

        if hsv is not None:
            h, s, v = cv2.split(hsv)
            hsv_red_mask = ((h <= 10) | (h >= 170)) & (s > 80) & (v > 80)
            combined_red = hsv_red_mask | norm_red_mask
        else:
            combined_red = norm_red_mask

        red_pixels = np.sum(combined_red & (contour_mask > 0))
        
        # Red saturation: percentage of contour pixels that are red
        red_saturation = red_pixels / total_pixels
        return min(1.0, red_saturation)
