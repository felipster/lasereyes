"""
Camera utility: Handles RPi camera capture using Picamera2.
Provides a unified interface for frame capture and conversion to OpenCV format.
"""

from typing import Tuple, Optional
import numpy as np
import cv2
from pathlib import Path
import pdb

from picamera2 import Picamera2
PICAMERA2_AVAILABLE = True

from libcamera import controls
LIBCAMERA_AVAILABLE = True


class PiCamera2Wrapper:
    """
    Wrapper around Picamera2 for consistent frame capture and OpenCV conversion.
    Handles initialization, frame capture, and format conversion.
    """
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        """
        Initialize Picamera2 camera.
        
        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Target frames per second
        """
        if not PICAMERA2_AVAILABLE:
            raise RuntimeError("picamera2 not available. Install with: pip install picamera2")
        
        self.width = width
        self.height = height
        self.fps = fps
        
        # Initialize camera
        self.camera = Picamera2()
        
        # Configure camera
        config = self.camera.create_video_configuration(
            main={"format": "RGB888", "size": (width, height)},
            controls={"FrameRate": fps}
        )
        self.camera.configure(config)
        
        # Start camera
        self.camera.start()
        
        # Get camera properties for calibration
        self.properties = self.camera.camera_properties
        
    def read(self) -> Tuple[bool, np.ndarray]:
        """
        Capture frame and convert to OpenCV format.
        
        Returns:
            Tuple of (success, frame_bgr)
            frame_bgr: Frame in BGR format (OpenCV standard)
        """
        try:
            # Capture frame - format depends on camera configuration
            request = self.camera.capture_request()
            frame = request.make_array("main")
            request.release()
            
            # RGB888 from Picamera2 is converted to BGR in the .make_array() call, so we can return it directly
            return True, frame
            
        except Exception as e:
            print(f"[ERROR] Failed to capture frame: {e}")
            return False, None
    
    def release(self):
        """Release camera resources."""
        if self.camera:
            self.camera.stop()
            self.camera.close()
    
    def set_brightness(self, value: float):
        """
        Set brightness (0.0 = dark, 1.0 = bright).
        
        Args:
            value: Brightness value (0.0 - 1.0)
        """
        if LIBCAMERA_AVAILABLE:
            try:
                # Map 0.0-1.0 to -1.0 to 1.0 range
                brightness = (value * 2.0) - 1.0
                self.camera.set_controls({controls.Brightness: brightness})
            except Exception as e:
                print(f"[WARNING] Could not set brightness: {e}")
    
    def set_contrast(self, value: float):
        """
        Set contrast (0.5 = normal, 2.0 = high).
        
        Args:
            value: Contrast value (0.5 - 2.0)
        """
        if LIBCAMERA_AVAILABLE:
            try:
                self.camera.set_controls({controls.Contrast: value})
            except Exception as e:
                print(f"[WARNING] Could not set contrast: {e}")
    
    def get_camera_matrix(self) -> np.ndarray:
        """
        Get camera intrinsic matrix from properties.
        
        Returns:
            3x3 camera matrix (K matrix)
        """
        try:
            # Default focal length for RPi Camera v3
            # These should be calibrated for your specific setup
            focal_length = 800.0  # pixels
            cx = self.width / 2.0
            cy = self.height / 2.0
            
            camera_matrix = np.array([
                [focal_length, 0.0, cx],
                [0.0, focal_length, cy],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
            
            return camera_matrix
        except Exception as e:
            print(f"[WARNING] Could not get camera matrix: {e}")
            return None


class CameraCapture:
    """
    Unified camera interface that handles both Picamera2 and fallback methods.
    """
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        """
        Initialize camera capture.
        
        Args:
            camera_id: Not used for Picamera2 (RPi camera only)
            width: Frame width
            height: Frame height
            fps: Target FPS
        """
        self.camera = None
        self.is_picamera2 = False
        
        # Try Picamera2 first (RPi camera)
        if PICAMERA2_AVAILABLE:
            try:
                self.camera = PiCamera2Wrapper(width, height, fps)
                self.is_picamera2 = True
                print("[CAMERA] Using Picamera2 (RPi native)")
                return
            except Exception as e:
                print(f"[WARNING] Picamera2 failed: {e}")
        
        raise RuntimeError("No camera capture method available")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame.
        
        Returns:
            Tuple of (success, frame_bgr)
        """
        if self.is_picamera2:
            return self.camera.read()
        else:
            ret, frame = self.camera.read()
            if ret and frame is not None:
                # Ensure frame is in BGR format
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    return ret, frame
            return ret, None
    
    def release(self):
        """Release camera resources."""
        if self.camera:
            if self.is_picamera2:
                self.camera.release()
            else:
                self.camera.release()
    
    def get_camera_matrix(self) -> np.ndarray:
        """Get camera matrix if available."""
        if self.is_picamera2 and hasattr(self.camera, 'get_camera_matrix'):
            return self.camera.get_camera_matrix()
        
        # Default fallback matrix
        return np.array([
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
