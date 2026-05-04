"""
PoseDetector: YOLO11-pose model wrapper.
Detects human face keypoints and computes 3D gaze direction.
"""

from typing import Optional, Dict, Tuple
import numpy as np
from ultralytics import YOLO


class PoseDetector:
    """
    YOLO11-pose model wrapper.
    Detects human face keypoints (5 points: left_eye, right_eye, nose, chin, forehead).
    Computes 3D gaze direction using camera calibration.
    """
    
    # COCO 17-keypoint indices (yolov8-pose / yolov8s-pose from Hailo Model Zoo)
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    
    def __init__(self, model_path: str = "yolo/yolo11n-pose.pt",
                 conf_threshold: float = 0.5,
                 camera_matrix: Optional[np.ndarray] = None,
                 device: str = 'cpu'):
        """
        Initialize pose model.

        If model_path ends in '.hef', uses HailoPoseInferencer (hailo_platform)
        instead of Ultralytics YOLO, bypassing CUDA device selection entirely.
        """
        self.conf_threshold = conf_threshold
        self.device = device
        self._inferencer = None

        if model_path.endswith('.hef'):
            from .hailo_pose_inferencer import HailoPoseInferencer
            self._inferencer = HailoPoseInferencer(model_path, conf_threshold=conf_threshold)
            self.model = None
            print(f"[POSE] Hailo NPU inference: {model_path}")
        else:
            self.model = YOLO(model_path)
            print(f"[POSE] Ultralytics inference ({device}): {model_path}")
        
        # Default RPi Camera v3 (OV5647) intrinsics at 640x480
        # Should be calibrated using calibrate_camera() for your specific setup
        if camera_matrix is None:
            self.K = np.array([
                [800.0,   0.0, 320.0],   # fx, 0, cx
                [  0.0, 800.0, 240.0],   # 0, fy, cy
                [  0.0,   0.0,   1.0]    # 0, 0, 1
            ], dtype=np.float32)
        else:
            self.K = camera_matrix
        
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        
        self.last_detection = None
        
    def detect(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Run pose detection on frame.

        Args:
            frame: Input image (BGR, from RPi camera)

        Returns:
            {
                'detected': bool,
                'keypoints': {
                    'nose': (x, y, conf),
                    'left_eye': (x, y, conf),
                    'right_eye': (x, y, conf),
                    'left_ear': (x, y, conf),
                    'right_ear': (x, y, conf)
                },
                'face_box': (x_min, y_min, x_max, y_max),
                'confidence': float
            }
        """
        if self._inferencer is not None:
            detection = self._inferencer.infer(frame)
            self.last_detection = detection
            return detection

        results = self.model(frame, device=self.device, verbose=False)

        detection = {
            'detected': False,
            'keypoints': {},
            'face_box': None,
            'confidence': 0.0
        }

        if len(results) == 0 or results[0].keypoints is None:
            self.last_detection = detection
            return detection

        keypoints = results[0].keypoints.xy[0]  # First person detected
        kpt_conf = results[0].keypoints.conf[0]  # Confidences

        # COCO 17-keypoint indices for the face region
        face_kpt_map = {
            'nose': self.NOSE,
            'left_eye': self.LEFT_EYE,
            'right_eye': self.RIGHT_EYE,
            'left_ear': self.LEFT_EAR,
            'right_ear': self.RIGHT_EAR,
        }
        for name, idx in face_kpt_map.items():
            if idx < len(kpt_conf) and kpt_conf[idx] > self.conf_threshold:
                x, y = keypoints[idx]
                detection['keypoints'][name] = (float(x), float(y), float(kpt_conf[idx]))
        
        # Bounding box (if available)
        if results[0].boxes:
            box = results[0].boxes.xyxy[0]
            detection['face_box'] = tuple(map(float, box))
            detection['confidence'] = float(results[0].boxes.conf[0])
        
        # Mark as detected if we have at least eyes and nose
        if 'left_eye' in detection['keypoints'] and \
           'right_eye' in detection['keypoints'] and \
           'nose' in detection['keypoints']:
            detection['detected'] = True
        
        self.last_detection = detection
        return detection
    
    def pixel_to_normalized_3d(self, u: float, v: float) -> np.ndarray:
        """
        Convert pixel coordinates to normalized 3D direction in camera frame.
        
        Uses camera intrinsics matrix K to properly project 2D pixels to 3D space.
        
        Args:
            u, v: Pixel coordinates (x, y in image space)
            
        Returns:
            Normalized 3D unit vector [vx, vy, vz] where Z=1 (normalized)
            This ray points from camera origin through the pixel into 3D space
        """
        # Remove principal point offset and scale by focal length
        x_norm = (u - self.cx) / self.fx
        y_norm = (v - self.cy) / self.fy
        z_norm = 1.0
        
        # Normalize to unit vector
        ray = np.array([x_norm, y_norm, z_norm])
        ray_unit = ray / np.linalg.norm(ray)
        
        return ray_unit
    
    def get_gaze_direction_3d(self, detection: Dict) -> Optional[np.ndarray]:
        """
        Compute 3D gaze direction from facial keypoints.
        
        Returns true 3D unit vector, not spherical angles.
        
        Args:
            detection: From self.detect()
            
        Returns:
            3D unit direction vector [vx, vy, vz] in camera frame
            OR None if detection failed
            
        Interpretation:
            - vz (forward): positive = looking forward at camera
            - vx (rightward): positive = looking right
            - vy (downward): positive = looking down
        """
        if not detection['detected']:
            return None
        
        kpts = detection['keypoints']
        if 'left_eye' not in kpts or 'right_eye' not in kpts or 'nose' not in kpts:
            return None
        
        # Extract 2D pixel coordinates
        u_left, v_left = kpts['left_eye'][:2]
        u_right, v_right = kpts['right_eye'][:2]
        
        # Convert to 3D camera rays (normalized using K matrix)
        ray_left = self.pixel_to_normalized_3d(u_left, v_left)
        ray_right = self.pixel_to_normalized_3d(u_right, v_right)
        
        # Compute gaze as midpoint between eye rays
        gaze_ray = (ray_left + ray_right) / 2
        gaze_ray = gaze_ray / np.linalg.norm(gaze_ray)  # Re-normalize
        
        return gaze_ray
    
    def get_gaze_angles_from_3d(self, gaze_3d: np.ndarray) -> Tuple[float, float]:
        """
        Convert 3D gaze vector to spherical angles.
        
        Args:
            gaze_3d: 3D unit direction vector [vx, vy, vz]
            
        Returns:
            (azimuth, elevation) in degrees where:
            - azimuth: 0° = forward, +90° = right, -90° = left
            - elevation: 0° = forward, +90° = up, -90° = down
        """
        vx, vy, vz = gaze_3d
        
        # Azimuth: angle in XZ plane from Z axis
        azimuth = np.arctan2(vx, vz) * 180 / np.pi
        
        # Elevation: angle from XZ plane
        elevation = np.arcsin(np.clip(vy, -1, 1)) * 180 / np.pi
        
        return (azimuth, elevation)
    
    def get_head_pose_euler(self, detection: Dict) -> Optional[Tuple[float, float, float]]:
        """
        Compute head pose (roll, pitch, yaw) from facial keypoints.
        
        Uses 5 keypoints to estimate full 3D head orientation.
        
        Args:
            detection: From self.detect()
            
        Returns:
            (roll, pitch, yaw) in degrees where:
            - roll: rotation around forward axis (head tilt left/right)
            - pitch: rotation around sideways axis (head nod up/down)
            - yaw: rotation around vertical axis (head turn left/right)
            
            OR None if detection failed
        """
        if not detection['detected']:
            return None
        
        kpts = detection['keypoints']
        # Requires left_ear + right_ear for roll/yaw estimation in COCO keypoint format.
        # chin/forehead are not available; returns None if ears are missing.
        required = ['left_eye', 'right_eye', 'nose', 'left_ear', 'right_ear']
        if not all(k in kpts for k in required):
            return None
        
        # Extract pixel coordinates (COCO keypoints: eyes, nose, ears)
        u_left, v_left = kpts['left_eye'][:2]
        u_right, v_right = kpts['right_eye'][:2]
        u_nose, v_nose = kpts['nose'][:2]
        u_left_ear, v_left_ear = kpts['left_ear'][:2]
        u_right_ear, v_right_ear = kpts['right_ear'][:2]

        # Convert to normalized 3D rays using camera matrix
        ray_left = self.pixel_to_normalized_3d(u_left, v_left)
        ray_right = self.pixel_to_normalized_3d(u_right, v_right)
        ray_nose = self.pixel_to_normalized_3d(u_nose, v_nose)
        ray_left_ear = self.pixel_to_normalized_3d(u_left_ear, v_left_ear)
        ray_right_ear = self.pixel_to_normalized_3d(u_right_ear, v_right_ear)

        # Eye midpoint → nose gives a rough "down the face" direction (inverted = up)
        eye_mid = (ray_left + ray_right) / 2
        face_up = eye_mid - ray_nose
        face_up = face_up / (np.linalg.norm(face_up) + 1e-6)

        # Ear-to-ear vector approximates the face-right axis
        face_right = ray_right_ear - ray_left_ear
        face_right = face_right / (np.linalg.norm(face_right) + 1e-6)
        
        # Face forward (cross product)
        face_forward = np.cross(face_right, face_up)
        face_forward = face_forward / (np.linalg.norm(face_forward) + 1e-6)
        
        # Compute angles from standard frame [0, 0, 1] (looking forward at camera)
        
        # Yaw (left-right turn): rotation around Y axis
        yaw = np.arctan2(face_forward[0], face_forward[2]) * 180 / np.pi
        
        # Pitch (up-down nod): rotation around X axis
        pitch = np.arcsin(np.clip(-face_forward[1], -1, 1)) * 180 / np.pi
        
        # Roll (head tilt): rotation around Z axis
        roll = np.arctan2(face_up[0], face_up[1]) * 180 / np.pi
        
        return (roll, pitch, yaw)

    def close(self):
        if self._inferencer is not None:
            self._inferencer.close()
