#!/usr/bin/env python3
"""
run_closed_loop.py: Entry point for laser eye tracking system.

Usage:
    python3 run_closed_loop.py --config config.yaml
"""

import sys
import argparse
import numpy as np
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from main_controller import LaserEyeController
from src.servo_controller import ServoController


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Laser Eye Tracking System')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum frames to process (None = infinite)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print debug information')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"[STARTUP] Loading config from {args.config}")
    config = load_config(args.config)
    
    # Extract servo limits
    servo_limits = np.array(config['servo_limits'], dtype=np.float32)
    
    # Extract model paths
    pose_model = config['models']['pose']
    laser_model = config['models']['laser']
    
    # Optional: Load camera calibration
    camera_matrix = None
    if 'camera_calibration' in config:
        calib_file = config['camera_calibration']
        if Path(calib_file).exists():
            calib_data = np.load(calib_file)
            camera_matrix = calib_data['camera_matrix']
            print(f"[STARTUP] Loaded camera calibration from {calib_file}")
    
    # Initialize controller
    print("[STARTUP] Initializing LaserEyeController...")
    controller = LaserEyeController(
        servo_limits=servo_limits,
        pose_model_path=pose_model,
        laser_model_path=laser_model,
        camera_matrix=camera_matrix,
        loop_rate_hz=config.get('loop_rate_hz', 30.0),
        verbose=args.verbose
    )
    
    # Center eyes to safe position
    print("[STARTUP] Centering eyes...")
    controller.servo_controller.center_eyes()
    
    # Run main loop
    print(f"[STARTUP] Starting main loop (camera_id={args.camera})...")
    print("[RUNNING] Press Ctrl+C to stop")
    
    try:
        controller.run(
            camera_source=args.camera,
            max_frames=args.max_frames
        )
    except Exception as e:
        print(f"[ERROR] Exception during run: {e}")
        import traceback
        traceback.print_exc()
        controller.servo_controller.emergency_stop()
    
    print("[SHUTDOWN] Laser eye controller shut down successfully")


if __name__ == '__main__':
    main()
