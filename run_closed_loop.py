#!/usr/bin/env python3
"""
run_closed_loop.py: Entry point for laser eye tracking system with live streaming.

Usage:
    python3 run_closed_loop.py --config config.yaml
    python3 run_closed_loop.py --config config.yaml --laser-method hsv --stream
    python3 run_closed_loop.py --config config.yaml --laser-method adaptive --verbose
"""

import sys
import argparse
import numpy as np
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from main_controller_streaming import LaserEyeControllerStreaming


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Laser Eye Tracking System with Classical Signal Processing'
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum frames to process (None = infinite)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print debug information')
    parser.add_argument('--stream', action='store_true', default=True,
                        help='Enable camera stream visualization')
    parser.add_argument('--laser-method', type=str, default=None,
                        choices=['hsv', 'adaptive', 'bgr', 'brightness', 'temporal', 'hybrid'],
                        help='Laser detection method (overrides config.yaml)')
    parser.add_argument('--no-stream', dest='stream', action='store_false',
                        help='Disable camera stream visualization')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"[STARTUP] Loading config from {args.config}")
    config = load_config(args.config)
    
    # Extract servo limits
    servo_limits = np.array(config['servo_limits'], dtype=np.float32)

    # Extract model path
    pose_model = config['models']['pose']

    # Optional: Load camera calibration
    camera_matrix = None
    if 'camera_calibration' in config:
        calib_file = config['camera_calibration']
        if Path(calib_file).exists():
            calib_data = np.load(calib_file)
            camera_matrix = calib_data['camera_matrix']
            print(f"[STARTUP] Loaded camera calibration from {calib_file}")

    # Build laser config: CLI --laser-method overrides config.yaml method field
    laser_config = config.get('detectors', {}).get('laser', {})
    if args.laser_method is not None:
        laser_config = dict(laser_config)   # copy so we don't mutate config
        laser_config['method'] = args.laser_method

    laser_method = laser_config.get('method', 'hsv')

    pose_config = config.get('detectors', {}).get('pose', {})

    # Initialize controller with streaming
    print("[STARTUP] Initializing LaserEyeControllerStreaming...")
    controller = LaserEyeControllerStreaming(
        servo_limits=servo_limits,
        pose_model_path=pose_model,
        laser_config=laser_config,
        pose_config=pose_config,
        camera_matrix=camera_matrix,
        loop_rate_hz=config.get('loop_rate_hz', 30.0),
        verbose=args.verbose,
        enable_visualization=args.stream
    )
    print(f"[STARTUP] Pose inference device: {pose_config.get('device', 'cpu')}")

    # Print laser detection parameters
    print(f"[STARTUP] Laser Detection Method: {laser_method}")
    print(f"[STARTUP] HSV Ranges: {laser_config.get('hsv_h_ranges', [[0,8],[172,180]])}")
    print(f"[STARTUP] HSV S/V min: {laser_config.get('hsv_s_min', 150)} / {laser_config.get('hsv_v_min', 150)}")
    print(f"[STARTUP] Confidence Threshold: {laser_config.get('confidence_threshold', 0.5)}")
    
    # Center eyes to safe position
    print("[STARTUP] Centering eyes...")
    controller.servo_controller.center_eyes()
    
    # Run main loop
    print(f"[STARTUP] Starting main loop (camera_id={args.camera}, method={args.laser_method})...")
    print("[RUNNING] Press 'q' in stream to stop, or Ctrl+C")
    
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
