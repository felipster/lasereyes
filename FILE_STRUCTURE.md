# File Structure Documentation

This document describes the new project structure with modules organized under `src/` instead of `core/`.

## Directory Layout

```
lasereyes/
├── src/                           # Source code modules
│   ├── __init__.py               # Package initialization
│   ├── servo_controller.py       # ServoController class
│   ├── pose_detector.py          # PoseDetector class (YOLO11-pose wrapper)
│   ├── laser_detector.py         # LaserDetector class (YOLO11-detect wrapper)
│   ├── tracking_controller.py    # TrackingController class
│   └── pid_controller.py         # PIDController class
│
├── tests/                         # Unit and integration tests
│   ├── test_servo_controller.py  # ServoController tests
│   ├── test_pose_detector.py     # PoseDetector tests
│   ├── test_pid_controller.py    # PIDController tests
│   └── test_tracking.py          # TrackingController integration tests
│
├── main_controller.py            # LaserEyeController orchestrator class
├── run_closed_loop.py            # Entry point script
├── config.yaml                   # Configuration file
├── README.md                      # Original project README
├── LICENSE                        # License file
└── agent_chat/                    # Documentation and analysis
    ├── closed_loop_architecture.md       # Complete architecture reference
    ├── gaze_estimation_3d_analysis.md    # 3D gaze mathematical analysis
    ├── INTEGRATION_SUMMARY.md            # Integration checklist
    └── ... (other documentation)
```

## Module Structure

### `src/` - Core Modules

Each module implements a specific component of the system:

#### `servo_controller.py`
- **Class**: `ServoController`
- **Purpose**: Hardware interface to PCA9685 servo driver
- **Dependencies**: `board`, `busio`, `adafruit_pca9685`, `adafruit_servokit`
- **Key Methods**:
  - `set_angle(channel, angle)` - Set individual servo angle with bounds checking
  - `set_eye_angles(left_az, left_el, right_az, right_el)` - Set all eye servos at once
  - `center_eyes()` - Move eyes to center position (safe)
  - `emergency_stop()` - Emergency shutdown routine

#### `pose_detector.py`
- **Class**: `PoseDetector`
- **Purpose**: YOLO11-pose model wrapper with 3D gaze estimation
- **Dependencies**: `ultralytics`, `numpy`
- **Key Methods**:
  - `detect(frame)` - Run pose detection on frame
  - `pixel_to_normalized_3d(u, v)` - Convert pixel → 3D ray using K matrix
  - `get_gaze_direction_3d(detection)` - Get 3D gaze vector
  - `get_gaze_angles_from_3d(gaze_3d)` - Convert 3D vector → (azimuth, elevation)
  - `get_head_pose_euler(detection)` - Estimate head pose (roll, pitch, yaw)

#### `laser_detector.py`
- **Class**: `LaserDetector`
- **Purpose**: YOLO11-detect model wrapper for laser dot detection
- **Dependencies**: `ultralytics`, `opencv-python`, `numpy`
- **Key Methods**:
  - `detect(frame)` - Detect laser dots in frame
  - Uses optional HSV pre-filtering for speed

#### `tracking_controller.py`
- **Class**: `TrackingController`
- **Purpose**: Multi-target tracking with Kalman filtering and PID control
- **Dependencies**: `opencv-python`, `numpy`, `pid_controller`
- **Key Methods**:
  - `update(pose_detection, laser_detections, dt, frame_shape, pose_detector)` - Main update loop
  - `_extract_target_positions()` - Extract target eye angles from pose detection
  - `_associate_lasers()` - Associate detected dots to eyes
  - `_compute_errors()` - Compute tracking errors

#### `pid_controller.py`
- **Class**: `PIDController`
- **Purpose**: Simple PID controller for servo axis feedback
- **Dependencies**: None (pure Python)
- **Key Methods**:
  - `update(error, dt)` - Compute PID output
  - `reset()` - Clear controller state

### Root Level Scripts

#### `main_controller.py`
- **Class**: `LaserEyeController`
- **Purpose**: Main orchestrator of the entire system
- **Instantiates**: All subsystem controllers
- **Key Methods**:
  - `run(camera_source, max_frames)` - Main execution loop
  - `shutdown(cap)` - Clean shutdown

#### `run_closed_loop.py`
- **Type**: Entry point script
- **Purpose**: Loads config, initializes system, runs main loop
- **Usage**: `python3 run_closed_loop.py --config config.yaml`

#### `config.yaml`
- **Type**: Configuration file (YAML)
- **Contains**: Servo limits, model paths, PID gains, Kalman parameters
- **Customization**: Edit for your specific setup

## Import Structure

### For Users of the Library

```python
from src.servo_controller import ServoController
from src.pose_detector import PoseDetector
from src.laser_detector import LaserDetector
from src.tracking_controller import TrackingController
from src.pid_controller import PIDController
from main_controller import LaserEyeController
```

### For Running the System

```bash
python3 run_closed_loop.py --config config.yaml
```

## Testing

Run all tests:
```bash
python3 -m pytest tests/
```

Run specific test file:
```bash
python3 -m pytest tests/test_servo_controller.py -v
```

## Dependencies

### Required Packages
- `ultralytics` - YOLO11 model framework
- `opencv-python` - Computer vision and image processing
- `numpy` - Numerical computing
- `PyYAML` - YAML configuration parsing

### Hardware Packages (RPi only)
- `board` - Blinka board definitions
- `busio` - Bus communication
- `adafruit-pca9685` - PCA9685 servo driver
- `adafruit-servokit` - ServoKit abstraction layer

### Development Packages
- `pytest` - Testing framework
- `pytest-cov` - Code coverage

## Configuration

All runtime parameters are in `config.yaml`:

```yaml
loop_rate_hz: 30                    # Main loop frequency
models:
  pose: "path/to/model.pt"         # Pose detection model
  laser: "path/to/model.pt"        # Laser detection model
servo_limits: [...]                 # Servo angle bounds
pid_gains: {...}                    # PID tuning parameters
```

## Camera Calibration

To calibrate your camera and obtain K matrix:

```python
from src.pose_detector import calibrate_camera, capture_calibration_images

# Step 1: Capture images
capture_calibration_images('./calibration_images/', num_images=25)

# Step 2: Run calibration
K, dist = calibrate_camera('./calibration_images/')

# Step 3: Save and use
np.savez('calibration.npz', camera_matrix=K, distortion_coefficients=dist)
```

Then update `config.yaml`:
```yaml
camera_calibration: "calibration.npz"
```

## Architecture Diagram

```
run_closed_loop.py (entry point)
        ↓
    config.yaml (parameters)
        ↓
LaserEyeController (main orchestrator)
        ├─→ ServoController (hardware output)
        ├─→ PoseDetector (target detection)
        ├─→ LaserDetector (feedback detection)
        └─→ TrackingController
            ├─→ PIDController × 4 (servo control)
            └─→ Kalman Filters × 2 (position smoothing)
```

## Running the System

### Basic Usage

```bash
# Run with default configuration
python3 run_closed_loop.py

# Run with custom config
python3 run_closed_loop.py --config my_config.yaml

# Run on specific camera with debug output
python3 run_closed_loop.py --camera 0 --verbose

# Run for limited frames (testing)
python3 run_closed_loop.py --max-frames 100
```

### Development Usage

```python
# Manual control for testing
import numpy as np
from src.servo_controller import ServoController

servo_limits = np.array([
    [-22.5, 22.5],
    [-30.0, 30.0],
    [-22.5, 22.5],
    [-30.0, 30.0],
    [0.0, 90.0],
    [0.0, 90.0]
])

controller = ServoController(servo_limits)
controller.center_eyes()  # Safe position
controller.set_eye_angles(0, 5, 0, -5)  # Set specific angles
```

## Migration from `core/` to `src/`

If you had code using the old `core/` structure, update imports:

### Old (core/)
```python
from core.servo_controller import ServoController
```

### New (src/)
```python
from src.servo_controller import ServoController
```

All class names and functionality remain identical—only the module path changed.

## Further Documentation

See `agent_chat/` directory for detailed documentation:

- **`closed_loop_architecture.md`** - Complete architecture reference with class implementations
- **`gaze_estimation_3d_analysis.md`** - Mathematical foundation for 3D gaze estimation
- **`INTEGRATION_SUMMARY.md`** - Integration checklist and parameter table
- **`augmentation_and_gpu_guide.md`** - Dataset training recommendations
- **`phase2_dataset_analysis.md`** - Dataset collection strategy
