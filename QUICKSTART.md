# Quick Start Guide

## Structure

```
lasereyes/
├── src/                    # 6 core modules
│   ├── servo_controller.py
│   ├── pose_detector.py
│   ├── laser_detector.py
│   ├── tracking_controller.py
│   ├── pid_controller.py
│   └── __init__.py
├── tests/                  # 4 test files
├── main_controller.py
├── run_closed_loop.py
├── config.yaml
├── FILE_STRUCTURE.md
└── IMPLEMENTATION_COMPLETE.md
```

## Running the System

### 1. Configure
Edit `config.yaml` with your model paths:
```yaml
models:
  pose: "path/to/yolo11n-pose-5kpt.pt"
  laser: "path/to/laser_detector_best.pt"
```

### 2. Run
```bash
python3 run_closed_loop.py --config config.yaml
```

### 3. Optional: Debug
```bash
python3 run_closed_loop.py --config config.yaml --verbose
```

## Key Files

| File | Purpose | Lines |
|------|---------|-------|
| `src/servo_controller.py` | Hardware control | 91 |
| `src/pose_detector.py` | Face detection + 3D gaze | 297 |
| `src/laser_detector.py` | Laser dot detection | 86 |
| `src/tracking_controller.py` | Kalman + PID control | 220 |
| `src/pid_controller.py` | PID tuning | 45 |
| `main_controller.py` | System orchestration | 123 |
| `run_closed_loop.py` | Entry point | 90 |
| `config.yaml` | Configuration | 110 |

## Features

- Modular OOP architecture  
- Camera calibration support  
- Proper 3D gaze estimation  
- Kalman filtering (position smoothing)  
- PID control (servo feedback)  
- YAML configuration  
- Comprehensive test suite  
- Emergency stop functionality  

## Camera Calibration (Optional)

```python
from src.pose_detector import capture_calibration_images, calibrate_camera

# Capture images
capture_calibration_images('./calibration_images/', num_images=25)

# Run calibration
K, dist = calibrate_camera('./calibration_images/')
```

## Testing

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific test
python3 -m pytest tests/test_pid_controller.py -v
```

## Documentation

- **FILE_STRUCTURE.md** - Complete structure guide
- **IMPLEMENTATION_COMPLETE.md** - Implementation details
- **closed_loop_architecture.md** - Architecture reference

## Next Steps

1. Update model paths in `config.yaml`
2. [Optional] Calibrate camera
3. Tune PID gains for your servos
4. Run: `python3 run_closed_loop.py`

---
