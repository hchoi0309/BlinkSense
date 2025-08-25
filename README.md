# BlinkSense

Real-time drowsiness detection system using computer vision and machine learning to monitor eye-state patterns and alert users when signs of drowsiness are detected.

## Live Demo

**Try it now:** [https://blinksense.onrender.com](https://blinksense.onrender.com)

## Detection Metrics

- **EAR (Eye Aspect Ratio)**: Geometric measure of eye openness
- **P_Open**: CNN probability that eyes are open
- **PERCLOS**: Percentage of time eyes closed over 60-second window
- **Closed Streak**: Continuous duration of closed eyes
- **Alert Thresholds**: 5+ seconds closed OR 15%+ PERCLOS

## Alert System

- **Visual Alert**: Red overlay with warning message
- **Audio Alert**: Beep sound (user interaction required for browser audio)
- **Smart Cooldown**: 30-second alert suppression to prevent spam
- **PERCLOS Reset**: Buffer cleared after alerts for fresh baseline

## Project Structure

```
BlinkSense/
├── src/                    # Python training pipeline
│   ├── models/            # PyTorch model architectures
│   ├── training/          # Model training scripts
│   └── preprocess/        # Data processing utilities
├── blinksense_web/        # Django web application
│   ├── static/           # Client-side assets (JS, CSS, models)
│   ├── templates/        # HTML templates
│   └── realtime/         # Detection views and consumers
├── data/mrl/             # MRL eye dataset
├── model_registry/       # Trained model artifacts
└── configs/              # Configuration files
```

## Usage

1. **Visit Website**: Navigate to [https://blinksense.onrender.com](https://blinksense.onrender.com)
2. **Grant Camera Access**: Allow webcam permissions when prompted
3. **Start Detection**: Click "Start Camera & Detection"
4. **Calibration**: System calibrates EAR threshold for ~15 seconds
5. **Active Monitoring**: Real-time detection with live metrics display
6. **Alert Response**: Take breaks when drowsiness detected

## Model Details

**SimpleResNetEye Architecture:**
- Input: 64x64 grayscale eye crop images
- Output: Binary classification + 64-dim embeddings
- Training: Subject-exclusive validation splits
- Performance: ~95% accuracy on MRL dataset