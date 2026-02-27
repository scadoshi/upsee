# Up-See

Camera + AI in Rust. Identify and count things on-device using ONNX models via tract. Starting with a pullup counter as proof of concept.

## Tech Stack

### Core Crates

| Crate | Purpose |
|-------|---------|
| `tract-onnx` | ONNX model inference on-device |
| `nokhwa` | Cross-platform webcam capture |
| `image` | Frame manipulation, resizing |
| `anyhow` | Error handling |

### Later

| Crate | Purpose |
|-------|---------|
| `tokio` | Async runtime for continuous capture loop |
| `serde` | Serialize results to JSON |
| `kornia-rs` | Advanced vision ops (native Rust, faster than Python OpenCV) |
| `rayon` | Parallel batch processing |

## Hardware

- Raspberry Pi (3B+, 4, or 5)
- USB webcam
- MicroSD 32GB+

## ONNX

Open Neural Network Exchange. Universal file format for models - the "PDF of machine learning." Models trained in TensorFlow/PyTorch/etc can be loaded in any runtime (like tract for Rust). Not inherently smaller or faster - just portable.

## MoveNet Model

Source: https://huggingface.co/qualcomm/Movenet

Two versions:
- `movenet-onnx-float.zip` (full precision) - currently using
- `movenet-onnx-w8a16_mixed_int16.zip` (quantized, smaller/faster for Pi)

**Input**: 192x192 pixels, NCHW format, values 0-1 range
**Output**: [1, 1, 17, 3] - 17 keypoints, each with (y, x, confidence)

17 keypoints: nose, left/right eye, left/right ear, left/right shoulder, left/right elbow, left/right wrist, left/right hip, left/right knee, left/right ankle

For pullups: indices 5-6 (shoulders), 9-10 (wrists), 0 (nose for chin-over-bar)

## Architecture

```
[Camera] → nokhwa
    |
    v (frame every ~100ms)
[Preprocessing] → resize 192x192, normalize 0-1
    |
    v
[tract inference] → MoveNet ONNX
    |
    v
[Keypoint extraction] → parse wrist/shoulder/chin coords
    |
    v
[State machine] → detect up/down transitions
    |
    v
[Rep counter] → increment on state change
    |
    v
[Output] → terminal, log, or pipe to API
```

## Rep Counting Logic

State machine:
- **DOWN**: Arms extended, wrists far below shoulders (wrist_y >> shoulder_y)
- **UP**: Chin above bar, wrists near/above shoulders
- **Rep**: DOWN → UP → DOWN transition

Track `wrist_y` relative to `shoulder_y` over frames. Hysteresis threshold to avoid noise.
