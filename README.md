# upsee

Real-time pullup counter in Rust. Uses a webcam and [MoveNet](https://huggingface.co/qualcomm/Movenet) pose estimation via [tract](https://github.com/sonos/tract) to track shoulder and wrist keypoints, detect pullup reps with a state machine, and display a live count in the terminal.

## How it works

1. Captures frames from your webcam
2. Square-crops and resizes to 192x192 for MoveNet input
3. Runs on-device ONNX inference to extract 17 body keypoints
4. Compares shoulder position to wrist position each frame
5. Uses a Down/Up state machine with hysteresis to count reps
6. Skips low-confidence frames to avoid miscounts

## Setup

### Prerequisites

- Rust toolchain (`rustup`)
- USB webcam

### Get the model

```sh
mkdir -p models
cd models
wget https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/movenet/releases/v0.46.0/movenet-onnx-float.zip
unzip movenet-onnx-float.zip
cd ..
```

### Run

```sh
cargo run --release
```

Stand back far enough that the camera can see your full upper body for best results. The terminal clears and displays the current rep count as you go.

## License

[MIT](LICENSE)
