# upsee

Real-time pullup counter in Rust. Uses a webcam and [MoveNet](https://huggingface.co/qualcomm/Movenet) pose estimation via [tract](https://github.com/sonos/tract) to track shoulder and wrist keypoints, detect pullup reps with a state machine, and display a live count in the terminal.

## How it works

1. Captures frames from your webcam (`src/camera.rs`)
2. Square-crops and resizes to 192x192, runs MoveNet on-device, and reads the shoulder and wrist keypoints out (`src/movenet.rs`)
3. Feeds each pose to a counter that flips Down to Up when the shoulders reach the wrists and back when they drop away, with a gap between the two so a wobble at either end never counts twice. Frames under a confidence floor are ignored (`src/domain.rs`)

The counter, the milestone messages and the crop bounds are plain functions with no camera or model behind them, and `cargo test` covers them. The thresholds are `Thresholds::default()` in `src/domain.rs`.

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
