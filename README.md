# upsee

Camera + AI pose estimation in Rust. Uses [MoveNet](https://huggingface.co/qualcomm/Movenet) via [tract](https://github.com/sonos/tract) for on-device keypoint detection. Starting as a pullup counter, built to generalize.

## Setup

### Prerequisites

- Rust toolchain (`rustup`)
- USB webcam

### Get the model

Download a MoveNet ONNX model and unzip it in the project root so you have `movenet-onnx-float/movenet.onnx` at the top level:

```sh
wget https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/movenet/releases/v0.46.0/movenet-onnx-float.zip
unzip movenet-onnx-float.zip
```

This downloads the full-precision float model. A [quantized version](https://huggingface.co/qualcomm/Movenet) is also available if you need smaller/faster inference (e.g. on a Raspberry Pi).

### Run

```sh
cargo run
```

Captures a frame from your webcam, runs MoveNet inference, and prints 17 body keypoints with coordinates and confidence scores.

## License

[MIT](LICENSE)
