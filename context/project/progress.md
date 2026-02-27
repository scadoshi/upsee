# Progress

## Done

- [ ] Neural network fundamentals (3Blue1Brown playlist)
- [x] Clone tract, run mobilenetv2 example
- [x] Find and download MoveNet ONNX model
- [x] Load model in tract, run inference on test image
- [x] Parse and print 17-keypoint output
- [x] Add nokhwa for webcam capture
- [x] Capture frame and run inference on it
- [x] Pipe frames to tract in continuous loop
- [x] Square crop webcam frames to preserve aspect ratio (improved confidence)
- [x] Implement state machine for rep counting (Down/Up enum with hysteresis)
- [x] Count and display reps in real-time (clear terminal, show count)
- [x] Add confidence threshold to skip low-quality frames
- [x] Motivational messages at rep milestones

## Known Issues

- Miscounts when moving camera around or standing too close; best to start near pullup position
- Confidence still moderate (~0.6-0.7); stepping back from camera improves it
- Quantized model not compatible with tract (QuantizeLinear op unsupported)

## Next

- [ ] Tune thresholds with more real pullup data
- [ ] Temporal smoothing to reduce noisy keypoint jitter
- [ ] Test on Raspberry Pi
- [ ] Multi-threading (capture thread + inference thread) for higher frame throughput

## Learning - AI / ML Fundamentals

### Understood
- How inference works (load model, feed input, read output)
- ONNX as a portable model format
- Keypoint-based pose estimation (MoveNet 17-point body model)
- Tensor shapes and preprocessing (resize, normalize, NCHW layout)
- Hysteresis for state machine noise filtering
- Confidence thresholds for filtering bad frames
- Debug vs release build performance impact on inference

### To Learn
- [ ] Forward propagation basics
- [ ] Difference between training and inference
- [ ] Backpropagation in depth
- [ ] Loss functions and how models are trained
- [ ] Model architectures (CNNs, attention, residual networks)
- [ ] Quantization (why int8 models are faster, accuracy tradeoffs)
- [ ] Fine-tuning / transfer learning (adapting a model to your domain)
- [ ] Data labeling and dataset preparation
- [ ] Evaluation metrics (precision, recall, mAP for detection)
- [ ] Edge deployment (optimizing for ARM/Pi, memory constraints)
- [ ] Multi-model pipelines (detection → classification → action)
- [ ] Temporal smoothing (filtering noisy keypoints across frames)

## Resources

- tract: https://github.com/sonos/tract
- nokhwa: https://docs.rs/nokhwa/latest/nokhwa/
- MoveNet: https://huggingface.co/qualcomm/Movenet
- Neural networks: https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
