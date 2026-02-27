# Progress

## Done

- [x] Neural network fundamentals (3Blue1Brown playlist)
- [x] Clone tract, run mobilenetv2 example
- [x] Find and download MoveNet ONNX model
- [x] Load model in tract, run inference on test image
- [x] Parse and print 17-keypoint output
- [x] Add nokhwa for webcam capture
- [x] Capture frame and run inference on it

## Next

- [ ] Pipe frames to tract in continuous loop
- [ ] Implement state machine for rep counting
- [ ] Track coordinate history with hysteresis
- [ ] Count and display reps in real-time
- [ ] Test on Raspberry Pi

## Learning - AI / ML Fundamentals

### Understood
- Forward propagation basics
- How inference works (load model, feed input, read output)
- ONNX as a portable model format
- Tensor shapes and preprocessing (resize, normalize, NCHW layout)
- Keypoint-based pose estimation (MoveNet 17-point body model)
- Difference between training and inference

### To Learn
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
