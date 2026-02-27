use image::imageops::{FilterType, resize};
use nokhwa::{
    Camera,
    pixel_format::RgbFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType},
};
use std::fs::create_dir_all;
use tract_onnx::prelude::*;

const KEYPOINT_NAMES: [&str; 17] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
];

const FRAMES_PATH: &str = "frames";

fn main() -> anyhow::Result<()> {
    let mut camera = Camera::new(
        CameraIndex::Index(0),
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate),
    )?;
    camera.open_stream()?;
    for _ in 0..30 {
        camera.frame()?;
    }
    create_dir_all(FRAMES_PATH)?;
    camera
        .frame()?
        .decode_image::<RgbFormat>()?
        .save(format!("{}/frame.jpeg", FRAMES_PATH))?;

    let model = tract_onnx::onnx()
        .model_for_path("movenet-onnx-float/movenet.onnx")?
        .into_optimized()?
        .into_runnable()?;

    let image = image::open(format!("{}/frame.jpeg", FRAMES_PATH))
        .unwrap()
        .to_rgb8();
    let resized = resize(&image, 192, 192, FilterType::Triangle);

    // MoveNet expects NCHW format with values in 0-1 range
    let image: Tensor = tract_ndarray::Array4::from_shape_fn((1, 3, 192, 192), |(_, c, y, x)| {
        resized[(x as _, y as _)][c] as f32 / 255.0
    })
    .into();

    let result = model.run(tvec!(image.into()))?;

    // MoveNet outputs [1, 1, 17, 3] - 17 keypoints with (y, x, confidence)
    let output = result[0].to_array_view::<f32>()?;
    println!("Output shape: {:?}", output.shape());

    // Print each keypoint
    for i in 0..17 {
        let y = output[[0, 0, i, 0]];
        let x = output[[0, 0, i, 1]];
        let confidence = output[[0, 0, i, 2]];
        println!(
            "{:15}: x={:.3}, y={:.3}, conf={:.3}",
            KEYPOINT_NAMES[i], x, y, confidence
        );
    }

    Ok(())
}
