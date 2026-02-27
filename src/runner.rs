use image::imageops::{FilterType, resize};
use nokhwa::{
    Camera,
    pixel_format::RgbFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType},
};
use tract_onnx::prelude::*;

use crate::square::Square;

const MODEL_PATH: &str = "models/movenet-onnx-float/movenet.onnx";

// Keypoint indices
const LEFT_SHOULDER: usize = 5;
const RIGHT_SHOULDER: usize = 6;
const LEFT_WRIST: usize = 9;
const RIGHT_WRIST: usize = 10;

// Hysteresis thresholds for shoulder_y - wrist_y
// (large = hanging/down, small = pulled up)
const UP_THRESHOLD: f32 = 0.05; // shoulders nearly at wrist/bar level
const DOWN_THRESHOLD: f32 = 0.15; // shoulders dropped away from bar
const MIN_CONFIDENCE: f32 = 0.4; // skip frames with low keypoint confidence

#[derive(PartialEq)]
enum PullupState {
    Down,
    Up,
}

pub struct Runner;

impl Runner {
    pub fn run() -> anyhow::Result<()> {
        let mut camera = Camera::new(
            CameraIndex::Index(0),
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate),
        )?;
        camera.open_stream()?;
        for _ in 0..30 {
            camera.frame()?;
        }

        let model = tract_onnx::onnx()
            .model_for_path(MODEL_PATH)?
            .into_optimized()?
            .into_runnable()?;

        let mut state = PullupState::Down;
        let mut reps = 0u32;

        println!("Count: {}", reps);
        loop {
            let mut image = camera.frame()?.decode_image::<RgbFormat>()?;
            let resized = resize(image.square().inner(), 192, 192, FilterType::Triangle);
            let image: Tensor =
                tract_ndarray::Array4::from_shape_fn((1, 3, 192, 192), |(_, c, y, x)| {
                    resized[(x as _, y as _)][c] as f32 / 255.0
                })
                .into();
            let result = model.run(tvec!(image.into()))?;
            let output = result[0].to_array_view::<f32>()?;

            let avg_confidence = (output[[0, 0, LEFT_SHOULDER, 2]]
                + output[[0, 0, RIGHT_SHOULDER, 2]]
                + output[[0, 0, LEFT_WRIST, 2]]
                + output[[0, 0, RIGHT_WRIST, 2]])
                / 4.0;

            if avg_confidence < MIN_CONFIDENCE {
                continue;
            }

            let avg_shoulder_y =
                (output[[0, 0, LEFT_SHOULDER, 0]] + output[[0, 0, RIGHT_SHOULDER, 0]]) / 2.0;
            let avg_wrist_y =
                (output[[0, 0, LEFT_WRIST, 0]] + output[[0, 0, RIGHT_WRIST, 0]]) / 2.0;
            let diff = avg_shoulder_y - avg_wrist_y;

            match state {
                PullupState::Down => {
                    if diff < UP_THRESHOLD {
                        state = PullupState::Up;
                        reps += 1;
                        print!("\x1B[2J\x1B[H");
                        println!("Count: {}", reps);
                        if reps == 5 {
                            println!("\nJust getting started!");
                        } else if reps == 10 {
                            println!("\nThat's what I'm talking about!");
                        } else if reps == 15 {
                            println!("\nHoly smokes, man!");
                        } else if reps == 20 {
                            println!("\nDidn't know I was hanging with BIG DOG!");
                        } else if reps == 25 {
                            println!("\nIs anybody else seeing this?!");
                        } else if reps == 30 {
                            println!(
                                "\nI don't have words for what I am currently witnessing. I am at your mercy, my lord."
                            );
                        } else if reps > 30 {
                            println!("\nAll hail the Pullup God!");
                        }
                    }
                }
                PullupState::Up => {
                    if diff > DOWN_THRESHOLD {
                        state = PullupState::Down;
                    }
                }
            }
            if false {
                break;
            }
        }
        Ok(())
    }
}
