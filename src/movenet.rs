//! `MoveNet` through tract: a frame in, the four keypoints out.

use crate::{
    camera::Frame,
    domain::{Keypoint, Pose, square_bounds},
};
use image::imageops::{FilterType, crop_imm, resize};
use tract_onnx::prelude::*;

const MODEL_PATH: &str = "models/movenet-onnx-float/movenet.onnx";
/// `MoveNet` Lightning takes a square of this many pixels.
const INPUT_SIDE: u32 = 192;

/// Indices into `MoveNet`'s 17 keypoints.
const LEFT_SHOULDER: usize = 5;
const RIGHT_SHOULDER: usize = 6;
const LEFT_WRIST: usize = 9;
const RIGHT_WRIST: usize = 10;

type Model = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

pub struct MoveNet(Model);

impl MoveNet {
    pub fn load() -> anyhow::Result<Self> {
        Ok(Self(
            tract_onnx::onnx()
                .model_for_path(MODEL_PATH)?
                .into_optimized()?
                .into_runnable()?,
        ))
    }

    /// Crops the frame square, scales it to the model's input, and reads the
    /// shoulders and wrists out of the answer.
    pub fn pose(&self, frame: &Frame) -> anyhow::Result<Pose> {
        let (x, y, side) = square_bounds(frame.width(), frame.height());
        let square = crop_imm(frame, x, y, side, side).to_image();
        let resized = resize(&square, INPUT_SIDE, INPUT_SIDE, FilterType::Triangle);
        // x and y are bounded by INPUT_SIDE, so the casts cannot truncate.
        #[expect(clippy::cast_possible_truncation)]
        let input: Tensor = tract_ndarray::Array4::from_shape_fn(
            (1, 3, INPUT_SIDE as usize, INPUT_SIDE as usize),
            |(_, c, y, x)| f32::from(resized[(x as u32, y as u32)][c]) / 255.0,
        )
        .into();
        let output = self.0.run(tvec!(input.into()))?;
        let output = output
            .first()
            .ok_or_else(|| anyhow::anyhow!("model returned no output"))?
            .to_array_view::<f32>()?;
        // Each keypoint is (y, x, confidence).
        let keypoint = |index| Keypoint {
            y: output[[0, 0, index, 0]],
            confidence: output[[0, 0, index, 2]],
        };
        Ok(Pose {
            left_shoulder: keypoint(LEFT_SHOULDER),
            right_shoulder: keypoint(RIGHT_SHOULDER),
            left_wrist: keypoint(LEFT_WRIST),
            right_wrist: keypoint(RIGHT_WRIST),
        })
    }
}
