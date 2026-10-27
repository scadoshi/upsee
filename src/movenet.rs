//! `MoveNet` through tract: a frame in, the four keypoints out.

use std::fmt;

use anyhow::Context;
use image::RgbImage;
use image::imageops::{FilterType, crop_imm, resize};
use tract_onnx::prelude::tract_ndarray::{Array3, ArrayViewD, Axis, Ix4};
use tract_onnx::prelude::{
    Framework, Graph, InferenceModelExt, SimplePlan, Tensor, TypedFact, TypedOp, tvec,
};

use crate::camera::Frame;
use crate::domain::{Keypoint, Pose, square_bounds};

/// Where the README's download unpacks the model, relative to the working directory.
const MODEL_PATH: &str = "models/movenet-onnx-float/movenet.onnx";
/// `MoveNet` Lightning takes a square of this many pixels.
const INPUT_SIDE: u32 = 192;

/// Indices into `MoveNet`'s 17 keypoints.
const LEFT_SHOULDER: usize = 5;
const RIGHT_SHOULDER: usize = 6;
const LEFT_WRIST: usize = 9;
const RIGHT_WRIST: usize = 10;

/// The optimized, runnable form of the ONNX graph.
type Model = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// The `MoveNet` Lightning model, loaded and ready to run.
pub(crate) struct MoveNet(Model);

/// Prints nothing from the graph.
impl fmt::Debug for MoveNet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MoveNet").finish_non_exhaustive()
    }
}

impl MoveNet {
    /// Loads and optimizes the model at [`MODEL_PATH`].
    ///
    /// # Errors
    ///
    /// Returns an error if the file is missing or tract cannot load or optimize it.
    pub(crate) fn load() -> anyhow::Result<Self> {
        Ok(Self(
            tract_onnx::onnx()
                .model_for_path(MODEL_PATH)
                .with_context(|| {
                    format!("loading {MODEL_PATH}; the README shows how to download it")
                })?
                .into_optimized()?
                .into_runnable()?,
        ))
    }

    /// Crops the frame square, scales it to the model's input, and reads the
    /// shoulders and wrists out of the answer.
    ///
    /// # Errors
    ///
    /// Returns an error if the input cannot be shaped, or the model fails to run or returns no output or an output of the wrong shape.
    pub(crate) fn pose(&self, frame: &Frame) -> anyhow::Result<Pose> {
        let (x, y, side) = square_bounds(frame.width(), frame.height());
        let square = crop_imm(frame, x, y, side, side).to_image();
        let resized = resize(&square, INPUT_SIDE, INPUT_SIDE, FilterType::Triangle);
        let input = input(resized)?;
        let output = self.0.run(tvec!(input.into()))?;
        let output = output
            .first()
            .ok_or_else(|| anyhow::anyhow!("model returned no output"))?
            .to_array_view::<f32>()?;
        keypoints(&output)
    }
}

/// Reads the shoulders and wrists out of the model's `[1, 1, 17, 3]` output, where each keypoint is (y, x, confidence).
fn keypoints(output: &ArrayViewD<'_, f32>) -> anyhow::Result<Pose> {
    let wrong_shape = || {
        anyhow::anyhow!(
            "model output has shape {:?}, expected [1, 1, 17, 3]",
            output.shape()
        )
    };
    let output = output
        .view()
        .into_dimensionality::<Ix4>()
        .map_err(|_| wrong_shape())?;
    let read = |index, field| {
        output
            .get([0, 0, index, field])
            .copied()
            .ok_or_else(wrong_shape)
    };
    let keypoint = |index| -> anyhow::Result<Keypoint> {
        Ok(Keypoint {
            y: read(index, 0)?,
            confidence: read(index, 2)?,
        })
    };
    Ok(Pose {
        left_shoulder: keypoint(LEFT_SHOULDER)?,
        right_shoulder: keypoint(RIGHT_SHOULDER)?,
        left_wrist: keypoint(LEFT_WRIST)?,
        right_wrist: keypoint(RIGHT_WRIST)?,
    })
}

/// The model's input: the image as 1x3xHxW, channels first, each value scaled to 0..=1.
fn input(image: RgbImage) -> anyhow::Result<Tensor> {
    let height = usize::try_from(image.height())?;
    let width = usize::try_from(image.width())?;
    let channels_last = Array3::from_shape_vec((height, width, 3), image.into_raw())?;
    let channels_first = channels_last.permuted_axes([2, 0, 1]).insert_axis(Axis(0));
    Ok(channels_first
        .mapv(|value| f32::from(value) / 255.0)
        .as_standard_layout()
        .into_owned()
        .into())
}

#[cfg(test)]
mod tests {
    use tract_onnx::prelude::tract_ndarray::{ArrayD, IxDyn};

    use super::*;

    #[test]
    fn input_is_channels_first_and_scaled_to_one() -> anyhow::Result<()> {
        let byte = |value: u32| value.to_le_bytes()[0];
        let image = RgbImage::from_fn(INPUT_SIDE, INPUT_SIDE, |x, y| {
            image::Rgb([byte(x), byte(y), byte(x ^ y)])
        });
        let tensor = input(image.clone())?;
        let view = tensor
            .to_array_view::<f32>()?
            .into_dimensionality::<Ix4>()?;
        let side = usize::try_from(INPUT_SIDE)?;
        assert_eq!(view.shape(), [1, 3, side, side]);
        for ((_, c, y, x), got) in view.indexed_iter() {
            let pixel = image.get_pixel(u32::try_from(x)?, u32::try_from(y)?);
            let want = f32::from(pixel[c]) / 255.0;
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "pixel ({x}, {y}) channel {c}"
            );
        }
        Ok(())
    }

    #[test]
    fn keypoints_read_y_and_confidence_by_movenet_index() -> anyhow::Result<()> {
        let output = ArrayD::from_shape_fn(IxDyn(&[1, 1, 17, 3]), |index| {
            f32::from(u8::try_from(index[2]).unwrap()) * 10.0
                + f32::from(u8::try_from(index[3]).unwrap())
        });
        let pose = keypoints(&output.view())?;
        assert_eq!(
            pose.left_shoulder,
            Keypoint {
                y: 50.0,
                confidence: 52.0
            }
        );
        assert_eq!(
            pose.right_shoulder,
            Keypoint {
                y: 60.0,
                confidence: 62.0
            }
        );
        assert_eq!(
            pose.left_wrist,
            Keypoint {
                y: 90.0,
                confidence: 92.0
            }
        );
        assert_eq!(
            pose.right_wrist,
            Keypoint {
                y: 100.0,
                confidence: 102.0
            }
        );
        Ok(())
    }

    #[test]
    fn an_output_of_the_wrong_shape_is_an_error() {
        for shape in [&[1, 1, 5, 3][..], &[1, 1, 17, 2], &[17, 3], &[]] {
            let output = ArrayD::<f32>::zeros(IxDyn(shape));
            assert!(keypoints(&output.view()).is_err(), "shape {shape:?}");
        }
    }
}
