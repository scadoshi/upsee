//! The webcam: the first camera on the machine, at its highest frame rate.

use image::{ImageBuffer, Rgb};
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType},
};

pub type Frame = ImageBuffer<Rgb<u8>, Vec<u8>>;

/// Frames read and thrown away while the sensor settles.
const WARMUP_FRAMES: usize = 30;

pub struct Camera(nokhwa::Camera);

impl Camera {
    pub fn open() -> anyhow::Result<Self> {
        let mut camera = nokhwa::Camera::new(
            CameraIndex::Index(0),
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate),
        )?;
        camera.open_stream()?;
        for _ in 0..WARMUP_FRAMES {
            camera.frame()?;
        }
        Ok(Self(camera))
    }

    pub fn frame(&mut self) -> anyhow::Result<Frame> {
        Ok(self.0.frame()?.decode_image::<RgbFormat>()?)
    }
}
