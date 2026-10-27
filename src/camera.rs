//! The webcam: the first camera on the machine, at its highest frame rate.

use std::fmt;

use image::{ImageBuffer, Rgb};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};

/// One decoded RGB frame.
pub(crate) type Frame = ImageBuffer<Rgb<u8>, Vec<u8>>;

/// Frames read and thrown away while the sensor settles.
const WARMUP_FRAMES: usize = 30;

/// An open stream from the first camera.
pub(crate) struct Camera(nokhwa::Camera);

/// Prints nothing from the device.
impl fmt::Debug for Camera {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Camera").finish_non_exhaustive()
    }
}

impl Camera {
    /// Opens the first camera at its highest frame rate and reads past the warmup frames.
    ///
    /// # Errors
    ///
    /// Returns an error if there is no camera, it cannot stream RGB, or a warmup frame fails.
    pub(crate) fn open() -> anyhow::Result<Self> {
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

    /// Reads and decodes the next frame.
    ///
    /// # Errors
    ///
    /// Returns an error if the camera gives no frame or it does not decode as RGB.
    pub(crate) fn frame(&mut self) -> anyhow::Result<Frame> {
        Ok(self.0.frame()?.decode_image::<RgbFormat>()?)
    }
}
