use image::{ImageBuffer, Rgb, SubImage, imageops::crop};

pub trait Square<'a, I> {
    fn square(&'a mut self) -> SubImage<&'a mut I>;
}

impl<'a> Square<'a, ImageBuffer<Rgb<u8>, Vec<u8>>> for ImageBuffer<Rgb<u8>, Vec<u8>> {
    fn square(&'a mut self) -> SubImage<&'a mut ImageBuffer<Rgb<u8>, Vec<u8>>> {
        let (width, height) = self.dimensions();
        if width > height {
            let x_offset = (width - height) / 2;
            crop(self, x_offset, 0, height, height)
        } else if height > width {
            let y_offset = (height - width) / 2;
            crop(self, 0, y_offset, height, height)
        } else {
            crop(self, 0, 0, width, height)
        }
    }
}
