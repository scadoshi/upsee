//! upsee: counts pullups from the webcam, live in the terminal.

mod camera;
mod domain;
mod movenet;

use camera::Camera;
use domain::{Counter, Thresholds, milestone};
use movenet::MoveNet;

/// Loads the model and opens the camera, then counts reps until the process is stopped.
fn main() -> anyhow::Result<()> {
    let model = MoveNet::load()?;
    let mut camera = Camera::open()?;
    let mut counter = Counter::new(Thresholds::default());
    println!("Count: {}", counter.reps());
    loop {
        let pose = model.pose(&camera.frame()?)?;
        if let Some(reps) = counter.observe(&pose) {
            // Clears the terminal and moves the cursor home.
            print!("\x1B[2J\x1B[H");
            println!("Count: {reps}");
            if let Some(message) = milestone(reps) {
                println!("\n{message}");
            }
        }
    }
}
