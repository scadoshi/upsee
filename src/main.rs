mod camera;
mod domain;
mod movenet;

use camera::Camera;
use domain::{Counter, Thresholds, milestone};
use movenet::MoveNet;
use std::process::ExitCode;

fn run() -> anyhow::Result<()> {
    let mut camera = Camera::open()?;
    let model = MoveNet::load()?;
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

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
