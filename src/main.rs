mod runner;
mod square;
fn main() {
    if let Err(e) = runner::Runner::run() {
        eprintln!("{}", e);
    }
}
