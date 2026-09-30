//! The decisions: which frames count, when a rep completes, what to say at a
//! milestone, and where to crop a frame square. No camera, no model. The
//! caller passes the pose in.

/// One body keypoint: its vertical position in the frame, 0 at the top and 1 at
/// the bottom, and the model's confidence in it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Keypoint {
    pub y: f32,
    pub confidence: f32,
}

/// The four keypoints a pullup is judged on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose {
    pub left_shoulder: Keypoint,
    pub right_shoulder: Keypoint,
    pub left_wrist: Keypoint,
    pub right_wrist: Keypoint,
}

impl Pose {
    /// Mean confidence across the four keypoints.
    pub fn confidence(&self) -> f32 {
        (self.left_shoulder.confidence
            + self.right_shoulder.confidence
            + self.left_wrist.confidence
            + self.right_wrist.confidence)
            / 4.0
    }

    /// How far the shoulders hang below the wrists, as a fraction of the frame.
    /// Large when hanging, near zero at the top of a pullup.
    pub fn hang(&self) -> f32 {
        f32::midpoint(self.left_shoulder.y, self.right_shoulder.y)
            - f32::midpoint(self.left_wrist.y, self.right_wrist.y)
    }
}

/// The hang values that flip the counter, and the confidence a frame needs to
/// be looked at. The gap between `up` and `down` is what keeps a wobble at
/// either end from counting twice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Thresholds {
    /// A hang below this is the top of a rep.
    pub up: f32,
    /// A hang above this is back at the bottom.
    pub down: f32,
    pub min_confidence: f32,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            up: 0.05,
            down: 0.15,
            min_confidence: 0.4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Down,
    Up,
}

/// Counts reps from a stream of poses. A rep is a rise past `up` from below;
/// the next one needs a drop past `down` first.
#[derive(Debug, Clone)]
pub struct Counter {
    thresholds: Thresholds,
    phase: Phase,
    reps: u32,
}

impl Counter {
    pub fn new(thresholds: Thresholds) -> Self {
        Self {
            thresholds,
            phase: Phase::Down,
            reps: 0,
        }
    }

    pub fn reps(&self) -> u32 {
        self.reps
    }

    /// Feeds one frame's pose. Returns the new total when this frame completes
    /// a rep. A frame under the confidence floor changes nothing.
    pub fn observe(&mut self, pose: &Pose) -> Option<u32> {
        if pose.confidence() < self.thresholds.min_confidence {
            return None;
        }
        let hang = pose.hang();
        match self.phase {
            Phase::Down if hang < self.thresholds.up => {
                self.phase = Phase::Up;
                self.reps += 1;
                Some(self.reps)
            }
            Phase::Up if hang > self.thresholds.down => {
                self.phase = Phase::Down;
                None
            }
            _ => None,
        }
    }
}

/// Something to say when the count reaches a milestone.
pub fn milestone(reps: u32) -> Option<&'static str> {
    Some(match reps {
        5 => "Just getting started!",
        10 => "That's what I'm talking about!",
        15 => "Holy smokes, man!",
        20 => "Didn't know I was hanging with BIG DOG!",
        25 => "Is anybody else seeing this?!",
        30 => "I don't have words for what I am currently witnessing. I am at your mercy, my lord.",
        31.. => "All hail the Pullup God!",
        _ => return None,
    })
}

/// The centered square inside a `width` by `height` frame: its top-left corner
/// and side.
pub fn square_bounds(width: u32, height: u32) -> (u32, u32, u32) {
    let side = width.min(height);
    ((width - side) / 2, (height - side) / 2, side)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pose(hang: f32, confidence: f32) -> Pose {
        let point = |y| Keypoint { y, confidence };
        Pose {
            left_shoulder: point(0.5 + hang),
            right_shoulder: point(0.5 + hang),
            left_wrist: point(0.5),
            right_wrist: point(0.5),
        }
    }

    #[test]
    fn hang_is_how_far_the_shoulders_sit_below_the_wrists() {
        let pose = Pose {
            left_shoulder: Keypoint {
                y: 0.6,
                confidence: 1.0,
            },
            right_shoulder: Keypoint {
                y: 0.8,
                confidence: 1.0,
            },
            left_wrist: Keypoint {
                y: 0.4,
                confidence: 1.0,
            },
            right_wrist: Keypoint {
                y: 0.2,
                confidence: 1.0,
            },
        };
        assert!((pose.hang() - 0.4).abs() < 1e-6);
    }

    #[test]
    fn confidence_is_the_mean_of_the_four() {
        let mut pose = pose(0.0, 1.0);
        pose.left_wrist.confidence = 0.0;
        assert!((pose.confidence() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn a_rise_from_the_bottom_counts_one_rep() {
        let mut counter = Counter::new(Thresholds::default());
        assert_eq!(counter.observe(&pose(0.3, 1.0)), None);
        assert_eq!(counter.observe(&pose(0.02, 1.0)), Some(1));
        assert_eq!(counter.reps(), 1);
    }

    #[test]
    fn staying_at_the_top_does_not_count_again() {
        let mut counter = Counter::new(Thresholds::default());
        assert_eq!(counter.observe(&pose(0.02, 1.0)), Some(1));
        for _ in 0..10 {
            assert_eq!(counter.observe(&pose(0.01, 1.0)), None);
        }
        assert_eq!(counter.reps(), 1);
    }

    #[test]
    fn the_next_rep_needs_a_full_drop_first() {
        let mut counter = Counter::new(Thresholds::default());
        assert_eq!(counter.observe(&pose(0.02, 1.0)), Some(1));
        // Dips into the band between the thresholds and comes back up: no rep.
        assert_eq!(counter.observe(&pose(0.10, 1.0)), None);
        assert_eq!(counter.observe(&pose(0.02, 1.0)), None);
        // Drops past `down`, rises past `up`: a rep.
        assert_eq!(counter.observe(&pose(0.20, 1.0)), None);
        assert_eq!(counter.observe(&pose(0.02, 1.0)), Some(2));
    }

    #[test]
    fn a_wobble_inside_the_band_flips_nothing() {
        let mut counter = Counter::new(Thresholds::default());
        for hang in [0.10, 0.08, 0.12, 0.06, 0.14] {
            assert_eq!(counter.observe(&pose(hang, 1.0)), None);
        }
        assert_eq!(counter.reps(), 0);
    }

    #[test]
    fn a_low_confidence_frame_is_ignored_whatever_it_shows() {
        let mut counter = Counter::new(Thresholds::default());
        assert_eq!(counter.observe(&pose(0.02, 0.39)), None);
        assert_eq!(counter.reps(), 0);
        assert_eq!(counter.observe(&pose(0.02, 0.4)), Some(1));
        assert_eq!(counter.observe(&pose(0.30, 0.1)), None);
        assert_eq!(
            counter.observe(&pose(0.02, 1.0)),
            None,
            "the ignored drop did not reset the phase"
        );
    }

    #[test]
    fn thresholds_are_the_callers() {
        let mut counter = Counter::new(Thresholds {
            up: 0.5,
            down: 0.9,
            min_confidence: 0.0,
        });
        assert_eq!(counter.observe(&pose(0.45, 0.0)), Some(1));
        assert_eq!(counter.observe(&pose(0.85, 0.0)), None);
        assert_eq!(counter.observe(&pose(0.45, 0.0)), None);
        assert_eq!(counter.observe(&pose(0.95, 0.0)), None);
        assert_eq!(counter.observe(&pose(0.45, 0.0)), Some(2));
    }

    #[test]
    fn milestones_land_on_every_fifth_rep_and_then_stay() {
        assert_eq!(milestone(1), None);
        assert_eq!(milestone(4), None);
        assert_eq!(milestone(5), Some("Just getting started!"));
        assert_eq!(milestone(6), None);
        assert!(milestone(30).is_some_and(|m| m.ends_with("my lord.")));
        assert_eq!(milestone(31), Some("All hail the Pullup God!"));
        assert_eq!(milestone(100), Some("All hail the Pullup God!"));
    }

    #[test]
    fn square_bounds_center_the_short_side() {
        assert_eq!(square_bounds(1920, 1080), (420, 0, 1080));
        assert_eq!(square_bounds(1080, 1920), (0, 420, 1080));
        assert_eq!(square_bounds(640, 640), (0, 0, 640));
        assert_eq!(square_bounds(7, 4), (1, 0, 4));
    }
}
