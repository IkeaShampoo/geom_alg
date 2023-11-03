use std::{
    fmt,
    time::{Duration, Instant},
};

pub struct Stopwatch {
    accumulated_time: Duration,
    last_time: Instant,
    is_running: bool,
}

impl Stopwatch {
    pub fn new() -> Stopwatch {
        Stopwatch {
            accumulated_time: Duration::from_secs(0),
            last_time: Instant::now(),
            is_running: false,
        }
    }
    pub fn new_running() -> Stopwatch {
        Stopwatch {
            accumulated_time: Duration::from_secs(0),
            last_time: Instant::now(),
            is_running: true,
        }
    }
    pub fn start(&mut self) {
        if !self.is_running {
            self.is_running = true;
            self.last_time = Instant::now();
        }
    }
    pub fn pause(&mut self) {
        if self.is_running {
            self.accumulated_time += Instant::now() - self.last_time;
            self.is_running = false;
        }
    }
    pub fn reset(&mut self) {
        self.accumulated_time = Duration::from_secs(0);
        self.is_running = false;
    }
    pub fn read_nanos(&self) -> u128 {
        match self.is_running {
            true => self.accumulated_time + (Instant::now() - self.last_time),
            false => self.accumulated_time,
        }
        .as_nanos()
    }
}

impl fmt::Display for Stopwatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.read_nanos().fmt(f)
    }
}
