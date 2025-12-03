//! Phone Sensor Processing - Temporal Feature Extraction
//!
//! Processes raw sensor data from phone sensors and extracts
//! temporal features suitable for neural network input.
//!
//! Supported sensors:
//! - Accelerometer (motion, gravity)
//! - Gyroscope (rotation)
//! - Magnetometer (compass)
//! - Light sensor
//! - Proximity sensor
//! - Barometer (pressure/altitude)
//! - GPS (location)

use chrono::{DateTime, Utc};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use thiserror::Error;
use tracing::{debug, trace, warn};

/// Sensor processing errors
#[derive(Error, Debug)]
pub enum SensorError {
    #[error("Unknown sensor type: {0}")]
    UnknownSensor(String),
    #[error("Buffer underflow: need {needed} samples, have {have}")]
    BufferUnderflow { needed: usize, have: usize },
    #[error("Invalid reading: {0}")]
    InvalidReading(String),
    #[error("Processing error: {0}")]
    ProcessingError(String),
}

/// Sensor types supported
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SensorType {
    Accelerometer,
    Gyroscope,
    Magnetometer,
    Light,
    Proximity,
    Barometer,
    Gps,
    Microphone,
    Touch,
}

impl SensorType {
    /// Get expected number of values for this sensor type
    pub fn dimensions(&self) -> usize {
        match self {
            SensorType::Accelerometer => 3, // x, y, z
            SensorType::Gyroscope => 3,     // x, y, z
            SensorType::Magnetometer => 3,  // x, y, z
            SensorType::Light => 1,         // lux
            SensorType::Proximity => 1,     // distance (cm or binary)
            SensorType::Barometer => 1,     // pressure (hPa)
            SensorType::Gps => 3,           // lat, lon, accuracy
            SensorType::Microphone => 1,    // amplitude or FFT bin
            SensorType::Touch => 2,         // x, y position
        }
    }
}

/// Accuracy level of sensor reading
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SensorAccuracy {
    Unreliable,
    Low,
    Medium,
    High,
}

/// A single sensor reading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorReading {
    /// Type of sensor
    pub sensor_type: SensorType,
    /// Timestamp of reading
    pub timestamp: DateTime<Utc>,
    /// Raw sensor values
    pub values: Vec<f32>,
    /// Accuracy level
    pub accuracy: SensorAccuracy,
}

/// Processed sensor features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorFeatures {
    /// Combined feature vector for neural network input
    pub vector: Array1<f32>,
    /// Timestamp of latest reading
    pub timestamp: DateTime<Utc>,
    /// Per-sensor statistics
    pub stats: HashMap<SensorType, SensorStats>,
}

/// Statistics for a single sensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorStats {
    /// Mean values over window
    pub mean: Vec<f32>,
    /// Standard deviation
    pub std: Vec<f32>,
    /// Min values
    pub min: Vec<f32>,
    /// Max values
    pub max: Vec<f32>,
    /// Dominant frequency (if applicable)
    pub dominant_freq: Option<f32>,
    /// Activity level (0-1)
    pub activity: f32,
}

/// Configuration for sensor processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorConfig {
    /// Sample rate target (Hz)
    pub sample_rate_hz: f32,
    /// Buffer size for each sensor
    pub buffer_size: usize,
    /// Low-pass filter cutoff (Hz)
    pub lowpass_cutoff_hz: f32,
    /// High-pass filter cutoff (Hz)
    pub highpass_cutoff_hz: f32,
    /// Output feature vector dimension
    pub output_dim: usize,
}

impl Default for SensorConfig {
    fn default() -> Self {
        Self {
            sample_rate_hz: 50.0,
            buffer_size: 100,
            lowpass_cutoff_hz: 20.0,
            highpass_cutoff_hz: 0.1,
            output_dim: 32,
        }
    }
}

/// IIR filter state for real-time filtering
#[derive(Debug, Clone)]
struct IirFilter {
    // Biquad filter coefficients
    b: [f32; 3],
    a: [f32; 3],
    // Filter state (2 previous inputs and outputs)
    x: [f32; 2],
    y: [f32; 2],
}

impl IirFilter {
    /// Create a low-pass Butterworth filter
    fn lowpass(cutoff_hz: f32, sample_rate_hz: f32) -> Self {
        let omega = 2.0 * std::f32::consts::PI * cutoff_hz / sample_rate_hz;
        let cos_omega = omega.cos();
        let sin_omega = omega.sin();
        let alpha = sin_omega / (2.0 * 0.707); // Q = 0.707 (Butterworth)

        let b0 = (1.0 - cos_omega) / 2.0;
        let b1 = 1.0 - cos_omega;
        let b2 = (1.0 - cos_omega) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        Self {
            b: [b0 / a0, b1 / a0, b2 / a0],
            a: [1.0, a1 / a0, a2 / a0],
            x: [0.0, 0.0],
            y: [0.0, 0.0],
        }
    }

    /// Create a high-pass Butterworth filter
    fn highpass(cutoff_hz: f32, sample_rate_hz: f32) -> Self {
        let omega = 2.0 * std::f32::consts::PI * cutoff_hz / sample_rate_hz;
        let cos_omega = omega.cos();
        let sin_omega = omega.sin();
        let alpha = sin_omega / (2.0 * 0.707);

        let b0 = (1.0 + cos_omega) / 2.0;
        let b1 = -(1.0 + cos_omega);
        let b2 = (1.0 + cos_omega) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        Self {
            b: [b0 / a0, b1 / a0, b2 / a0],
            a: [1.0, a1 / a0, a2 / a0],
            x: [0.0, 0.0],
            y: [0.0, 0.0],
        }
    }

    /// Process a single sample
    fn process(&mut self, input: f32) -> f32 {
        let output = self.b[0] * input + self.b[1] * self.x[0] + self.b[2] * self.x[1]
            - self.a[1] * self.y[0] - self.a[2] * self.y[1];

        // Shift state
        self.x[1] = self.x[0];
        self.x[0] = input;
        self.y[1] = self.y[0];
        self.y[0] = output;

        output
    }

    /// Reset filter state
    fn reset(&mut self) {
        self.x = [0.0, 0.0];
        self.y = [0.0, 0.0];
    }
}

/// Sensor buffer with filtering
struct SensorBuffer {
    sensor_type: SensorType,
    readings: VecDeque<SensorReading>,
    max_size: usize,
    lowpass_filters: Vec<IirFilter>,
    highpass_filters: Vec<IirFilter>,
    filtered_buffer: VecDeque<Vec<f32>>,
}

impl SensorBuffer {
    fn new(sensor_type: SensorType, config: &SensorConfig) -> Self {
        let dims = sensor_type.dimensions();

        Self {
            sensor_type,
            readings: VecDeque::with_capacity(config.buffer_size),
            max_size: config.buffer_size,
            lowpass_filters: (0..dims)
                .map(|_| IirFilter::lowpass(config.lowpass_cutoff_hz, config.sample_rate_hz))
                .collect(),
            highpass_filters: (0..dims)
                .map(|_| IirFilter::highpass(config.highpass_cutoff_hz, config.sample_rate_hz))
                .collect(),
            filtered_buffer: VecDeque::with_capacity(config.buffer_size),
        }
    }

    fn add_reading(&mut self, reading: SensorReading) {
        // Validate dimensions
        let expected_dims = self.sensor_type.dimensions();
        if reading.values.len() != expected_dims {
            warn!(
                "Sensor {:?} expected {} values, got {}",
                self.sensor_type, expected_dims, reading.values.len()
            );
            return;
        }

        // Apply filters
        let filtered: Vec<f32> = reading.values.iter()
            .enumerate()
            .map(|(i, &v)| {
                let low = self.lowpass_filters[i].process(v);
                self.highpass_filters[i].process(low)
            })
            .collect();

        self.readings.push_back(reading);
        self.filtered_buffer.push_back(filtered);

        // Trim to max size
        while self.readings.len() > self.max_size {
            self.readings.pop_front();
            self.filtered_buffer.pop_front();
        }
    }

    fn compute_stats(&self) -> Option<SensorStats> {
        if self.filtered_buffer.is_empty() {
            return None;
        }

        let n = self.filtered_buffer.len() as f32;
        let dims = self.sensor_type.dimensions();

        // Compute mean
        let mut mean = vec![0.0f32; dims];
        for sample in &self.filtered_buffer {
            for (i, &v) in sample.iter().enumerate() {
                mean[i] += v;
            }
        }
        for m in &mut mean {
            *m /= n;
        }

        // Compute std, min, max
        let mut variance = vec![0.0f32; dims];
        let mut min = vec![f32::MAX; dims];
        let mut max = vec![f32::MIN; dims];

        for sample in &self.filtered_buffer {
            for (i, &v) in sample.iter().enumerate() {
                variance[i] += (v - mean[i]).powi(2);
                min[i] = min[i].min(v);
                max[i] = max[i].max(v);
            }
        }

        let std: Vec<f32> = variance.iter()
            .map(|&v| (v / n).sqrt())
            .collect();

        // Compute activity level (normalized variance)
        let activity: f32 = std.iter().map(|&s| s * s).sum::<f32>().sqrt();
        let activity = (activity / 10.0).min(1.0); // Normalize

        Some(SensorStats {
            mean,
            std,
            min,
            max,
            dominant_freq: None, // TODO: FFT for frequency analysis
            activity,
        })
    }

    fn reset(&mut self) {
        self.readings.clear();
        self.filtered_buffer.clear();
        for filter in &mut self.lowpass_filters {
            filter.reset();
        }
        for filter in &mut self.highpass_filters {
            filter.reset();
        }
    }
}

/// Main sensor processor
pub struct SensorProcessor {
    config: SensorConfig,
    buffers: HashMap<SensorType, SensorBuffer>,
    last_features: Option<SensorFeatures>,
}

impl SensorProcessor {
    /// Create new sensor processor
    pub fn new(config: SensorConfig) -> Self {
        let mut buffers = HashMap::new();

        // Initialize buffers for common sensors
        for sensor_type in &[
            SensorType::Accelerometer,
            SensorType::Gyroscope,
            SensorType::Magnetometer,
            SensorType::Light,
            SensorType::Proximity,
        ] {
            buffers.insert(*sensor_type, SensorBuffer::new(*sensor_type, &config));
        }

        Self {
            config,
            buffers,
            last_features: None,
        }
    }

    /// Add a sensor reading
    pub fn add_reading(&mut self, reading: SensorReading) -> Result<(), SensorError> {
        // Get or create buffer for this sensor type
        let buffer = self.buffers
            .entry(reading.sensor_type)
            .or_insert_with(|| SensorBuffer::new(reading.sensor_type, &self.config));

        buffer.add_reading(reading);
        Ok(())
    }

    /// Process all sensor data and extract features
    pub fn process(&mut self) -> Result<SensorFeatures, SensorError> {
        let mut all_stats = HashMap::new();
        let mut feature_vector = Vec::new();
        let mut latest_timestamp = Utc::now();

        // Process each sensor buffer
        for (sensor_type, buffer) in &self.buffers {
            if let Some(stats) = buffer.compute_stats() {
                // Add to feature vector
                feature_vector.extend(&stats.mean);
                feature_vector.extend(&stats.std);
                feature_vector.push(stats.activity);

                // Update timestamp
                if let Some(latest) = buffer.readings.back() {
                    if latest.timestamp > latest_timestamp {
                        latest_timestamp = latest.timestamp;
                    }
                }

                all_stats.insert(*sensor_type, stats);
            }
        }

        // Pad or truncate to output dimension
        let output_dim = self.config.output_dim;
        if feature_vector.len() < output_dim {
            feature_vector.resize(output_dim, 0.0);
        } else if feature_vector.len() > output_dim {
            feature_vector.truncate(output_dim);
        }

        let features = SensorFeatures {
            vector: Array1::from(feature_vector),
            timestamp: latest_timestamp,
            stats: all_stats,
        };

        self.last_features = Some(features.clone());
        Ok(features)
    }

    /// Get the last computed features
    pub fn get_last_features(&self) -> Option<&SensorFeatures> {
        self.last_features.as_ref()
    }

    /// Reset all sensor buffers
    pub fn reset(&mut self) {
        for buffer in self.buffers.values_mut() {
            buffer.reset();
        }
        self.last_features = None;
    }

    /// Get activity level for a specific sensor
    pub fn get_activity(&self, sensor_type: SensorType) -> Option<f32> {
        self.buffers
            .get(&sensor_type)
            .and_then(|b| b.compute_stats())
            .map(|s| s.activity)
    }

    /// Get combined activity level across all sensors
    pub fn get_total_activity(&self) -> f32 {
        let mut total = 0.0;
        let mut count = 0;

        for buffer in self.buffers.values() {
            if let Some(stats) = buffer.compute_stats() {
                total += stats.activity;
                count += 1;
            }
        }

        if count > 0 {
            total / count as f32
        } else {
            0.0
        }
    }
}

/// Helper to create sensor readings from raw Android sensor events
pub fn from_android_event(
    sensor_type_id: i32,
    values: &[f32],
    timestamp_ns: i64,
    accuracy: i32,
) -> Option<SensorReading> {
    let sensor_type = match sensor_type_id {
        1 => SensorType::Accelerometer,
        4 => SensorType::Gyroscope,
        2 => SensorType::Magnetometer,
        5 => SensorType::Light,
        8 => SensorType::Proximity,
        6 => SensorType::Barometer,
        _ => return None,
    };

    let accuracy = match accuracy {
        3 => SensorAccuracy::High,
        2 => SensorAccuracy::Medium,
        1 => SensorAccuracy::Low,
        _ => SensorAccuracy::Unreliable,
    };

    // Convert nanoseconds to DateTime
    let secs = timestamp_ns / 1_000_000_000;
    let nsecs = (timestamp_ns % 1_000_000_000) as u32;
    let timestamp = DateTime::from_timestamp(secs, nsecs)
        .unwrap_or_else(Utc::now);

    Some(SensorReading {
        sensor_type,
        timestamp,
        values: values.to_vec(),
        accuracy,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensor_processor_creation() {
        let config = SensorConfig::default();
        let processor = SensorProcessor::new(config);
        assert!(processor.get_last_features().is_none());
    }

    #[test]
    fn test_add_reading() {
        let config = SensorConfig::default();
        let mut processor = SensorProcessor::new(config);

        let reading = SensorReading {
            sensor_type: SensorType::Accelerometer,
            timestamp: Utc::now(),
            values: vec![0.0, 0.0, 9.8],
            accuracy: SensorAccuracy::High,
        };

        assert!(processor.add_reading(reading).is_ok());
    }

    #[test]
    fn test_process_features() {
        let config = SensorConfig {
            output_dim: 32,
            ..Default::default()
        };
        let mut processor = SensorProcessor::new(config);

        // Add multiple readings
        for i in 0..10 {
            let reading = SensorReading {
                sensor_type: SensorType::Accelerometer,
                timestamp: Utc::now(),
                values: vec![0.1 * i as f32, 0.0, 9.8],
                accuracy: SensorAccuracy::High,
            };
            processor.add_reading(reading).unwrap();
        }

        let features = processor.process().unwrap();
        assert_eq!(features.vector.len(), 32);
    }

    #[test]
    fn test_filter() {
        let mut filter = IirFilter::lowpass(10.0, 100.0);

        // Process a step input
        let mut output = 0.0;
        for _ in 0..100 {
            output = filter.process(1.0);
        }

        // Should converge close to 1.0
        assert!((output - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_activity_detection() {
        let config = SensorConfig::default();
        let mut processor = SensorProcessor::new(config);

        // Add quiet readings
        for _ in 0..50 {
            let reading = SensorReading {
                sensor_type: SensorType::Accelerometer,
                timestamp: Utc::now(),
                values: vec![0.0, 0.0, 9.8],
                accuracy: SensorAccuracy::High,
            };
            processor.add_reading(reading).unwrap();
        }

        let quiet_activity = processor.get_activity(SensorType::Accelerometer).unwrap_or(0.0);

        // Reset and add active readings
        processor.reset();
        for i in 0..50 {
            let reading = SensorReading {
                sensor_type: SensorType::Accelerometer,
                timestamp: Utc::now(),
                values: vec![(i as f32 * 0.5).sin() * 5.0, 0.0, 9.8],
                accuracy: SensorAccuracy::High,
            };
            processor.add_reading(reading).unwrap();
        }

        let active_activity = processor.get_activity(SensorType::Accelerometer).unwrap_or(0.0);

        // Active should be higher
        assert!(active_activity > quiet_activity);
    }
}
