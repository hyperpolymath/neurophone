//! NeuroSymbolic Phone - Main Orchestrator
//!
//! Integrates all components:
//! - LSM (spiking neural network)
//! - ESN (echo state network)
//! - Bridge (state encoding)
//! - Sensors (phone input)
//! - LLM (local Llama 3.2)
//! - Claude Client (cloud fallback)

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Mutex, RwLock};
use tracing::{info, warn, debug, error};

// Re-export component crates
pub use bridge::{Bridge, BridgeConfig, IntegratedState, StateEncoding};
pub use claude_client::{ClaudeClient, ClaudeConfig, ClaudeModel, HybridInference};
pub use esn::{EchoStateNetwork, EsnConfig, Activation, HierarchicalEsn};
pub use llm::{LocalLlm, LlmConfig, GenerationParams, ChatMessage, Role};
pub use lsm::{LiquidStateMachine, LsmConfig, LifParameters};
pub use sensors::{SensorProcessor, SensorConfig, SensorReading, SensorType};

/// System configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    /// LSM configuration
    pub lsm: LsmConfig,
    /// ESN configuration
    pub esn: EsnConfig,
    /// Bridge configuration
    pub bridge: BridgeConfig,
    /// Sensor configuration
    pub sensor: SensorConfig,
    /// LLM configuration
    pub llm: LlmConfig,
    /// Claude configuration (optional)
    pub claude: Option<ClaudeConfig>,
    /// Processing loop interval (ms)
    pub loop_interval_ms: u64,
    /// Enable debug logging
    pub debug: bool,
}

impl Default for SystemConfig {
    fn default() -> Self {
        // Optimized for Oppo Reno 13 (Dimensity 8350, 12GB RAM)
        Self {
            lsm: LsmConfig {
                dimensions: (8, 8, 8), // 512 neurons
                p_exc: 0.3,
                p_inh: 0.2,
                frac_inh: 0.2,
                spectral_radius: 0.9,
                input_scale: 1.0,
                dt: 1.0, // 1ms timestep
            },
            esn: EsnConfig {
                reservoir_size: 300,
                spectral_radius: 0.95,
                leaking_rate: 0.3,
                input_scale: 0.5,
                feedback_scale: 0.0,
                sparsity: 0.9,
                ridge_param: 1e-6,
                use_bias: true,
                noise_level: 1e-4,
            },
            bridge: BridgeConfig {
                lsm_dim: 512,
                esn_dim: 300,
                llm_dim: 2048,
                history_size: 100,
                sparse_threshold: 0.1,
                temporal_window_ms: 1000.0,
            },
            sensor: SensorConfig {
                sample_rate_hz: 50.0,
                buffer_size: 100,
                lowpass_cutoff_hz: 20.0,
                highpass_cutoff_hz: 0.1,
                output_dim: 32,
            },
            llm: LlmConfig::default(),
            claude: None,
            loop_interval_ms: 20, // 50 Hz
            debug: false,
        }
    }
}

/// System state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemState {
    pub running: bool,
    pub last_update: f64,
    pub lsm_active: bool,
    pub esn_active: bool,
    pub llm_loaded: bool,
    pub claude_connected: bool,
    pub current_neural_state: Option<String>,
    pub processing_latency_ms: f64,
}

impl Default for SystemState {
    fn default() -> Self {
        Self {
            running: false,
            last_update: 0.0,
            lsm_active: false,
            esn_active: false,
            llm_loaded: false,
            claude_connected: false,
            current_neural_state: None,
            processing_latency_ms: 0.0,
        }
    }
}

/// Main neurosymbolic system
pub struct NeuroSymbolicSystem {
    config: SystemConfig,
    state: Arc<RwLock<SystemState>>,

    // Neural components
    lsm: Arc<Mutex<LiquidStateMachine>>,
    esn: Arc<Mutex<EchoStateNetwork>>,
    bridge: Arc<Mutex<Bridge>>,

    // Input processing
    sensors: Arc<Mutex<SensorProcessor>>,

    // Language models
    local_llm: Arc<Mutex<LocalLlm>>,
    hybrid: Arc<Mutex<HybridInference>>,

    // Communication channels
    sensor_tx: Option<mpsc::Sender<SensorReading>>,

    // Timing
    simulation_time_ms: f64,
}

/// System output events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemOutput {
    NeuralState(IntegratedState),
    LlmResponse(String),
    Action(Action),
    Error(String),
}

/// Actions the system can take
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action {
    Vibrate { duration_ms: u32, intensity: f32 },
    Notification { title: String, body: String },
    Speak { text: String },
    AdjustBrightness { level: f32 },
    Custom { name: String, params: serde_json::Value },
}

impl NeuroSymbolicSystem {
    /// Create new system with default configuration
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_config(SystemConfig::default())
    }

    /// Create new system with custom configuration
    pub fn with_config(config: SystemConfig) -> Result<Self, Box<dyn std::error::Error>> {
        info!("Initializing NeuroSymbolic system...");

        // Initialize LSM
        let lsm = LiquidStateMachine::new(config.lsm.clone(), config.sensor.output_dim)?;
        info!("LSM initialized: {} neurons", lsm.size());

        // Initialize ESN
        let esn = EchoStateNetwork::new(
            config.esn.clone(),
            config.lsm.dimensions.0 * config.lsm.dimensions.1 * config.lsm.dimensions.2,
            Activation::Tanh,
        )?;
        info!("ESN initialized: {} neurons", esn.size());

        // Initialize Bridge
        let bridge = Bridge::new(config.bridge.clone());
        info!("Bridge initialized");

        // Initialize Sensors
        let sensors = SensorProcessor::new(config.sensor.clone());
        info!("Sensor processor initialized");

        // Initialize Local LLM
        let local_llm = LocalLlm::new(config.llm.clone());
        info!("Local LLM wrapper initialized");

        // Initialize Hybrid inference
        let hybrid = HybridInference::new(config.claude.clone());
        info!("Hybrid inference initialized");

        Ok(Self {
            config,
            state: Arc::new(RwLock::new(SystemState::default())),
            lsm: Arc::new(Mutex::new(lsm)),
            esn: Arc::new(Mutex::new(esn)),
            bridge: Arc::new(Mutex::new(bridge)),
            sensors: Arc::new(Mutex::new(sensors)),
            local_llm: Arc::new(Mutex::new(local_llm)),
            hybrid: Arc::new(Mutex::new(hybrid)),
            sensor_tx: None,
            simulation_time_ms: 0.0,
        })
    }

    /// Start the system
    pub async fn start(&mut self) -> Result<mpsc::Receiver<SystemOutput>, Box<dyn std::error::Error>> {
        info!("Starting NeuroSymbolic system...");

        // Create channels
        let (sensor_tx, mut sensor_rx) = mpsc::channel::<SensorReading>(256);
        let (output_tx, output_rx) = mpsc::channel::<SystemOutput>(64);

        self.sensor_tx = Some(sensor_tx);

        // Load local LLM
        {
            let mut llm = self.local_llm.lock().await;
            if let Err(e) = llm.load().await {
                warn!("Failed to load local LLM: {}. Running in sensor-only mode.", e);
            } else {
                let mut state = self.state.write().await;
                state.llm_loaded = true;
            }
        }

        // Update state
        {
            let mut state = self.state.write().await;
            state.running = true;
            state.lsm_active = true;
            state.esn_active = true;
        }

        // Clone Arcs for async task
        let lsm = Arc::clone(&self.lsm);
        let esn = Arc::clone(&self.esn);
        let bridge = Arc::clone(&self.bridge);
        let sensors = Arc::clone(&self.sensors);
        let state = Arc::clone(&self.state);
        let loop_interval = Duration::from_millis(self.config.loop_interval_ms);

        // Spawn main processing loop
        tokio::spawn(async move {
            let mut sim_time = 0.0f64;
            let dt = 1.0; // 1ms simulation timestep

            loop {
                let loop_start = Instant::now();

                // Check if still running
                {
                    let s = state.read().await;
                    if !s.running {
                        break;
                    }
                }

                // Process incoming sensor data
                while let Ok(reading) = sensor_rx.try_recv() {
                    let mut sens = sensors.lock().await;
                    if let Err(e) = sens.add_reading(reading) {
                        warn!("Sensor error: {}", e);
                    }
                }

                // Extract sensor features
                let sensor_features = {
                    let mut sens = sensors.lock().await;
                    match sens.process() {
                        Ok(features) => Some(features.vector.clone()),
                        Err(_) => None,
                    }
                };

                // Run neural processing if we have input
                if let Some(input) = sensor_features {
                    // LSM step
                    let lsm_output = {
                        let mut l = lsm.lock().await;
                        l.step(&input)
                    };

                    // Get LSM state (firing rates)
                    let lsm_state = {
                        let l = lsm.lock().await;
                        l.get_state(100.0) // 100ms window
                    };

                    // ESN step (uses LSM output)
                    let esn_state = {
                        let mut e = esn.lock().await;
                        e.step(&lsm_output)
                    };

                    // Bridge: integrate states
                    let integrated = {
                        let mut b = bridge.lock().await;
                        b.integrate_states(&lsm_state, &esn_state, sim_time as f32)
                    };

                    // Send output if integration succeeded
                    if let Ok(integrated_state) = integrated {
                        let _ = output_tx.send(SystemOutput::NeuralState(integrated_state.clone())).await;

                        // Update system state
                        let context = {
                            let b = bridge.lock().await;
                            b.generate_llm_context()
                        };

                        let mut s = state.write().await;
                        s.current_neural_state = Some(context);
                        s.last_update = sim_time;
                    }
                }

                sim_time += dt;

                // Update processing latency
                {
                    let mut s = state.write().await;
                    s.processing_latency_ms = loop_start.elapsed().as_secs_f64() * 1000.0;
                }

                // Sleep to maintain loop rate
                let elapsed = loop_start.elapsed();
                if elapsed < loop_interval {
                    tokio::time::sleep(loop_interval - elapsed).await;
                }
            }

            info!("Main processing loop ended");
        });

        info!("NeuroSymbolic system started");
        Ok(output_rx)
    }

    /// Stop the system
    pub async fn stop(&mut self) {
        info!("Stopping NeuroSymbolic system...");
        let mut state = self.state.write().await;
        state.running = false;
    }

    /// Send a sensor reading
    pub async fn send_sensor(&self, reading: SensorReading) -> Result<(), &'static str> {
        if let Some(ref tx) = self.sensor_tx {
            tx.send(reading).await.map_err(|_| "Channel closed")
        } else {
            Err("System not started")
        }
    }

    /// Query the local LLM
    pub async fn query_local(
        &self,
        message: &str,
        params: Option<GenerationParams>,
    ) -> Result<String, String> {
        let mut llm = self.local_llm.lock().await;

        if !llm.is_loaded() {
            return Err("Local LLM not loaded".into());
        }

        // Get neural context
        let neural_ctx = {
            let bridge = self.bridge.lock().await;
            bridge.generate_llm_context()
        };

        // Create message with context
        let chat_msg = ChatMessage {
            role: Role::User,
            content: message.to_string(),
            neural_context: Some(neural_ctx),
        };

        let params = params.unwrap_or_default();

        match llm.chat(chat_msg, params).await {
            Ok(result) => Ok(result.text),
            Err(e) => Err(format!("LLM error: {}", e)),
        }
    }

    /// Query Claude (cloud)
    pub async fn query_claude(
        &self,
        message: &str,
    ) -> Result<String, String> {
        let mut hybrid = self.hybrid.lock().await;

        let claude = hybrid.claude().ok_or("Claude not configured")?;

        // Get neural context
        let neural_ctx = {
            let bridge = self.bridge.lock().await;
            bridge.generate_llm_context()
        };

        match claude.send_message_with_context(message, &neural_ctx).await {
            Ok(response) => Ok(response),
            Err(e) => Err(format!("Claude error: {}", e)),
        }
    }

    /// Smart query: chooses local or cloud based on context
    pub async fn query(
        &self,
        message: &str,
        prefer_local: bool,
    ) -> Result<String, String> {
        let state = self.state.read().await;

        // Estimate complexity (simple heuristic)
        let complexity = (message.len() as f32 / 500.0).min(1.0);

        // Check hybrid inference decision
        let use_cloud = {
            let hybrid = self.hybrid.lock().await;
            hybrid.should_use_cloud(complexity, state.llm_loaded)
        };

        if prefer_local && state.llm_loaded {
            self.query_local(message, None).await
        } else if use_cloud {
            self.query_claude(message).await
        } else if state.llm_loaded {
            self.query_local(message, None).await
        } else {
            Err("No LLM available".into())
        }
    }

    /// Get current system state
    pub async fn get_state(&self) -> SystemState {
        self.state.read().await.clone()
    }

    /// Get current neural state context
    pub async fn get_neural_context(&self) -> String {
        let bridge = self.bridge.lock().await;
        bridge.generate_llm_context()
    }

    /// Reset all neural components
    pub async fn reset(&mut self) {
        {
            let mut lsm = self.lsm.lock().await;
            lsm.reset();
        }
        {
            let mut esn = self.esn.lock().await;
            esn.reset();
        }
        {
            let mut bridge = self.bridge.lock().await;
            bridge.reset();
        }
        {
            let mut sensors = self.sensors.lock().await;
            sensors.reset();
        }

        self.simulation_time_ms = 0.0;
        info!("System reset complete");
    }
}

impl Default for NeuroSymbolicSystem {
    fn default() -> Self {
        Self::new().expect("Failed to create default system")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_system_creation() {
        let sys = NeuroSymbolicSystem::new().unwrap();
        let state = sys.get_state().await;
        assert!(!state.running);
    }

    #[tokio::test]
    async fn test_system_start_stop() {
        let mut sys = NeuroSymbolicSystem::new().unwrap();
        let _rx = sys.start().await.unwrap();

        let state = sys.get_state().await;
        assert!(state.running);

        sys.stop().await;
        tokio::time::sleep(Duration::from_millis(50)).await;

        let state = sys.get_state().await;
        assert!(!state.running);
    }
}
