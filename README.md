# NeuroPhone - NeuroSymbolic AI for Mobile

A neurosymbolic AI system for Oppo Reno 13 (and other Android devices) that combines:
- **LSM (Liquid State Machine)** - Spiking neural network reservoir for temporal sensor processing
- **ESN (Echo State Network)** - Echo state reservoir for state prediction
- **Bridge** - State encoding/decoding between neural and symbolic components
- **Local LLM (Llama 3.2)** - On-device language model for privacy-preserving AI
- **Claude API** - Cloud fallback for complex reasoning tasks

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Oppo Reno 13                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐  │
│  │  Sensors    │─────▶│    LSM      │─────▶│   Bridge    │  │
│  │  (temporal  │      │  (spiking   │      │  (state     │  │
│  │   input)    │      │  reservoir) │      │  encoding)  │  │
│  └─────────────┘      └─────────────┘      └──────┬──────┘  │
│                                                   │         │
│                                                   ▼         │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐  │
│  │   Output    │◀─────│    ESN      │◀────▶│    LLM      │  │
│  │  (actions)  │      │  (echo      │      │ (Llama 3.2) │  │
│  │             │      │  reservoir) │      │             │  │
│  └─────────────┘      └─────────────┘      └─────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ (cloud fallback)
                    ┌─────────────────────┐
                    │      Claude         │
                    │  (complex queries)  │
                    └─────────────────────┘
```

## Project Structure

```
neurophone/
├── Cargo.toml                    # Rust workspace
├── crates/
│   ├── lsm/                      # Liquid State Machine (spiking neurons)
│   ├── esn/                      # Echo State Network (reservoir)
│   ├── bridge/                   # State encoding/context generation
│   ├── sensors/                  # Phone sensor processing
│   ├── llm/                      # Local LLM (Llama 3.2) interface
│   ├── claude-client/            # Claude API client
│   ├── neurophone-core/          # Main orchestrator
│   └── neurophone-android/       # Android JNI bindings
├── android/                      # Android app (Kotlin)
│   └── app/
│       └── src/main/
│           ├── java/ai/neurophone/
│           └── res/
├── scripts/
│   ├── setup.sh                  # Development setup
│   └── build-android.sh          # Cross-compile for Android
└── config/
    └── default.toml              # Default configuration
```

## Components

### LSM (Liquid State Machine)
- 3D grid of Leaky Integrate-and-Fire neurons (default: 8x8x8 = 512 neurons)
- Distance-dependent connectivity
- Excitatory and inhibitory neurons
- Real-time spike processing at 1kHz

### ESN (Echo State Network)
- 300-neuron reservoir with 0.95 spectral radius
- Leaky integrator dynamics
- Ridge regression for output training
- Hierarchical ESN support

### Bridge
- Integrates LSM and ESN states
- Generates natural language context for LLMs
- Temporal pattern detection
- Salience and urgency computation

### Sensors
- Accelerometer, gyroscope, magnetometer
- Light, proximity sensors
- IIR filtering (low-pass, high-pass)
- Feature extraction at 50Hz

### Local LLM
- Llama 3.2 1B/3B support via llama.cpp
- Optimized for Dimensity 8350
- Chat templates and streaming
- Neural context injection

### Claude Client
- Messages API integration
- Automatic retry with exponential backoff
- Hybrid inference (local/cloud decision)
- Neural state context injection

## Getting Started

### Prerequisites

- Rust 1.75+
- Android NDK 26+
- Android Studio (for app development)
- Oppo Reno 13 or Android 8.0+ device

### Setup

```bash
# Clone and setup
cd neurophone
./scripts/setup.sh

# Build native libraries
./scripts/build-android.sh

# Open Android project
# android/ folder in Android Studio
```

### Configuration

Set your Claude API key:
```bash
export ANTHROPIC_API_KEY="your-api-key"
```

Or configure in `config/default.toml`:
```toml
[claude]
api_key = "your-api-key"
model = "claude-sonnet-4-20250514"

[llm]
model_path = "/data/local/tmp/llama-3.2-1b-q4_k_m.gguf"
n_threads = 4
context_size = 2048
```

### Download LLM Model

For on-device inference, download a quantized Llama 3.2 model:

```bash
# Example: Llama 3.2 1B Instruct Q4_K_M (~700MB)
# From: https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF

# Push to device
adb push llama-3.2-1b-instruct-q4_k_m.gguf /data/local/tmp/
```

## Usage

### Basic Query

```kotlin
// Initialize
NativeLib.init()
NativeLib.start()

// Query with neural context
val response = NativeLib.query("What's the current activity?", preferLocal = true)
```

### Direct Neural Context

```kotlin
val context = NativeLib.getNeuralContext()
// Returns formatted neural state:
// [NEURAL_STATE]
// Description: Neural state: moderately active (salience: 0.45)
// Context: Recent activity level: 0.38. Detected patterns: oscillation
// ...
// [/NEURAL_STATE]
```

### Sensor Processing

```kotlin
// Sensors are automatically processed when system is running
// Access via neural context or state JSON
val state = NativeLib.getState() // JSON string
```

## Performance

Optimized for Oppo Reno 13 (Dimensity 8350):

| Component | Latency | Notes |
|-----------|---------|-------|
| Sensor processing | <1ms | 50Hz loop |
| LSM step | <2ms | 512 neurons |
| ESN step | <1ms | 300 neurons |
| Bridge integration | <1ms | Per step |
| Local LLM (1B) | 50-100ms/token | Q4 quantized |
| Claude API | 500-2000ms | Network dependent |

## API Reference

### Kotlin/Android

```kotlin
object NativeLib {
    fun init(configJson: String? = null): Boolean
    fun start(): Boolean
    fun stop()
    fun processSensor(sensorType: Int, values: FloatArray, timestamp: Long, accuracy: Int): Boolean
    fun queryLocal(message: String): String
    fun queryClaude(message: String): String
    fun query(message: String, preferLocal: Boolean = true): String
    fun getNeuralContext(): String
    fun getState(): String
    fun reset()
    fun isRunning(): Boolean
}
```

### Rust

```rust
use neurophone_core::{NeuroSymbolicSystem, SystemConfig};

let mut system = NeuroSymbolicSystem::with_config(config)?;
let _rx = system.start().await?;

// Send sensor data
system.send_sensor(reading).await?;

// Query
let response = system.query("What's happening?", true).await?;

// Get neural context
let context = system.get_neural_context().await;
```

## License

MIT License - See LICENSE file

## Contributing

Contributions welcome! Please read CONTRIBUTING.md first.
