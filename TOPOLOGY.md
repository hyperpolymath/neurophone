<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
<!-- TOPOLOGY.md — Project architecture map and completion dashboard -->
<!-- Last updated: 2026-02-19 -->

# NeuroPhone — Project Topology

## System Architecture

```
                        ┌─────────────────────────────────────────┐
                        │              ANDROID USER               │
                        │        (Compose UI / Oppo Reno 13)      │
                        └───────────────────┬─────────────────────┘
                                            │ JNI / Kotlin Bridge
                                            ▼
                        ┌─────────────────────────────────────────┐
                        │           NEUROPHONE CORE (RUST)        │
                        │    (Orchestration, Bridge, Routing)     │
                        └──────────┬───────────────────┬──────────┘
                                   │                   │
                                   ▼                   ▼
                        ┌───────────────────────┐  ┌────────────────────────────────┐
                        │ NEURAL ENGINE (ON-DEV)│  │ INFERENCE LAYER                │
                        │ - LSM (Spiking Resvr) │  │ - Local Llama 3.2 (llama.cpp)  │
                        │ - ESN (Echo Resvr)    │  │ - Claude API (Fallback)        │
                        │ - Sensor Fusion       │  │ - Bridge (State Encoding)      │
                        └──────────┬────────────┘  └──────────┬─────────────────────┘
                                   │                          │
                                   └────────────┬─────────────┘
                                                ▼
                        ┌─────────────────────────────────────────┐
                        │           PHONE HARDWARE                │
                        │  ┌───────────┐  ┌───────────┐  ┌───────┐│
                        │  │ Accel/Gyro│  │ NPU Accel │  │ Light/││
                        │  │ (50Hz)    │  │ (Dim8350) │  │ Prox  ││
                        │  └───────────┘  └───────────┘  └───────┘│
                        └─────────────────────────────────────────┘

                        ┌─────────────────────────────────────────┐
                        │          REPO INFRASTRUCTURE            │
                        │  Justfile Automation  .machine_readable/  │
                        │  Rust / JNI / Kotlin  RSR Bronze (Cert)   │
                        └─────────────────────────────────────────┘
```

## Completion Dashboard

```
COMPONENT                          STATUS              NOTES
─────────────────────────────────  ──────────────────  ─────────────────────────────────
NEURAL ENGINE (RUST)
  LSM (Liquid State Machine)        ██████████ 100%    512 LIF neurons active
  ESN (Echo State Network)          ██████████ 100%    300-neuron reservoir stable
  Sensor Fusion (Accel/Gyro)        ██████████ 100%    50Hz loop verified
  Bridge (Neural ↔ Symbolic)        ██████████ 100%    Context generation verified

INFERENCE & UI
  Local LLM (Llama 3.2)             ████████░░  80%    Q4 quantization optimized
  Claude Client (Fallback)          ██████████ 100%    Retry logic stable
  Android App (Kotlin/Compose)      ████████░░  80%    UI components stable
  JNI / Native Bridge               ██████████ 100%    Kotlin ↔ Rust verified

REPO INFRASTRUCTURE
  Justfile Automation               ██████████ 100%    Standard build/JNI tasks
  .machine_readable/                ██████████ 100%    STATE tracking active
  Performance Benchmarks            ██████████ 100%    Neural steps < 3ms

─────────────────────────────────────────────────────────────────────────────
OVERALL:                            █████████░  ~90%   On-device AI stable
```

## Key Dependencies

```
Sensors ────────► LSM Reservoir ──────► Bridge Context ──────► Local LLM
     │                 │                   │                      │
     ▼                 ▼                   ▼                      ▼
Accel/Gyro ─────► ESN Predict ──────► Query Routing ──────► Claude API
```

## Update Protocol

This file is maintained by both humans and AI agents. When updating:

1. **After completing a component**: Change its bar and percentage
2. **After adding a component**: Add a new row in the appropriate section
3. **After architectural changes**: Update the ASCII diagram
4. **Date**: Update the `Last updated` comment at the top of this file

Progress bars use: `█` (filled) and `░` (empty), 10 characters wide.
Percentages: 0%, 10%, 20%, ... 100% (in 10% increments).
