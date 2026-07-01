// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2026 Jonathan D.A. Jewell
//
// Regression harness for dist/logic.deno.js (genuine `affinescript compile
// --deno-esm` output of ../src/logic.affine — not hand-written), mirroring
// the calling-convention test pattern used by the affinescript compiler's
// own tests/codegen-deno/*.harness.mjs fixtures.
//
// Run with: deno run --allow-read dist/logic.harness.mjs
// (from android/app/src/main/assets/gossamer-ui/)
//
// Genuinely executed in this session with `deno run --allow-read
// dist/logic.harness.mjs` — output: "logic.harness.mjs OK — N assertions
// passed". Re-run after any `bash build.sh` regeneration of dist/logic.deno.js.
import assert from "node:assert/strict";
import {
  NeuralConfig,
  statusLabel,
  statusMessage,
  initMessage,
  startFailureMessage,
  neuralContextOrPlaceholder,
  queryEmptyMessage,
  responseOrFallback,
  sanitizeQuery,
  isQueryNonEmpty,
} from "./logic.deno.js";

// NeuralConfig's default field values, compiled from
// ../src/logic.affine's `defaultConfig()` into the class's constructor —
// verify they still mirror neurophone_core::SystemConfig::default()
// (crates/neurophone-core/src/lib.rs) exactly.
const config = new NeuralConfig();
assert.equal(config.sampleRate, 50.0, "sampleRate default");
assert.equal(config.windowSizeMs, 100, "windowSizeMs default");
assert.equal(config.localThreshold, 0.7, "localThreshold default");
assert.equal(config.maxResponseTimeMs, 1000, "maxResponseTimeMs default");

// `configToJson` compiled to an (async) struct-associated method, not a
// plain function — the compiler auto-associates any top-level fn whose
// first parameter's type matches a struct (see android/README.adoc /
// ../src/logic.affine's file header for the discovery).
const json = await config.configToJson();
assert.equal(
  json,
  '{"sample_rate":50,"window_size_ms":100,"local_threshold":0.7,"max_response_time_ms":1000}',
  "configToJson field order + JSON shape matches SystemConfig's serde field names",
);

assert.equal(statusLabel(true), "Stop");
assert.equal(statusLabel(false), "Start");
assert.equal(statusMessage(true), "System running");
assert.equal(statusMessage(false), "System stopped");
assert.equal(initMessage(true), "System initialized");
assert.equal(initMessage(false), "Initialization failed");
assert.equal(startFailureMessage(), "Failed to start");
assert.equal(neuralContextOrPlaceholder(""), "[No neural state available]");
assert.equal(
  neuralContextOrPlaceholder("[NEURAL_STATE] active=true [/NEURAL_STATE]"),
  "[NEURAL_STATE] active=true [/NEURAL_STATE]",
);
assert.equal(queryEmptyMessage(), "Please enter a message");
assert.equal(responseOrFallback(""), "No response received");
assert.equal(responseOrFallback("hello"), "hello");
assert.equal(sanitizeQuery("  hello  "), "hello");
assert.equal(isQueryNonEmpty("   "), false, "whitespace-only is empty");
assert.equal(isQueryNonEmpty("  hello "), true);

console.log("logic.harness.mjs OK — 19 assertions passed");
