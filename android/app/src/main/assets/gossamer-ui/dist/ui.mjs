// SPDX-License-Identifier: MPL-2.0
//
// GENERATED-OUTPUT STUB — hand-written stand-in for the Deno-ESM that the
// AffineScript compiler will emit from src/ui.affine + src/bridge.affine.
//
// Why this exists: the AffineScript toolchain (OCaml/Dune `affinescript`
// compiler, see README.adoc "Build wiring") is not yet vendored in this repo or
// CI. Committing a faithful stub keeps the Gossamer webview loadable and
// reviewable today. `build.sh` overwrites this file with real compiled output
// once the toolchain is available.
//
// TODO(#83 rebase): delete this stub once `deno task build:ui` (or CI) produces
//   dist/ui.mjs from the .affine sources. Keep behaviour in sync with ui.affine
//   until then.
//
// This stub mirrors src/ui.affine 1:1 and the bridge contract in
// src/bridge.affine. It assumes the Gossamer host injects `globalThis.gossamer`
// with `invoke(cmd, args) => Promise<string>`.

const host = globalThis.gossamer ?? {
  // Fallback shim so the page is inspectable in a plain browser during dev.
  // TODO(#83): remove once the real host is always present.
  invoke: async (cmd) => {
    console.warn(`[gossamer-ui] no host bridge; stub invoke('${cmd}')`);
    return cmd === "get_neural_context" ? "[stub] no native bridge" : "";
  },
};

// --- bridge.affine port ---------------------------------------------------
const defaultConfig = {
  loop_interval_ms: 20,
  debug: false,
  sensor: { sample_rate_hz: 50.0, buffer_size: 100, output_dim: 32 },
  lsm: { dimensions: [8, 8, 8], spectral_radius: 0.9 },
  esn: { reservoir_size: 300, spectral_radius: 0.95 },
};

const bridge = {
  init: async (config) => (await host.invoke("init", { configJson: JSON.stringify(config) })) === "true",
  start: async () => (await host.invoke("start", {})) === "true",
  stop: async () => { await host.invoke("stop", {}); },
  query: (message, preferLocal) => host.invoke("query", { message, preferLocal }),
  queryLocal: (message) => host.invoke("query_local", { message }),
  queryClaude: (message) => host.invoke("query_claude", { message }),
  getNeuralContext: () => host.invoke("get_neural_context", {}),
  getState: () => host.invoke("get_state", {}),
};

// --- ui.affine port -------------------------------------------------------
const $ = (id) => document.getElementById(id);
const el = {
  statusText: $("statusText"),
  neuralContextText: $("neuralContextText"),
  inputField: $("inputField"),
  responseText: $("responseText"),
  startButton: $("startButton"),
  sendButton: $("sendButton"),
  localSwitch: $("localSwitch"),
  activityIndicator: $("activityIndicator"),
};

let isSystemRunning = false;
let contextTimer = null;

function updateUI() {
  el.startButton.textContent = isSystemRunning ? "Stop" : "Start";
  el.sendButton.disabled = !isSystemRunning;
  el.inputField.disabled = !isSystemRunning;
}

async function pollContextOnce() {
  if (!isSystemRunning) return;
  try {
    el.neuralContextText.textContent = await bridge.getNeuralContext();
  } catch (_e) {
    // non-fatal; keep polling
  }
  contextTimer = setTimeout(pollContextOnce, 500);
}

function startContextUpdates() { contextTimer = setTimeout(pollContextOnce, 500); }
function stopContextUpdates() { if (contextTimer != null) { clearTimeout(contextTimer); contextTimer = null; } }

async function startSystem() {
  if (await bridge.start()) {
    isSystemRunning = true;
    startContextUpdates();
    updateUI();
    el.statusText.textContent = "System running";
  } else {
    el.statusText.textContent = "Failed to start";
  }
}

async function stopSystem() {
  await bridge.stop();
  isSystemRunning = false;
  stopContextUpdates();
  updateUI();
  el.statusText.textContent = "System stopped";
}

const toggleSystem = () => (isSystemRunning ? stopSystem() : startSystem());

async function sendQuery() {
  const message = el.inputField.value.trim();
  if (message === "") { el.responseText.textContent = "Please enter a message"; return; }
  el.activityIndicator.hidden = false;
  el.sendButton.disabled = true;
  try {
    const response = await bridge.query(message, el.localSwitch.checked);
    el.responseText.textContent = response === "" ? "No response received" : response;
  } catch (e) {
    el.responseText.textContent = "Error: " + e.message;
  } finally {
    el.activityIndicator.hidden = true;
    el.sendButton.disabled = false;
  }
}

async function initializeSystem() {
  el.statusText.textContent = "Initializing...";
  try {
    const ok = await bridge.init(defaultConfig);
    el.statusText.textContent = ok ? "System initialized" : "Initialization failed";
  } catch (e) {
    el.statusText.textContent = "Error: " + e.message;
  }
}

async function main() {
  el.startButton.addEventListener("click", toggleSystem);
  el.sendButton.addEventListener("click", sendQuery);
  updateUI();
  await initializeSystem();
}

main();
