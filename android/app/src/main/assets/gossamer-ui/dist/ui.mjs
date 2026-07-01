// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2026 Jonathan D.A. Jewell
//
// Hand-written DOM + NeurophoneBridge harness for NeurophoneActivity's
// webview. Loaded by ../index.html as a module script.
//
// This is deliberately NOT compiled from AffineScript. Everything that
// benefits from being typed and verified — status-label selection, message
// formatting, config-to-JSON encoding, response fallbacks — lives in
// ../src/logic.affine and is imported below from the genuinely compiled
// ./logic.deno.js (`affinescript compile --deno-esm`, committed, see
// android/README.adoc "AffineScript verification"). What's left here is
// exactly the DOM event wiring and the `window.NeurophoneBridge.*` calls,
// neither of which this compiler can express yet:
//   * DOM bindings: `stdlib/Canvas.affine` (hyperpolymath/affinescript)
//     documents that even Canvas2D crosses the boundary as opaque `Json`
//     today, pending `affinescript-dom` runtime support (blocked on that
//     project's issue #255) — there is no `getElementById`/
//     `addEventListener` typed surface to target.
//   * `window.NeurophoneBridge.query(...)`-shaped calls: the deno-esm
//     backend's `extern fn` lowering (`lib/codegen_deno.ml`) only supports
//     a compiler-maintained intrinsic table or a bare same-named global
//     *function* call — it cannot express an arbitrary caller-supplied
//     `obj.method(args)`, which is what the Android JS-interface object is.
//
// This mirrors the shape of this compiler's own `tests/codegen-deno/
// *.harness.mjs` test convention (hand-written JS harness imports compiled
// pure functions and drives them) — not a new pattern invented for this PR.
//
// Every `NeurophoneBridge.*` call is synchronous (Android's
// addJavascriptInterface semantics — see NeurophoneBridge.java); there is
// no gossamer-style async `Promise`/postMessage round-trip to wait on here.

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

const host = globalThis.NeurophoneBridge ?? {
  // Fallback shim so the page is inspectable in a plain desktop browser
  // during UI development, where no NeurophoneBridge JS-interface exists.
  init: () => { console.warn("[neurophone-ui] no NeurophoneBridge host; stub init()"); return false; },
  start: () => false,
  stop: () => {},
  isRunning: () => false,
  query: () => null,
  getNeuralContext: () => null,
  reset: () => {},
};

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

function updateUiEnabled() {
  el.startButton.textContent = statusLabel(isSystemRunning);
  el.sendButton.disabled = !isSystemRunning;
  el.inputField.disabled = !isSystemRunning;
}

function pollContextOnce() {
  if (!isSystemRunning) return;
  try {
    const ctx = host.getNeuralContext();
    el.neuralContextText.textContent = neuralContextOrPlaceholder(ctx ?? "");
  } catch (_e) {
    // non-fatal; keep polling
  }
  contextTimer = setTimeout(pollContextOnce, 500);
}

function startContextUpdates() { contextTimer = setTimeout(pollContextOnce, 500); }
function stopContextUpdates() { if (contextTimer != null) { clearTimeout(contextTimer); contextTimer = null; } }

function startSystem() {
  if (host.start()) {
    isSystemRunning = true;
    startContextUpdates();
    updateUiEnabled();
    el.statusText.textContent = statusMessage(true);
  } else {
    el.statusText.textContent = startFailureMessage();
  }
}

function stopSystem() {
  host.stop();
  isSystemRunning = false;
  stopContextUpdates();
  updateUiEnabled();
  el.statusText.textContent = statusMessage(false);
}

const toggleSystem = () => (isSystemRunning ? stopSystem() : startSystem());

function sendQuery() {
  const message = sanitizeQuery(el.inputField.value);
  if (!isQueryNonEmpty(el.inputField.value)) {
    el.responseText.textContent = queryEmptyMessage();
    return;
  }
  el.activityIndicator.hidden = false;
  el.sendButton.disabled = true;
  try {
    const response = host.query(message, el.localSwitch.checked);
    el.responseText.textContent = responseOrFallback(response ?? "");
  } catch (e) {
    el.responseText.textContent = "Error: " + e.message;
  } finally {
    el.activityIndicator.hidden = true;
    el.sendButton.disabled = false;
  }
}

function initializeSystem() {
  el.statusText.textContent = "Initializing...";
  try {
    const config = new NeuralConfig();
    (async () => {
      const configJson = await config.configToJson();
      const ok = host.init(configJson);
      el.statusText.textContent = initMessage(ok);
    })();
  } catch (e) {
    el.statusText.textContent = "Error: " + e.message;
  }
}

function main() {
  el.startButton.addEventListener("click", toggleSystem);
  el.sendButton.addEventListener("click", sendQuery);
  updateUiEnabled();
  initializeSystem();
}

main();
