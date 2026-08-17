/* EL4090 Occupied-Body Support Envelope — interactive instrument.
 *
 * Mode A (pose -> envelope): 18 joint sliders drive the occupied support
 *   envelope P_K(q); the plan view colors every polygon face by the capsule
 *   that binds it, cross-referenced to the h_k bar strip below.
 *
 * Mode B (envelope -> rejection): the five envelope sliders build the allowed
 *   hexagon V; the server computes the per-joint rejected sub-intervals at a
 *   feasible reference and returns magenta rejected arcs + amber accessible
 *   HAA arcs, which the stage draws as stylized foot-direction bands.
 *
 * Revision 1 additions:
 *   - leading+trailing throttle so sliders update live during a drag
 *   - full 3D capsule proxies (radius) + the joint skeleton in BOTH views
 *   - honest reference reporting (amber banner when the requested custom
 *     reference is infeasible and a fallback is used)
 *   - a top-down <-> 3D view toggle with a custom offline 3D renderer
 *     (drag-to-rotate yaw/pitch, wheel zoom, depth-sorted, no external libs)
 *   - rejection-band animation: a marker sweeps each rejected interval along
 *     the TRUE FK foot trajectory with the other joints pinned, the swept leg
 *     animating in the skeleton (respects prefers-reduced-motion)
 *
 * All math lives in envelope_server.py (importing the read-only model); the
 * browser only renders JSON.
 */
(() => {
"use strict";

/* ------------------------------------------------------------- constants */
const LEGS = ["LB", "LF", "LM", "RB", "RF", "RM"];
const JOINTS = ["HAA", "HFE", "KFE"];
const JOINT_NAMES = LEGS.flatMap(leg => JOINTS.map(jt => leg + "_" + jt));

const LEG_COLOR = {
  LB: "#5B9BD5", LF: "#F0A03A", LM: "#6FBF6F",
  RB: "#E06C6C", RF: "#B58AE6", RM: "#D08B70",
};
const BASE_COLOR = "#9AA5AB";
const MODEL = {
  occ: "#22B8AF", reach: "#3BC6E6", rej: "#E85B9E",
  acc: "#E8B04B", bone: "#E7E2D2", brass: "#CF9B2F",
};
const RESTING = new Array(18).fill(0).map((_, i) => [0, 0.6, -0.6][i % 3]);

const THROTTLE_MS = 70;          // leading + trailing throttle window
const SWEEP_DURATION = 2.6;      // seconds per rejected-interval sweep
const FOCAL_3D = 4.0;            // perspective factor for the 3D view
const CAMERA_DEFAULT_YAW = 0.62;   // "grab the object" drag: right-drag swings +x toward screen-right
// Default pitch raised to 0.90 (was 0.48). With the corrected vertical
// (world +z -> screen-up), the near-side legs are sheared toward the camera,
// so at low pitch their feet still project ABOVE the torso ("not standing").
// pitch >= 0.78 is required for every foot to project below the base; 0.90
// gives margin. Down-drag lowers the camera (pitch decreases) and tips the
// model's top toward the viewer.
const CAMERA_DEFAULT_PITCH = 0.90;
const CAMERA_DEFAULT_ZOOM = 1.0;
const PAN_GAIN = 0.35;           // calibrated screen pixels -> modest world-space pan

function pose(o) {
  const q = new Array(18).fill(0);
  for (const leg of LEGS) {
    for (const jt of JOINTS) {
      const jn = leg + "_" + jt;
      const idx = JOINT_NAMES.indexOf(jn);
      if (o[jn] !== undefined) q[idx] = o[jn];
      else if (o[jt] !== undefined) q[idx] = o[jt];
    }
  }
  return q;
}
const PRESETS = {
  spider: { q: pose({ HFE: 0.6, KFE: -0.6 }), margin: 0.0 },
  mammal: {
    q: pose({ RF_HAA: -1.308, RM_HAA: 1.308, RB_HAA: 1.308,
              LF_HAA: -1.308, LM_HAA: 1.308, LB_HAA: 1.308,
              HFE: 1.0, KFE: -0.608 }),
    margin: 0.045,
  },
  "wide-low": {
    q: pose({ RF_HAA: -0.9, RM_HAA: 0.9, RB_HAA: 0.9,
              LF_HAA: -0.9, LM_HAA: 0.9, LB_HAA: 0.9,
              HFE: 1.2, KFE: -1.2 }),
    margin: 0.185,
  },
  tuck: { q: pose({ HFE: 2.2, KFE: -2.2 }), margin: 0.0 },
  zero: { q: pose({}), margin: 0.0 },
};

/* ---------------------------------------------------------------- state */
const state = {
  mode: "A",
  q: RESTING.slice(),
  capsule: "full",
  aMargin: 0.0, aK: 32, aLegacy: true, aReach: true,
  env: { w_f: 0.70, w_m: 0.70, w_b: 0.70, x_f: 0.75, x_b: -0.72 },
  bMargin: 0.02, bK: 32,
  bReference: "resting", bRejected: true, bAccessible: true, bAnimate: true,
  // Mode C HFE/KFE pins come from the JOINT POSE rack (state.q[3*i+1] / [3*i+2]),
  // so the pin sliders live in the rack, not in the mode panel.
  view: "topdown",                       // "topdown" | "3d"
  computeMode: "release",                // "live" | "release"
  planCamera: { panX: 0, panY: 0, zoom: 1 },
  // The 3D camera looks at a USER target in ground-relative viewer coordinates.
  // The scene-fit scale is auto-derived from the bounding box each render, but
  // the target is persistent (pan/rotate/recenter never reset it to a centroid).
  camera: {
    yaw: CAMERA_DEFAULT_YAW, pitch: CAMERA_DEFAULT_PITCH, zoom: CAMERA_DEFAULT_ZOOM,
    target: { x: 0, y: 0, z: 0 }, scale: 300,
  },
  camBounds: { cx: 0, cy: 0, cz: 0, scale: 300 },
  dragging: false,
  reqId: 0,
  lastData: null,
  previewData: null,                       // committed data + exact pose-only geometry
  previewing: false,
  anim: null,                            // rejection-sweep animation state
  proj: null,                            // current [x,y] projector (per render)
};

/* ------------------------------------------------------------ formatting */
function fmt(x, n = 3) {
  if (x == null || !isFinite(x)) return "∅";
  return (x < 0 ? "-" : "+") + Math.abs(x).toFixed(n);
}
function fmtN(x, n = 3) {
  if (x == null || !isFinite(x)) return "∅";
  return x.toFixed(n);
}
function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}
function capsuleColor(name) {
  const leg = String(name).split("_")[0];
  if (leg === "base") return BASE_COLOR;
  return LEG_COLOR[leg] || MODEL.bone;
}
function capColor(d, idx) {
  if (idx == null || !d.capsule_names[idx]) return MODEL.bone;
  return capsuleColor(d.capsule_names[idx]);
}
function fmtPose(q) {
  if (!q) return "—";
  const parts = [];
  for (let i = 0; i < q.length; i += 3) {
    const leg = LEGS[i / 3] || "?";
    parts.push(leg + " " + [q[i], q[i + 1], q[i + 2]].map(v => fmt(v, 2)).join(" "));
  }
  return parts.join("   ");
}
function reducedMotion() {
  return !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
}

/* Presentation is deliberately independent from model/control state. */
function preferredStyle() {
  try {
    const saved = localStorage.getItem("el4090-visual-style");
    return saved === "classic" ? "classic" : "refined";
  } catch (_err) {
    return "refined";
  }
}
function applyStyle(style, persist = true) {
  const next = style === "classic" ? "classic" : "refined";
  document.body.dataset.style = next;
  const select = document.getElementById("styleSelect");
  if (select) select.value = next;
  if (persist) {
    try { localStorage.setItem("el4090-visual-style", next); } catch (_err) {}
  }
}
function preferredComputeMode() {
  try {
    return localStorage.getItem("el4090-compute-mode") === "live" ? "live" : "release";
  } catch (_err) {
    return "release";
  }
}
function applyComputeMode(mode, persist = true) {
  const next = mode === "live" ? "live" : "release";
  state.computeMode = next;
  const live = document.getElementById("computeLive");
  const release = document.getElementById("computeRelease");
  if (live) {
    live.classList.toggle("active", next === "live");
    live.setAttribute("aria-pressed", next === "live");
  }
  if (release) {
    release.classList.toggle("active", next === "release");
    release.setAttribute("aria-pressed", next === "release");
  }
  document.body.dataset.compute = next;
  if (persist) {
    try { localStorage.setItem("el4090-compute-mode", next); } catch (_err) {}
  }
  if (next === "live" && state.previewing) {
    posePreviewEpoch += 1;
    posePreviewPending = null;
    clearSliderPreview();
    fire();
  }
}

/* ---------------------------------------------------------------- LED */
function setBusy(s) {
  const led = document.getElementById("led");
  const txt = document.getElementById("liveText");
  led.classList.remove("ready", "busy", "warn");
  if (s === "warn") { led.classList.add("warn"); txt.textContent = "ERROR"; }
  else if (s) { led.classList.add("busy"); txt.textContent = "COMPUTE"; }
  else { led.classList.add("ready"); txt.textContent = "READY"; }
}

/* --------------------------------------------------------- joint rack */
function buildJointRack() {
  const grid = document.getElementById("jointGrid");
  grid.innerHTML = "";
  for (const leg of LEGS) {
    const col = document.createElement("div");
    col.className = "joint-col";
    col.dataset.leg = leg;
    const head = document.createElement("div");
    head.className = "joint-leg";
    const chip = document.createElement("span");
    chip.className = "leg-chip";
    chip.style.background = LEG_COLOR[leg];
    const name = document.createElement("span");
    name.className = "leg-name";
    name.textContent = leg;
    head.append(chip, name);
    col.appendChild(head);
    for (const jt of JOINTS) {
      const jn = leg + "_" + jt;
      const idx = JOINT_NAMES.indexOf(jn);
      const row = document.createElement("div");
      row.className = "joint-row";
      row.dataset.joint = jt;   // "HAA" | "HFE" | "KFE" — lets Mode C lock HAA as swept
      const lab = document.createElement("div");
      lab.className = "jt";
      const l = document.createElement("span"); l.textContent = jt;
      const v = document.createElement("span"); v.className = "jt-val"; v.id = "val_" + jn;
      lab.append(l, v);
      const inp = document.createElement("input");
      inp.type = "range"; inp.min = -3; inp.max = 3; inp.step = 0.01;
      inp.value = state.q[idx];
      inp.id = "sl_" + jn;
      inp.setAttribute("aria-label", jn + " joint angle");
      inp.addEventListener("input", () => {
        state.q[idx] = parseFloat(inp.value);
        v.textContent = fmt(parseFloat(inp.value), 2);
        inp.dataset.commitDirty = "1";
        handleSliderInput("pose");
      });
      bindSliderCommit(inp);
      row.append(lab, inp);
      col.appendChild(row);
    }
    grid.appendChild(col);
  }
}
function syncJointRack() {
  JOINT_NAMES.forEach((jn, idx) => {
    const v = document.getElementById("val_" + jn);
    if (v) v.textContent = fmt(state.q[idx], 2);
  });
}
function updateJointRackState() {
  const note = document.getElementById("jointModeNote");
  const modeBresting = state.mode === "B" && state.bReference === "resting";
  const modeC = state.mode === "C";
  if (state.mode === "A") note.textContent = "q[18] · drives the envelope";
  else if (modeC) note.textContent = "HFE/KFE = rejection pins · HAA = check value";
  else if (modeBresting) note.textContent = "q[18] · reference locked to resting pose";
  else note.textContent = "q[18] · custom reference pose";
  document.querySelectorAll(".joint-col").forEach(c => {
    const head = c.querySelector(".joint-leg");
    const old = head && head.querySelector(".lock-tag");
    if (old) old.remove();
    c.classList.toggle("disabled", modeBresting);
    if (modeBresting) {
      // Mode B resting: the whole pose is the fixed reference.
      c.querySelectorAll("input").forEach(i => { i.disabled = true; });
      if (head) {
        const t = document.createElement("span");
        t.className = "lock-tag";
        t.textContent = "LOCKED";
        head.appendChild(t);
      }
    } else {
      // Mode A (all joints drive the envelope) and Mode C (HFE/KFE are the pins,
      // HAA is a check value) both expose all 18 sliders.
      c.querySelectorAll("input").forEach(i => { i.disabled = false; });
    }
  });
  syncJointRack();   // show real values (Mode C HAA rows are check values, not SWEPT)
}

/* ------------------------------------------------------------ payload */
function payload() {
  if (state.mode === "A") {
    return {
      q: state.q.slice(),
      margin: state.aMargin, K: state.aK,
      capsule_set: state.capsule,
      show_reachable: state.aReach, show_legacy: state.aLegacy,
    };
  }
  if (state.mode === "C") {
    // HFE/KFE pins + the HAA check value come from the JOINT POSE rack:
    // per leg i, HAA = q[3i], HFE = q[3i+1], KFE = q[3i+2].
    const hfe = LEGS.map((_, i) => state.q[3 * i + 1]);
    const kfe = LEGS.map((_, i) => state.q[3 * i + 2]);
    const haa = LEGS.map((_, i) => state.q[3 * i]);
    return {
      envelope: { ...state.env },
      margin: state.bMargin, K: state.bK,
      capsule_set: state.capsule,
      hfe, kfe, haa,
      steps: 201, tolerance: 1e-6,
      show_rejected: state.bRejected, show_accessible: state.bAccessible,
    };
  }
  return {
    envelope: { ...state.env },
    margin: state.bMargin, K: state.bK,
    capsule_set: state.capsule,
    reference: state.bReference,
    reference_q: state.bReference === "custom" ? state.q.slice() : null,
    steps: 101, tolerance: 1e-6,
    show_rejected: state.bRejected, show_accessible: state.bAccessible,
  };
}
function endpoint() {
  return state.mode === "A" ? "/api/envelope"
    : state.mode === "B" ? "/api/rejection"
    : "/api/haa_rejection";
}

/* Leading-edge + trailing throttle: a burst of input events fires immediately
 * on the first event, then at most every THROTTLE_MS while the burst continues,
 * plus one final trailing fire once the burst ends (on release). */
let lastFire = 0;
let trailingTimer = null;
let lastRenderedReq = 0;      // reqId of the newest response already rendered
function schedule() {
  const now = performance.now();
  const remaining = lastFire + THROTTLE_MS - now;
  if (remaining <= 0) {
    lastFire = now;
    if (trailingTimer) { clearTimeout(trailingTimer); trailingTimer = null; }
    fire();
  } else {
    if (trailingTimer) clearTimeout(trailingTimer);
    trailingTimer = setTimeout(() => {
      lastFire = performance.now();
      trailingTimer = null;
      fire();
    }, remaining);
  }
}
function flushLiveCompute() {
  if (!trailingTimer) return;
  clearTimeout(trailingTimer);
  trailingTimer = null;
  lastFire = performance.now();
  fire();
}
// Full recomputes are single-flight and coalesced. While one request runs, a
// slider burst replaces one pending job with its newest payload. This keeps Live
// mode responsive without accumulating stale rejection calculations.
let firePending = null;
let fireRunning = false;
let fireDrainPromise = Promise.resolve();
function fire() {
  const id = ++state.reqId;
  firePending = { id, payload: payload(), route: endpoint() };
  if (!fireRunning) fireDrainPromise = drainFire();
  return fireDrainPromise;
}
async function drainFire() {
  fireRunning = true;
  try {
    while (firePending) {
      const job = firePending;
      firePending = null;
      await doFetch(job.id, job.payload, job.route);
    }
  } finally {
    fireRunning = false;
    if (firePending) fireDrainPromise = drainFire();
  }
}
async function doFetch(id, p, route) {
  setBusy(true);
  try {
    const res = await fetch(route, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(p),
    });
    const data = await res.json();
    // Render any response that is newer than what is already on screen. A plain
    // `id === state.reqId` guard would DISCARD a valid response whenever a newer
    // request is in flight (e.g. the leading edge of a fast slider burst) and
    // that newer request's response is slow/lost -- leaving the stage stale.
    // This is safe: responses always arrive in request order, and an older
    // response can never overwrite a newer rendered state.
    if (id <= lastRenderedReq) return;
    if (data && data.error) {
      setBusy("warn");
      showError("server: " + data.error);
      lastRenderedReq = id;
      return;
    }
    lastRenderedReq = id;
    render(data);
    setBusy(false);
  } catch (err) {
    if (id > lastRenderedReq) { setBusy("warn"); showError("network: " + err); }
  }
}

/* Pose previews are single-flight and coalesced: repeated inputs replace one
 * pending payload rather than building an FK request backlog. */
let posePreviewPending = null;
let posePreviewRunning = false;
let posePreviewEpoch = 0;
const POSE_GEOMETRY_KEYS = [
  "q", "capsule_set", "n_capsules", "capsule_names", "capsule_links",
  "capsule_radius", "capsules_3d", "skeleton", "feet",
];

function stageModeText() {
  return { A: "POSE → ENVELOPE", B: "ENVELOPE → REJECTION", C: "HAA REJECTION" }[state.mode];
}
function refreshStageModeLabel() {
  const label = document.getElementById("stageModeLabel");
  if (!label) return;
  const stage = document.getElementById("stage");
  const suffix = state.previewing
    ? (stage.dataset.previewKind === "pose" ? " · POSE PREVIEW" : " · PENDING")
    : "";
  label.textContent = stageModeText() + suffix;
}
function beginSliderPreview(kind) {
  state.previewing = true;
  const stage = document.getElementById("stage");
  stage.classList.add("pose-preview");
  stage.dataset.previewKind = kind;
  refreshStageModeLabel();
  stopAnimation();
}
function clearSliderPreview() {
  state.previewing = false;
  state.previewData = null;
  const stage = document.getElementById("stage");
  stage.classList.remove("pose-preview");
  delete stage.dataset.previewKind;
  refreshStageModeLabel();
}
function schedulePosePreview() {
  posePreviewPending = {
    epoch: posePreviewEpoch,
    payload: { q: state.q.slice(), capsule_set: state.capsule },
  };
  if (!posePreviewRunning) drainPosePreview();
}
function handleSliderInput(kind) {
  if (state.computeMode === "live") {
    clearSliderPreview();
    schedule();
    return;
  }
  beginSliderPreview(kind);
  if (kind === "pose") schedulePosePreview();
}
async function drainPosePreview() {
  posePreviewRunning = true;
  try {
    while (posePreviewPending) {
      const job = posePreviewPending;
      posePreviewPending = null;
      try {
        const res = await fetch("/api/pose", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(job.payload),
        });
        const data = await res.json();
        if (job.epoch !== posePreviewEpoch || !state.previewing || data.error || !state.lastData) continue;
        const merged = { ...state.lastData };
        POSE_GEOMETRY_KEYS.forEach(key => { if (data[key] !== undefined) merged[key] = data[key]; });
        state.previewData = merged;
        redrawStage();
      } catch (_err) {
        // The final committed request remains authoritative if preview fails.
      }
    }
  } finally {
    posePreviewRunning = false;
    if (posePreviewPending) drainPosePreview();
  }
}

let sliderCommitQueued = false;
function commitSliderInteraction() {
  if (sliderCommitQueued) return;
  sliderCommitQueued = true;
  queueMicrotask(() => {
    sliderCommitQueued = false;
    if (state.computeMode === "live") {
      flushLiveCompute();
      return;
    }
    posePreviewEpoch += 1;
    posePreviewPending = null;
    // Keep the latest exact pose visible and marked stale until the final
    // envelope/rejection response arrives and clears preview state.
    fire();
  });
}
function bindSliderCommit(el) {
  const commit = () => {
    if (el.dataset.commitDirty !== "1") return;
    el.dataset.commitDirty = "0";
    commitSliderInteraction();
  };
  el.addEventListener("change", commit);
  el.addEventListener("pointerup", commit);
  el.addEventListener("pointercancel", commit);
  el.addEventListener("blur", commit);
  el.addEventListener("keyup", e => {
    if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown"].includes(e.key)) commit();
  });
}

function showError(msg) {
  const banner = document.getElementById("stageBanner");
  banner.textContent = msg;
  banner.classList.remove("hidden", "amber");
  const ro = document.getElementById("readout");
  ro.innerHTML = `<div class="rej-wrap"><div class="rej-summary"><span class="pill">ERROR</span><span>${esc(msg)}</span></div></div>`;
}

/* ------------------------------------------------------- SVG helpers */
const NS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs, parent) {
  const el = document.createElementNS(NS, tag);
  for (const k in attrs) {
    if (attrs[k] != null) el.setAttribute(k, attrs[k]);
  }
  if (parent) parent.appendChild(el);
  return el;
}
function polyPoints(pts) {
  return pts.map(p => p[0].toFixed(4) + "," + p[1].toFixed(4)).join(" ");
}
function drawPoly(svg, pts, attrs) {
  return svgEl("polygon", Object.assign({ points: polyPoints(pts) }, attrs), svg);
}
function polyline(svg, pts, attrs) {
  return svgEl("polyline", Object.assign({ points: polyPoints(pts) }, attrs), svg);
}
function line(svg, a, b, attrs) {
  return svgEl("line", Object.assign({
    x1: a[0], y1: a[1], x2: b[0], y2: b[1],
  }, attrs), svg);
}
function dot(svg, p, attrs) {
  return svgEl("circle", Object.assign({ cx: p[0], cy: p[1], r: 6 }, attrs), svg);
}

function niceStep(range) {
  const raw = Math.pow(10, Math.floor(Math.log10(Math.max(range, 1e-6))));
  for (const m of [1, 2, 5, 10]) {
    const s = raw * m;
    if (range / s <= 8) return s;
  }
  return raw * 10;
}
function computeTransform(bounds) {
  const pad = Math.max(
    (bounds.x1 - bounds.x0) * 0.06,
    (bounds.y1 - bounds.y0) * 0.06,
    0.15,
  );
  const x0 = bounds.x0 - pad, y0 = bounds.y0 - pad;
  const x1 = bounds.x1 + pad, y1 = bounds.y1 + pad;
  const plan = state.planCamera;
  const S = (940 / Math.max(x1 - x0, y1 - y0, 1e-6)) * plan.zoom;
  return {
    S,
    ox: 500 - S * (x0 + x1) / 2 + plan.panX,
    oy: 500 + S * (y0 + y1) / 2 + plan.panY,
  };
}
function toV(p, T) {
  return [T.ox + T.S * p[0], T.oy - T.S * p[1]];
}
function X(arr, T) {
  return arr.map(p => [T.ox + T.S * p[0], T.oy - T.S * p[1]]);
}

function boundsOf(pts) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const p of pts) {
    if (!Array.isArray(p) || p.length < 2) continue;
    if (!isFinite(p[0]) || !isFinite(p[1])) continue;
    x0 = Math.min(x0, p[0]); y0 = Math.min(y0, p[1]);
    x1 = Math.max(x1, p[0]); y1 = Math.max(y1, p[1]);
  }
  if (!isFinite(x0)) return { x0: -1, y0: -1, x1: 1, y1: 1 };
  if (x1 - x0 < 1e-6) { x0 -= 0.1; x1 += 0.1; }
  if (y1 - y0 < 1e-6) { y0 -= 0.1; y1 += 0.1; }
  return { x0, y0, x1, y1 };
}

/* 3D projection ---------------------------------------------------------
 * Custom offline renderer: yaw about the world-z (body-up) axis, then pitch
 * about the rotated x-axis, a subtle perspective divide, and an orthographic
 * scale. Returns [sx, sy, depth]; depth is camera-space z for painter sort. */
function boundsOf3(pts) {
  let x0 = Infinity, y0 = Infinity, z0 = Infinity;
  let x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
  for (const p of pts) {
    if (!Array.isArray(p) || p.length < 3) continue;
    if (!isFinite(p[0]) || !isFinite(p[1]) || !isFinite(p[2])) continue;
    x0 = Math.min(x0, p[0]); y0 = Math.min(y0, p[1]); z0 = Math.min(z0, p[2]);
    x1 = Math.max(x1, p[0]); y1 = Math.max(y1, p[1]); z1 = Math.max(z1, p[2]);
  }
  if (!isFinite(x0)) return { cx: 0, cy: 0, cz: 0, scale: 300 };
  if (x1 - x0 < 1e-6) { x0 -= 0.1; x1 += 0.1; }
  if (y1 - y0 < 1e-6) { y0 -= 0.1; y1 += 0.1; }
  if (z1 - z0 < 1e-6) { z0 -= 0.1; z1 += 0.1; }
  const pad = Math.max(x1 - x0, y1 - y0, z1 - z0) * 0.06;
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, cz = (z0 + z1) / 2;
  const extent = Math.max(x1 - x0, y1 - y0, z1 - z0) + 2 * pad;
  return { cx, cy, cz, scale: 430 / extent };
}
function collect3DBounds(d) {
  const pts = [];
  const add3 = p => { if (Array.isArray(p) && p.length >= 3) pts.push(p); };
  const add2 = p => { if (Array.isArray(p) && p.length >= 2) pts.push([p[0], p[1], 0]); };
  const addPts = arr => { if (arr) arr.forEach(p => add2(p)); };
  if (d.capsules_3d) d.capsules_3d.forEach(seg => { add3(seg[0]); add3(seg[1]); });
  if (d.skeleton) {
    add3(d.skeleton.base);
    d.skeleton.legs.forEach(l => {
      add3(l.haa); add3(l.hfe); add3(l.kfe); add3(l.foot);
    });
  }
  if (state.mode === "A") {
    addPts(d.polygon); addPts(d.margin_polygon); addPts(d.reachable_polygon);
  } else {
    addPts(d.allowed_polygon); addPts(d.envelope_polygon);
  }
  if (d.magenta_arcs) d.magenta_arcs.forEach(a => { addPts(a.points); addPts(a.trajectory); });
  if (d.amber_arcs) d.amber_arcs.forEach(a => addPts(a.points));
  if (d.haa_arc_geometry && d.haa_arc_geometry.origins) d.haa_arc_geometry.origins.forEach(add3);
  if (d.haa_arc_geometry && d.haa_arc_geometry.markers) d.haa_arc_geometry.markers.forEach(add3);
  if (d.feet) d.feet.forEach(add3);
  return boundsOf3(pts);
}
function makeProject3D(b, groundOriginZ = 0) {
  const c = state.camera;
  const t = c.target || { x: 0, y: 0, z: 0 };
  return p => {
    // Viewer coordinates use the ground plane as z=0. The model remains in
    // its source frame; this translation exists only in the projection.
    const x = p[0] - t.x;
    const y = p[1] - t.y;
    const z = ((p[2] || 0) - groundOriginZ) - t.z;
    const cy = Math.cos(c.yaw), sy = Math.sin(c.yaw);
    const cp = Math.cos(c.pitch), sp = Math.sin(c.pitch);
    const x1 = cy * x - sy * y, y1 = sy * x + cy * y, z1 = z;
    const xc = x1, yc = cp * y1 + sp * z1, zc = sp * y1 + cp * z1;
    const persp = FOCAL_3D / (FOCAL_3D + zc * 0.9);
    // +500 centers the projection target in the 1000x1000 SVG viewBox so the
    // camera target (default the ground-plane origin) lands at stage centre.
    // [2] stays the camera-space depth for painter sort (no offset).
    return [500 + xc * b.scale * c.zoom * persp, 500 + (-yc) * b.scale * c.zoom * persp, zc];
  };
}
/* Pan the camera TARGET along the ground plane (z=0) by a screen-space delta.
 * Inverts the current yaw/pitch rotation so the scene follows the drag
 * ("grab the scene"). Returns the applied world-space (dx, dy). */
function panByScreenDelta(sdx, sdy, startTarget = null) {
  const c = state.camera;
  const t = c.target || (c.target = { x: 0, y: 0, z: 0 });
  const cp = Math.cos(c.pitch), sp = Math.sin(c.pitch);
  const cy = Math.cos(c.yaw), sy = Math.sin(c.yaw);
  const s = (c.scale || 300) * c.zoom;
  if (!isFinite(s) || s <= 0) return;
  // screen -> (yaw-rotated) -> world. persp at the target plane (z=0) is 1.
  const d1x = (sdx * PAN_GAIN) / s;
  const d1y = sp !== 1 ? (-(sdy * PAN_GAIN) / s) / cp : 0;
  const dwx = cy * d1x + sy * d1y;
  const dwy = -sy * d1x + cy * d1y;
  // Pointermove supplies total displacement since pointerdown, so derive from
  // that gesture's fixed starting target. This avoids compounding the total
  // delta on every event. Direct callers may still request an incremental pan.
  if (startTarget) {
    t.x = startTarget.x + dwx;
    t.y = startTarget.y + dwy;
  } else {
    t.x += dwx;
    t.y += dwy;
  }
}
function avgDepth(pts) {
  if (!pts.length) return 0;
  return pts.reduce((s, p) => s + p[2], 0) / pts.length;
}
/* Ground-plane z of the CURRENT pose: the lowest capsule/feet z, i.e. where
 * the robot stands. The 3D floor grid + envelope polygons are drawn here so
 * they sit at the robot's feet instead of at the body-yaw plane z=0 (which
 * used to occlude the legs/feet with the region polygon). */
function groundZ(d) {
  let zg = 0;
  (d.capsules_3d || []).forEach(s => { zg = Math.min(zg, s[0][2], s[1][2]); });
  (d.feet || []).forEach(f => zg = Math.min(zg, f[2] || 0));
  return zg;
}

function drawGrid(svg, T, bounds) {
  const major = niceStep(Math.max(bounds.x1 - bounds.x0, bounds.y1 - bounds.y0));
  const minor = major / 5;
  for (let x = Math.floor(bounds.x0 / minor) * minor; x <= bounds.x1 + 1e-9; x += minor) {
    const isMajor = Math.abs(x / major - Math.round(x / major)) < 1e-9;
    const a = toV([x, bounds.y0], T), b = toV([x, bounds.y1], T);
    line(svg, a, b, { class: isMajor ? "g-gridmajor" : "g-gridline", "shape-rendering": "crispEdges" });
  }
  for (let y = Math.floor(bounds.y0 / minor) * minor; y <= bounds.y1 + 1e-9; y += minor) {
    const isMajor = Math.abs(y / major - Math.round(y / major)) < 1e-9;
    const a = toV([bounds.x0, y], T), b = toV([bounds.x1, y], T);
    line(svg, a, b, { class: isMajor ? "g-gridmajor" : "g-gridline", "shape-rendering": "crispEdges" });
  }
  if (bounds.x0 < 0 && bounds.x1 > 0) {
    line(svg, toV([0, bounds.y0], T), toV([0, bounds.y1], T), { class: "g-axis", "shape-rendering": "crispEdges" });
  }
  if (bounds.y0 < 0 && bounds.y1 > 0) {
    line(svg, toV([bounds.x0, 0], T), toV([bounds.x1, 0], T), { class: "g-axis", "shape-rendering": "crispEdges" });
  }
  for (let x = Math.ceil(bounds.x0 / major) * major; x <= bounds.x1 + 1e-9; x += major) {
    if (Math.abs(x) < major * 0.001) continue;
    const p = toV([x, bounds.y0], T);
    const t = svgEl("text", { x: p[0], y: p[1] - 12, "text-anchor": "middle", class: "g-ticklabel" }, svg);
    t.textContent = (Math.round(x * 100) / 100).toString();
  }
  for (let y = Math.ceil(bounds.y0 / major) * major; y <= bounds.y1 + 1e-9; y += major) {
    if (Math.abs(y) < major * 0.001) continue;
    const p = toV([bounds.x0, y], T);
    const t = svgEl("text", { x: p[0] - 12, y: p[1] + 5, "text-anchor": "end", class: "g-ticklabel" }, svg);
    t.textContent = (Math.round(y * 100) / 100).toString();
  }
  // axis arrows + labels (offset toward the origin corner of the frame)
  const origin = toV([0, 0], T);
  const len = Math.min(70, Math.abs(T.S * bounds.x1 * 0.82));
  const lenY = Math.min(70, Math.abs(T.S * bounds.y1 * 0.82));
  const hx = len >= 6 ? origin[0] + len : origin[0];
  const hy = lenY >= 6 ? origin[1] - lenY : origin[1];
  if (len >= 6) {
    line(svg, origin, [hx, origin[1]], { class: "g-axisarrow" });
    svgEl("path", {
      d: `M${hx} ${origin[1]} L${hx - 13} ${origin[1] - 6} L${hx - 13} ${origin[1] + 6} Z`,
      class: "g-axisarrow",
    }, svg);
    const t = svgEl("text", { x: hx, y: origin[1] + 26, "text-anchor": "middle", class: "g-axislabel" }, svg);
    t.textContent = "+x fwd";
  }
  if (lenY >= 6) {
    line(svg, origin, [origin[0], hy], { class: "g-axisarrow" });
    svgEl("path", {
      d: `M${origin[0]} ${hy} L${origin[0] - 6} ${hy + 13} L${origin[0] + 6} ${hy + 13} Z`,
      class: "g-axisarrow",
    }, svg);
    const t = svgEl("text", { x: origin[0] - 26, y: hy, "text-anchor": "middle", class: "g-axislabel" }, svg);
    t.textContent = "+y left";
  }
}

function drawGrid3D(svg, proj, b, zg) {
  const major = niceStep(Math.max(
    Math.max(Math.abs(b.cx) + b.scale / 430 * 2, 1),
    Math.max(Math.abs(b.cy) + b.scale / 430 * 2, 1),
  ));
  const minor = major / 5;
  const ext = Math.max(1.2, b.scale / 430 * 2.2);
  // Grid + origin axes sit on the ground plane at z=zg (the robot's feet), so
  // the floor reads as under the legs rather than slicing through the torso.
  for (let x = -Math.ceil(ext / minor) * minor; x <= ext + 1e-9; x += minor) {
    const isMajor = Math.abs(x / major - Math.round(x / major)) < 1e-9;
    const a = proj([x, -ext, zg]), c = proj([x, ext, zg]);
    line(svg, [a[0], a[1]], [c[0], c[1]], { class: isMajor ? "g-gridmajor" : "g-gridline", "shape-rendering": "crispEdges" });
  }
  for (let y = -Math.ceil(ext / minor) * minor; y <= ext + 1e-9; y += minor) {
    const isMajor = Math.abs(y / major - Math.round(y / major)) < 1e-9;
    const a = proj([-ext, y, zg]), c = proj([ext, y, zg]);
    line(svg, [a[0], a[1]], [c[0], c[1]], { class: isMajor ? "g-gridmajor" : "g-gridline", "shape-rendering": "crispEdges" });
  }
  const ox = proj([0, 0, zg]);
  if (ox[0] > -100 && ox[0] < 1100 && ox[1] > -100 && ox[1] < 1100) {
    const a = proj([ext, 0, zg]), c = proj([0, ext, zg]);
    line(svg, [ox[0], ox[1]], [a[0], a[1]], { class: "g-axis" });
    line(svg, [ox[0], ox[1]], [c[0], c[1]], { class: "g-axis" });
  }
}

/* shared capsule + skeleton scene helpers -------------------------------- */
function scenePolygons(d) {
  const out = [];
  const add = (pts, cls) => { if (pts && pts.length >= 3) out.push({ pts, cls }); };
  if (state.mode === "A") {
    add(d.reachable_polygon, "poly-reach");
    add(d.polygon, "poly-occ");
    add(d.margin_polygon, "poly-margin");
    if (state.aLegacy && d.legacy_condition) {
      const c = d.legacy_condition, w = c[0], xf = c[3], xb = c[4];
      add([[xf, w], [0, w], [xb, w], [xb, -w], [0, -w], [xf, -w]], "poly-legacy");
    }
  } else {
    add(d.allowed_polygon, "poly-allowed");
    add(d.envelope_polygon, "poly-hex");
  }
  return out;
}

function drawCapsule2D(svg, seg, T) {
  const a = toV(seg.a, T), b = toV(seg.b, T);
  // The BASE/torso capsule is rendered as a thin faint dashed hairline ONLY
  // (no end-cap radius circles, no thick stroke): the torso would otherwise
  // form a large dark blob at the body center that occludes the envelope
  // polygon, axes, skeleton and arcs. The 18 leg capsules keep their radius.
  if (/^base/i.test(seg.name)) {
    line(svg, a, b, { class: "capsule-line-base", stroke: seg.color, "data-capsule": seg.name });
    return;
  }
  line(svg, a, b, { class: "capsule-line", stroke: seg.color, "data-capsule": seg.name });
  const rpx = Math.max(3, seg.r * T.S);
  svgEl("circle", { cx: a[0], cy: a[1], r: rpx, class: "capsule-cap", fill: "none", stroke: seg.color, "stroke-width": 2, opacity: 0.8, "data-capsule": seg.name }, svg);
  svgEl("circle", { cx: b[0], cy: b[1], r: rpx, class: "capsule-cap", fill: "none", stroke: seg.color, "stroke-width": 2, opacity: 0.8, "data-capsule": seg.name }, svg);
}
function capsuleSegments(d) {
  if (!d.capsules_3d) return [];
  return d.capsules_3d.map((seg, i) => ({
    a: seg[0], b: seg[1], r: d.capsule_radius[i] || 0,
    name: (d.capsule_names && d.capsule_names[i]) || "",
    color: capsuleColor((d.capsule_names && d.capsule_names[i]) || ""),
  }));
}
function drawSkeleton2D(svg, d, T) {
  if (!d.skeleton) return;
  const sk = d.skeleton;
  sk.legs.forEach(l => {
    const haa = toV(l.haa, T), hfe = toV(l.hfe, T), kfe = toV(l.kfe, T), foot = toV(l.foot, T);
    const color = LEG_COLOR[l.leg];
    // exactly 3 segments per leg: HAA-HFE, HFE-KFE, KFE-Foot (no base->hip,
    // no direct hip->foot line)
    polyline(svg, [haa, hfe, kfe, foot], { class: "skel-leg", stroke: color, "data-leg": l.leg });
    dot(svg, haa, { class: "skel-joint", r: 8, fill: color, "data-leg": l.leg });
    dot(svg, hfe, { class: "skel-joint", r: 5.5, fill: "#1A2429", stroke: color, "stroke-width": 2, "data-leg": l.leg });
    dot(svg, kfe, { class: "skel-joint", r: 4.5, fill: "#1A2429", stroke: color, "stroke-width": 2, "data-leg": l.leg });
  });
  if (sk.base) dot(svg, toV(sk.base, T), { class: "skel-base", r: 10, fill: BASE_COLOR });
}

/* ------------------------------------------------------- mode A render */
function collectA(d) {
  const pts = [];
  const add = a => { if (a) a.forEach(p => pts.push(p)); };
  add(d.polygon); add(d.margin_polygon); add(d.reachable_polygon);
  if (d.capsules_3d) d.capsules_3d.forEach(seg => { pts.push(seg[0]); pts.push(seg[1]); });
  if (d.skeleton) {
    pts.push(d.skeleton.base);
    d.skeleton.legs.forEach(l => {
      pts.push(l.haa); pts.push(l.hfe); pts.push(l.kfe); pts.push(l.foot);
    });
  }
  add(d.feet);
  if (state.aLegacy && d.legacy_condition) {
    const c = d.legacy_condition, w = c[0], xf = c[3], xb = c[4];
    add([[xf, w], [0, w], [xb, w], [xb, -w], [0, -w], [xf, -w]]);
  }
  return boundsOf(pts);
}

function renderModeA(d) {
  const svg = document.getElementById("stageSvg");
  svg.innerHTML = "";
  const bounds = collectA(d);
  const T = computeTransform(bounds);
  state.proj = p => toV([p[0], p[1]], T);
  setStageBanner(d);
  drawGrid(svg, T, bounds);

  if (d.reachable_polygon) drawPoly(svg, X(d.reachable_polygon, T), { class: "poly-reach" });

  capsuleSegments(d).forEach(seg => drawCapsule2D(svg, seg, T));

  d.polygon.forEach((_p, i) => {
    const a = toV(d.polygon[i], T);
    const b = toV(d.polygon[(i + 1) % d.polygon.length], T);
    const cap = d.edge_bindings[i];
    line(svg, a, b, {
      class: "poly-face",
      "data-cap": cap == null ? "null" : cap,
      stroke: capColor(d, cap),
    });
  });

  if (d.margin_polygon) drawPoly(svg, X(d.margin_polygon, T), { class: "poly-margin" });

  if (state.aLegacy && d.legacy_condition) {
    const c = d.legacy_condition, w = c[0], xf = c[3], xb = c[4];
    drawPoly(svg, X([[xf, w], [0, w], [xb, w], [xb, -w], [0, -w], [xf, -w]], T), { class: "poly-legacy" });
  }

  drawSkeleton2D(svg, d, T);
  d.feet.forEach(f => dot(svg, toV(f, T), { class: "foot-dot", r: 7 }));

  drawCardinalLabels(svg, d, T);
  svgEl("g", { id: "animLayer" }, svg);
}

function drawCardinalLabels(svg, d, T) {
  const dirs = d.directions;
  const supp = d.h_occ;
  for (const lab of ["+x", "+y", "-x", "-y"]) {
    const k = d.cardinal_indices ? d.cardinal_indices[lab] : null;
    if (k == null) continue;
    const u = dirs[k];
    let mid = null;
    for (let i = 0; i < d.polygon.length; i++) {
      const m = [(d.polygon[i][0] + d.polygon[(i + 1) % d.polygon.length][0]) / 2,
                 (d.polygon[i][1] + d.polygon[(i + 1) % d.polygon.length][1]) / 2];
      if (Math.abs(u[0] * m[0] + u[1] * m[1] - supp[k]) < 1e-3) { mid = m; break; }
    }
    if (!mid) continue;
    const bind = d.cardinal_bindings[lab];
    const col = bind.leg === "BASE" ? BASE_COLOR : LEG_COLOR[bind.leg];
    const out = [mid[0] + u[0] * 0.12, mid[1] + u[1] * 0.12];
    const p = toV(out, T);
    const t = svgEl("text", { x: p[0], y: p[1], "text-anchor": "middle", fill: col, class: "g-ticklabel" }, svg);
    t.textContent = bind.leg + "/" + bind.part;
  }
}

/* ------------------------------------------------------- mode B render */
function collectB(d) {
  const pts = [];
  const add = a => { if (a) a.forEach(p => pts.push(p)); };
  add(d.allowed_polygon); add(d.envelope_polygon); add(d.feet);
  if (d.capsules_3d) d.capsules_3d.forEach(seg => { pts.push(seg[0]); pts.push(seg[1]); });
  if (d.skeleton) {
    pts.push(d.skeleton.base);
    d.skeleton.legs.forEach(l => {
      pts.push(l.haa); pts.push(l.hfe); pts.push(l.kfe); pts.push(l.foot);
    });
  }
  if (d.magenta_arcs) d.magenta_arcs.forEach(a => { add(a.points); add(a.trajectory); });
  if (d.amber_arcs) d.amber_arcs.forEach(a => add(a.points));
  if (d.haa_arc_geometry) {
    add(d.haa_arc_geometry.origins);
    add(d.haa_arc_geometry.markers);
    d.haa_arc_geometry.arcs.forEach(a => add(a));
  }
  return boundsOf(pts);
}

function setStageBanner(d) {
  const banner = document.getElementById("stageBanner");
  banner.classList.remove("amber");
  if (!d.feasible_reference) {
    if (state.mode === "C") {
      // Mode C: no posture at all fits this envelope -> full-range rejection.
      banner.textContent = "NO FEASIBLE POSTURE · " +
        (d.reason || "envelope too small to fit the robot — HAA rejection is the full range");
      banner.classList.add("amber");
    } else {
      banner.textContent = "NO FEASIBLE REFERENCE · " + (d.reason || "envelope too tight for the reference pose");
    }
    banner.classList.remove("hidden");
  } else if (state.mode === "B" && d.reference_infeasible_requested) {
    banner.textContent = "reference pose infeasible at this envelope — using " +
      (d.reference_source || "fallback") + " instead (nearest feasible candidate)";
    banner.classList.add("amber");
    banner.classList.remove("hidden");
  } else {
    banner.textContent = "";
    banner.classList.add("hidden");
  }
}

function renderModeB(d) {
  const svg = document.getElementById("stageSvg");
  svg.innerHTML = "";
  setStageBanner(d);

  const bounds = collectB(d);
  const T = computeTransform(bounds);
  state.proj = p => toV([p[0], p[1]], T);
  drawGrid(svg, T, bounds);

  if (d.allowed_polygon) {
    drawPoly(svg, X(d.allowed_polygon, T), {
      class: "poly-allowed-derived",
      fill: "rgba(34,184,175,0.06)",
      stroke: MODEL.occ,
      "stroke-width": 4,
      opacity: 0.9,
    });
  }
  if (d.envelope_polygon) drawPoly(svg, X(d.envelope_polygon, T), { class: "poly-hex" });

  capsuleSegments(d).forEach(seg => drawCapsule2D(svg, seg, T));

  const feasible = !!d.feasible_reference;
  if (!feasible) { svgEl("g", { id: "animLayer" }, svg); return; }

  const haa = d.haa_arc_geometry;
  if (haa) {
    if (state.bAccessible) {
      haa.arcs.forEach(a => polyline(svg, X(a, T), { class: "arc-haa" }));
      haa.markers.forEach(m => dot(svg, toV(m, T), { class: "arc-marker", r: 9 }));
    }
  }

  drawSkeleton2D(svg, d, T);
  d.feet.forEach(f => dot(svg, toV(f, T), { class: "foot-dot", r: 7 }));

  if (state.bRejected) d.magenta_arcs.forEach(a => {
    polyline(svg, X(a.points, T), { class: "arc-rej", "data-joint": a.joint_name });
    if (a.trajectory) polyline(svg, X(a.trajectory, T), { class: "trajectory-line", "data-joint": a.joint_name });
  });
  if (state.bAccessible) d.amber_arcs.forEach(a => polyline(svg, X(a.points, T), { class: "arc-acc" }));
  // Mode C current-HAA markers on each leg's rejection arc (verify the band).
  if (d.haa_markers) d.haa_markers.forEach(m => {
    if (!m.point) return;
    const mem = haaMembership(d, m);
    dot(svg, toV(m.point, T), { class: "haa-marker " + mem, r: 11, "data-leg": m.leg });
  });
  svgEl("g", { id: "animLayer" }, svg);
}

/* ------------------------------------------------------------ 3D render */
function render3D(d) {
  const svg = document.getElementById("stageSvg");
  svg.innerHTML = "";
  setStageBanner(d);

  const b = collect3DBounds(d);
  state.camBounds = b;
  // The scene-fit scale auto-derives from the bounding box, but the camera
  // TARGET is the user-controlled pan point (default the ground origin) and
  // is NEVER overwritten here -- so pan/rotate/recenter persist across every
  // recompute and mode/preset switch.
  state.camera.scale = b.scale;
  const zg = groundZ(d);
  const proj = makeProject3D(b, zg);
  state.proj = proj;
  const stage = document.getElementById("stage");
  stage.dataset.cameraTargetX = String(state.camera.target.x);
  stage.dataset.cameraTargetY = String(state.camera.target.y);
  stage.dataset.cameraTargetZ = String(state.camera.target.z);
  stage.dataset.cameraYaw = String(state.camera.yaw);
  stage.dataset.cameraPitch = String(state.camera.pitch);
  stage.dataset.groundOriginZ = String(zg);
  drawGrid3D(svg, proj, b, zg);

  const items = [];

  // ground-plane envelope / allowed / overlay polygons (wireframe + translucent
  // fill) at the CURRENT pose's foot plane (z=zg), with a negative depth bias
  // so the region is always painter-sorted BEHIND the legs/feet that hang down
  // to the floor. The old z=0 plane sorted nearer than the leg capsules (which
  // span z -0.589..0) and occluded them as a black region. The polygon depth is
  // floored at its FARTHEST point (the ground plane extends beyond every leg),
  // so it robustly sorts behind every capsule/skeleton/arc regardless of view.
  scenePolygons(d).forEach(poly => {
    const pts2 = poly.pts.map(p => [p[0], p[1], zg]);
    const pr = pts2.map(proj);
    const depth = Math.min(avgDepth(pr), ...pr.map(p => p[2])) - 0.05;
    const cls = poly.cls;
    items.push({ depth, draw: () => {
      const sxy = pr.map(p => [p[0], p[1]]);
      if (cls === "poly-occ") svgEl("polygon", { points: polyPoints(sxy), class: "poly-occ-fill" }, svg);
      if (cls === "poly-allowed") svgEl("polygon", { points: polyPoints(sxy), class: "poly-allowed-fill" }, svg);
      svgEl("polygon", { points: polyPoints(sxy), class: cls }, svg);
    }});
  });

  // capsule proxies: thick stroke + projected end-cap circles. The BASE/torso
  // capsule is a thin faint dashed hairline only (no end caps / no thick bar)
  // so it never forms an occluding dark blob over the envelope center / axes.
  capsuleSegments(d).forEach(seg => {
    const a = proj(seg.a), b2 = proj(seg.b);
    const depth = (a[2] + b2[2]) / 2;
    if (/^base/i.test(seg.name)) {
      items.push({ depth, draw: () => {
        line(svg, [a[0], a[1]], [b2[0], b2[1]], {
          class: "capsule-line-base", stroke: seg.color, "data-capsule": seg.name,
        });
      }});
      return;
    }
    const rpx = Math.max(2.5, seg.r * b.scale * state.camera.zoom);
    items.push({ depth, draw: () => {
      line(svg, [a[0], a[1]], [b2[0], b2[1]], {
        class: "capsule-line", stroke: seg.color,
        "stroke-width": Math.max(3, rpx * 1.6), "data-capsule": seg.name,
      });
      svgEl("circle", { cx: a[0], cy: a[1], r: rpx, class: "capsule-cap", fill: "none", stroke: seg.color, "stroke-width": 2, opacity: 0.85, "data-capsule": seg.name }, svg);
      svgEl("circle", { cx: b2[0], cy: b2[1], r: rpx, class: "capsule-cap", fill: "none", stroke: seg.color, "stroke-width": 2, opacity: 0.85, "data-capsule": seg.name }, svg);
    }});
  });

  // joint skeleton (HAA/HFE/KFE/foot bones + markers, per-leg color, data-leg tagged)
  if (d.skeleton) {
    d.skeleton.legs.forEach(l => {
      const haa = proj(l.haa), hfe = proj(l.hfe), kfe = proj(l.kfe), foot = proj(l.foot);
      const depth = (haa[2] + hfe[2] + kfe[2] + foot[2]) / 4;
      const color = LEG_COLOR[l.leg];
      items.push({ depth, draw: () => {
        polyline(svg, [[haa[0], haa[1]], [hfe[0], hfe[1]], [kfe[0], kfe[1]], [foot[0], foot[1]]],
          { class: "skel-leg", stroke: color, "data-leg": l.leg });
        dot(svg, [haa[0], haa[1]], { class: "skel-joint", r: 8, fill: color, "data-leg": l.leg });
        dot(svg, [hfe[0], hfe[1]], { class: "skel-joint", r: 5.5, fill: "#1A2429", stroke: color, "stroke-width": 2, "data-leg": l.leg });
        dot(svg, [kfe[0], kfe[1]], { class: "skel-joint", r: 4.5, fill: "#1A2429", stroke: color, "stroke-width": 2, "data-leg": l.leg });
      }});
    });
    if (d.skeleton.base) {
      const bp = proj(d.skeleton.base);
      items.push({ depth: bp[2], draw: () => dot(svg, [bp[0], bp[1]], { class: "skel-base", r: 10, fill: BASE_COLOR }) });
    }
  }

  // magenta rejected bands (stylized arc) + true foot trajectories
  if (state.bRejected) (d.magenta_arcs || []).forEach(a => {
    const pr = a.points.map(proj);
    items.push({ depth: avgDepth(pr), draw: () =>
      polyline(svg, pr.map(p => [p[0], p[1]]), { class: "arc-rej", "data-joint": a.joint_name }) });
    if (a.trajectory) {
      const tr = a.trajectory.map(proj);
      items.push({ depth: avgDepth(tr) - 0.02, draw: () =>
        polyline(svg, tr.map(p => [p[0], p[1]]), { class: "trajectory-line", "data-joint": a.joint_name }) });
    }
  });

  // amber accessible HAA arcs + haa arc geometry
  if (state.bAccessible) {
    (d.amber_arcs || []).forEach(a => {
      const pr = a.points.map(proj);
      items.push({ depth: avgDepth(pr), draw: () =>
        polyline(svg, pr.map(p => [p[0], p[1]]), { class: "arc-acc" }) });
    });
    if (d.haa_arc_geometry) {
      d.haa_arc_geometry.arcs.forEach(a => {
        const pr = a.map(proj);
        items.push({ depth: avgDepth(pr), draw: () =>
          polyline(svg, pr.map(p => [p[0], p[1]]), { class: "arc-haa" }) });
      });
      d.haa_arc_geometry.markers.forEach(m => {
        const mp = proj(m);
        items.push({ depth: mp[2], draw: () => dot(svg, [mp[0], mp[1]], { class: "arc-marker", r: 9 }) });
      });
    }
  }

  // feet
  (d.feet || []).forEach(f => {
    const fp = proj(f);
    items.push({ depth: fp[2], draw: () => dot(svg, [fp[0], fp[1]], { class: "foot-dot", r: 8 }) });
  });

  // Mode C current-HAA markers on each leg's rejection arc (verify the band).
  (d.haa_markers || []).forEach(m => {
    if (!m.point) return;
    const mp = proj(m.point);
    const mem = haaMembership(d, m);
    items.push({ depth: mp[2] - 0.01, draw: () =>
      dot(svg, [mp[0], mp[1]], { class: "haa-marker " + mem, r: 11, "data-leg": m.leg }) });
  });

  items.sort((x, y) => x.depth - y.depth);     // far -> near painter sort
  items.forEach(it => it.draw());
  svgEl("g", { id: "animLayer" }, svg);

  // Blender-style orbit semantics in viewer coordinates: the model's ground
  // origin is display-world (0,0,0). At default/recenter target zero, this
  // reticle lands exactly at the SVG and displayed stage centre.
  const origin = proj([0, 0, zg]);
  const og = svgEl("g", {
    id: "worldOrigin", class: "world-origin",
    "data-world-x": "0", "data-world-y": "0", "data-world-z": "0",
    "data-model-z": String(zg),
  }, svg);
  line(og, [origin[0] - 13, origin[1]], [origin[0] + 13, origin[1]], { class: "world-origin-line" });
  line(og, [origin[0], origin[1] - 13], [origin[0], origin[1] + 13], { class: "world-origin-line" });
  dot(og, [origin[0], origin[1]], { class: "world-origin-dot", r: 3.5 });
}

/* ------------------------------------------- rejection-band animation */
function ensureAnimLayer() {
  let layer = document.getElementById("animLayer");
  if (!layer) layer = svgEl("g", { id: "animLayer" }, document.getElementById("stageSvg"));
  return layer;
}
function startAnimation(d) {
  stopAnimation();
  if (state.mode !== "B" || !state.bAnimate || reducedMotion()) return;
  if (!d || !d.feasible_reference) return;
  const arcs = (d.magenta_arcs || []).filter(a => a.trajectory && a.trajectory.length >= 2);
  if (!arcs.length) return;
  state.anim = {
    arcs, idx: 0, phase: 0, lastT: performance.now(), running: true,
  };
  state.anim.raf = requestAnimationFrame(animTick);
}
function stopAnimation() {
  if (state.anim) {
    if (state.anim.raf) cancelAnimationFrame(state.anim.raf);
    state.anim = null;
  }
  dimLeg(null);
  const layer = document.getElementById("animLayer");
  if (layer) layer.innerHTML = "";
  const label = document.getElementById("animLabel");
  if (label) label.textContent = "";
}
function animTick(now) {
  const st = state.anim;
  if (!st || !st.running) return;
  const dt = Math.min(0.06, (now - st.lastT) / 1000);
  st.lastT = now;
  st.phase += dt / SWEEP_DURATION;
  if (st.phase >= 1) {
    st.phase = 0;
    st.idx = (st.idx + 1) % st.arcs.length;
  }
  drawAnimOverlay(st);
  st.raf = requestAnimationFrame(animTick);
}
function dimLeg(leg) {
  const svg = document.getElementById("stageSvg");
  svg.querySelectorAll(".skel-leg.anim-dim").forEach(el => el.classList.remove("anim-dim"));
  if (leg) svg.querySelectorAll(".skel-leg[data-leg='" + leg + "']").forEach(el => el.classList.add("anim-dim"));
}
function drawAnimOverlay(st) {
  const layer = ensureAnimLayer();
  layer.innerHTML = "";
  const proj = state.proj || (p => [p[0], p[1]]);
  const arc = st.arcs[st.idx];
  const frac = Math.max(0, Math.min(1, st.phase));
  const ti = Math.min(arc.trajectory.length - 1, Math.round(frac * (arc.trajectory.length - 1)));

  // highlight the active band border + its true foot trajectory
  polyline(layer, arc.points.map(p => { const q = proj(p); return [q[0], q[1]]; }),
    { class: "anim-band", "data-joint": arc.joint_name });
  polyline(layer, arc.trajectory.map(p => { const q = proj(p); return [q[0], q[1]]; }),
    { class: "anim-trajectory" });

  // swept leg skeleton at this phase (server keyframes, no per-frame round trip)
  const frames = arc.leg_skeleton_frames || [];
  if (frames.length) {
    const fi = Math.min(frames.length - 1, Math.round(frac * (frames.length - 1)));
    const [haa, hfe, kfe, foot] = frames[fi];
    const hp = proj(haa), hfp = proj(hfe), kfp = proj(kfe), fp = proj(foot);
    const color = LEG_COLOR[arc.leg] || MODEL.bone;
    polyline(layer, [[hp[0], hp[1]], [hfp[0], hfp[1]], [kfp[0], kfp[1]], [fp[0], fp[1]]],
      { class: "anim-leg", stroke: color, "data-leg": arc.leg });
    dot(layer, [hp[0], hp[1]], { class: "anim-joint", r: 8, fill: color });
    dot(layer, [hfp[0], hfp[1]], { class: "anim-joint", r: 5.5, fill: "#1A2429", stroke: color, "stroke-width": 2 });
    dot(layer, [kfp[0], kfp[1]], { class: "anim-joint", r: 4.5, fill: "#1A2429", stroke: color, "stroke-width": 2 });
    dimLeg(arc.leg);
  }

  // moving foot highlight on the TRUE trajectory
  const fp = proj(arc.trajectory[ti]);
  svgEl("circle", { cx: fp[0], cy: fp[1], r: 12, class: "anim-foot" }, layer);

  // sweep readout
  const label = document.getElementById("animLabel");
  if (label) {
    const sw = arc.sweep ? arc.sweep[ti] : null;
    label.textContent = (st.idx + 1) + "/" + st.arcs.length + " · " + arc.joint_name +
      " · " + (sw != null ? fmt(sw, 2) + " rad" : "");
  }
}

/* ----------------------------------------------------------- readouts */
function readoutA(d) {
  const max = d.max_h_occ || 1e-9;
  const bars = d.h_occ.map((h, k) => {
    const cap = d.binding[k];
    const col = capColor(d, cap);
    const pct = Math.max(1.5, h / max * 100);
    const name = d.capsule_names[cap] || "";
    return `<div class="hk-bar" data-cap="${cap}" style="height:${pct.toFixed(1)}%;background:${col}" title="${esc(name)} limits this direction"></div>`;
  }).join("");
  const b = d.cardinal_bindings || {};
  const sw = x => `<span class="swatch" style="background:${x && x.leg === "BASE" ? BASE_COLOR : LEG_COLOR[(x || {}).leg] || BASE_COLOR}"></span>`;
  const cc = d.legacy_condition_clamped;
  const priors = d.priors.map(x => x.toFixed(3)).join(", ");
  return `
  <div class="hk-wrap">
    <h3 class="hk-title">h_k^occ per direction · colored by the body part that binds each face</h3>
    <div class="hk-bars">${bars}</div>
    <div class="hk-axis"><span>−180 · −x</span><span>−90 · −y</span><span>0 · +x</span><span>+90 · +y</span><span>+180 · −x</span></div>
    <div class="hk-summary">
      <span>K <b>${d.K}</b></span>
      <span>margin <b>${d.margin.toFixed(3)} m</b></span>
      <span>capsules <b>${d.capsule_set === "full" ? "full 19" : "torso-only"}</b></span>
      <span>max h_occ <b>${d.max_h_occ.toFixed(3)} m</b> @ <b>${d.max_h_occ_direction_deg.toFixed(0)}°</b> ${sw(d.max_h_occ_binding)}<b>${d.max_h_occ_binding.leg}/${d.max_h_occ_binding.part}</b></span>
      <span>legacy raw <b>[w ${d.legacy_width.toFixed(3)} · xf ${d.legacy_x_front.toFixed(3)} · xb ${d.legacy_x_back.toFixed(3)}]</b></span>
      <span>legacy clamped <b>[w ${cc[0].toFixed(3)} · xf ${cc[3].toFixed(3)} · xb ${cc[4].toFixed(3)}]</b></span>
      <span>priors <b>[${priors}]</b></span>
      <span>+x ${sw(b["+x"])}<b>${b["+x"] ? b["+x"].leg + "/" + b["+x"].part : "—"}</b></span>
      <span>+y ${sw(b["+y"])}<b>${b["+y"] ? b["+y"].leg + "/" + b["+y"].part : "—"}</b></span>
      <span>−x ${sw(b["-x"])}<b>${b["-x"] ? b["-x"].leg + "/" + b["-x"].part : "—"}</b></span>
      <span>−y ${sw(b["-y"])}<b>${b["-y"] ? b["-y"].leg + "/" + b["-y"].part : "—"}</b></span>
    </div>
  </div>`;
}

function refPoseLine(d) {
  if (!d.feasible_reference) return "";
  const used = d.reference_used_q || d.reference_q;
  let html = `<div class="ref-detail"><span class="dim">reference q (used)</span><code>${esc(fmtPose(used))}</code></div>`;
  if (d.reference_infeasible_requested && d.reference_requested_q) {
    html += `<div class="ref-detail warn"><span class="dim">requested q (infeasible)</span><code>${esc(fmtPose(d.reference_requested_q))}</code></div>`;
    html += `<div class="ref-detail warn">fell back to <b>${esc(d.reference_source)}</b> · the skeleton / arcs use the pose actually used</div>`;
  }
  return html;
}

function readoutB(d) {
  const feasible = !!d.feasible_reference;
  const cells = LEGS.map(leg => {
    const rows = JOINTS.map(jt => {
      const jn = leg + "_" + jt;
      const j = JOINT_NAMES.indexOf(jn);
      let reach, rejHtml;
      if (!feasible) {
        reach = "reach[∅]"; rejHtml = "rej(−)";
      } else {
        reach = "reach[" + fmt(d.accessible_box.lower[j], 2) + "," + fmt(d.accessible_box.upper[j], 2) + "]";
        const ints = d.rejected.per_joint_intervals_rad[j];
        rejHtml = ints.length
          ? "rej(" + ints.map(iv => fmtN(iv[0], 2) + "–" + fmtN(iv[1], 2)).join(";") + ")"
          : "rej(−)";
      }
      return `<div class="rc-line"><span class="dim">${jt}</span> <span class="dim">${reach}</span> <span class="rej">${rejHtml}</span></div>`;
    }).join("");
    return `<div class="rej-cell"><div class="rc-leg"><span class="leg-chip" style="background:${LEG_COLOR[leg]}"></span>${leg}</div>${rows}</div>`;
  }).join("");

  const refTag = !feasible
    ? `<span>reference <b>INFEASIBLE</b></span>
       <span>reason <b>${esc(d.reason || "envelope too tight for the reference pose")}</b></span>`
    : (d.reference_infeasible_requested
        ? `<span>reference <b>requested → INFEASIBLE</b></span>
           <span>using <b>${esc(d.reference_source)}</b> (nearest feasible candidate)</span>`
        : `<span>reference <b>FEASIBLE</b></span>
           <span>source <b>${esc(d.reference_source)}</b></span>`);

  const summary = feasible ? `
      ${refTag}
      <span class="pill">${d.rejected.rejected_joint_count} / 18 joints rejected</span>
      <span>max span <b>${d.rejected.max_rejected_span_rad.toFixed(3)} rad</b> · <b>${esc(d.rejected.max_rejected_joint_name || "—")}</b></span>`
    : refTag;
  const e = d.envelope || {};
  const envLine = `envelope w ${e.w_f.toFixed(2)}/${e.w_m.toFixed(2)}/${e.w_b.toFixed(2)} · x ${e.x_f.toFixed(2)}/${e.x_b.toFixed(2)} · +${(e.margin || 0).toFixed(3)} m · K ${d.K}`;
  return `
  <div class="rej-wrap">
    <div class="rej-summary">${summary}<span>${envLine}</span></div>
    ${refPoseLine(d)}
    <div class="rej-grid">${cells}</div>
  </div>`;
}

/* ------------------------------- mode C readout (focused HAA per leg) */
function accessibleFromIntervals(rejected, lo, hi) {
  // mirrors the server's accessible_interval_complement within the full box
  const out = [];
  let cursor = lo;
  for (const iv of rejected || []) {
    const rlo = iv[0], rhi = iv[1];
    if (rhi <= lo || rlo >= hi) continue;
    const low = Math.max(cursor, rlo);
    const high = Math.min(hi, rhi);
    if (cursor < low) out.push([cursor, low]);
    cursor = Math.max(cursor, high);
  }
  if (cursor < hi) out.push([cursor, hi]);
  return out;
}

/* Is the current HAA check value inside a rejected interval for that leg? */
function haaMembership(d, m) {
  const j = JOINT_NAMES.indexOf(m.leg + "_HAA");
  const ints = d.rejected.per_joint_intervals_rad[j] || [];
  return ints.some(iv => m.value >= iv[0] - 1e-6 && m.value <= iv[1] + 1e-6) ? "rej" : "acc";
}

function readoutC(d) {
  const feasible = !!d.feasible_reference;
  const cells = LEGS.map((leg, i) => {
    const j = JOINT_NAMES.indexOf(leg + "_HAA");
    const ints = d.rejected.per_joint_intervals_rad[j];
    const rejHtml = ints.length
      ? "rej(" + ints.map(iv => fmtN(iv[0], 2) + "–" + fmtN(iv[1], 2)).join(";") + ")"
      : "rej(−)";
    const accInts = accessibleFromIntervals(ints, -3, 3);
    const accHtml = accInts.length
      ? "acc(" + accInts.map(iv => fmtN(iv[0], 2) + "–" + fmtN(iv[1], 2)).join(";") + ")"
      : "acc(−)";
    const flag = feasible ? "ok" : "inf";
    const haaVal = state.q[3 * i];
    const mem = haaMembership(d, { leg, value: haaVal });
    const memHtml = `<b class="${mem}">${mem.toUpperCase()}</b>`;
    return `<div class="rej-cell">
      <div class="rc-leg"><span class="leg-chip" style="background:${LEG_COLOR[leg]}"></span>${leg}_HAA</div>
      <div class="rc-line"><span class="dim">HAA ${fmt(haaVal, 2)} ${memHtml} · HFE ${fmt(state.q[3 * i + 1], 2)} · KFE ${fmt(state.q[3 * i + 2], 2)} · ${flag}</span></div>
      <div class="rc-line rej">${rejHtml}</div>
      <div class="rc-line"><span class="dim">${accHtml}</span></div>
    </div>`;
  }).join("");

  const refTag = !feasible
    ? `<span>NO FEASIBLE POSTURE</span>
       <span>reason <b>${esc(d.reason || "envelope too small to fit the robot")}</b></span>`
    : d.rejection_mode === "fold"
      ? `<span>fold <b>semantics</b></span>
         <span>rejection with the other legs free to fold via HAA (HFE/KFE pinned)</span>`
      : `<span>pins <b>FEASIBLE</b></span>
         <span>rejection at the exact pins</span>`;
  const poseTag = d.current_feasible
    ? `<span>current pose <b>inside</b></span>`
    : `<span>current pose <b class="rej">LEAVES envelope</b></span>`;
  const summary = `<span class="pill">${d.rejected.rejected_joint_count} / 6 HAA joints rejected</span>`;
  const e = d.envelope || {};
  const envLine = `envelope w ${e.w_f.toFixed(2)}/${e.w_m.toFixed(2)}/${e.w_b.toFixed(2)} · x ${e.x_f.toFixed(2)}/${e.x_b.toFixed(2)} · +${(e.margin || 0).toFixed(3)} m · K ${d.K}`;
  return `
  <div class="rej-wrap">
    <div class="rej-summary">${refTag} ${summary} ${poseTag}<span>${envLine}</span></div>
    ${refPoseLine(d)}
    <div class="rej-grid">${cells}</div>
  </div>`;
}

/* ------------------------------------------------------------- dispatch */
function render(d) {
  state.lastData = d;
  clearSliderPreview();
  redrawStage();
  if (state.mode === "A") {
    document.getElementById("readout").innerHTML = readoutA(d);
    bindCrossRef(document.getElementById("stageSvg"), d);
  } else if (state.mode === "B") {
    document.getElementById("readout").innerHTML = readoutB(d);
  } else {
    document.getElementById("readout").innerHTML = readoutC(d);
  }
}
function redrawStage() {
  const d = state.previewData || state.lastData;
  if (!d) return;
  if (state.view === "3d") render3D(d);
  else if (state.mode === "A") renderModeA(d);
  else renderModeB(d);
  if (state.dragging) stopAnimation();
  else startAnimation(d);
}

function bindCrossRef(svg, d) {
  const faces = svg.querySelectorAll(".poly-face");
  const bars = document.querySelectorAll(".hk-bar");
  const hl = cap => {
    faces.forEach(f => f.classList.toggle("hover", f.dataset.cap === String(cap)));
    bars.forEach(b => b.classList.toggle("hover", b.dataset.cap === String(cap)));
  };
  faces.forEach(f => {
    f.addEventListener("mouseenter", () => hl(f.dataset.cap));
    f.addEventListener("mouseleave", () => hl(null));
  });
  bars.forEach(b => {
    b.addEventListener("mouseenter", () => hl(b.dataset.cap));
    b.addEventListener("mouseleave", () => hl(null));
  });
}

/* --------------------------------------------------- control bindings */
function bindSlider(id, onInput) {
  const el = document.getElementById(id);
  const val = document.getElementById(id + "Val");
  const handler = () => {
    const v = parseFloat(el.value);
    if (val) val.textContent = v.toFixed(el.step >= 1 ? 0 : 3);
    onInput(v);
    el.dataset.commitDirty = "1";
    handleSliderInput("parameter");
  };
  el.addEventListener("input", handler);
  bindSliderCommit(el);
}

function bindSegGroup(id, selector, onPick) {
  const g = document.getElementById(id);
  g.addEventListener("click", ev => {
    const b = ev.target.closest(selector);
    if (!b) return;
    g.querySelectorAll(selector).forEach(x => x.classList.toggle("active", x === b));
    onPick(b);
  });
}

function syncCapsules() {
  for (const gid of ["aCapsule", "bCapsule", "cCapsule"]) {
    document.querySelectorAll(`#${gid} .seg[data-capsule]`).forEach(b =>
      b.classList.toggle("active", b.dataset.capsule === state.capsule));
  }
}

function setView(v) {
  if (state.view === v) return;
  state.view = v;
  document.getElementById("viewTop").classList.toggle("active", v === "topdown");
  document.getElementById("viewTop").setAttribute("aria-pressed", v === "topdown");
  document.getElementById("view3d").classList.toggle("active", v === "3d");
  document.getElementById("view3d").setAttribute("aria-pressed", v === "3d");
  const hint = document.getElementById("stageHint");
  if (hint) hint.textContent = v === "3d"
    ? "right-drag to pan · wheel zoom · drag rotate"
    : "drag to pan · wheel zoom";
  document.body.dataset.view = v;
  const viewLabel = document.getElementById("stageViewLabel");
  if (viewLabel) viewLabel.textContent = v === "3d" ? "ORBIT 3D" : "TOP-DOWN";
  const svg = document.querySelector("#stageSvg");
  if (svg) svg.setAttribute("aria-label",
    v === "3d" ? "3D envelope view (drag rotate · right-drag pan · wheel zoom)" : "envelope plan view (drag pan · wheel zoom)");
  if (state.lastData) redrawStage();
}

/* Reset to the canonical instrument view: target at the ground-plane world
 * origin, default yaw/pitch/zoom. Never re-centers on the scene centroid. */
function recenterCamera() {
  state.planCamera = { panX: 0, panY: 0, zoom: 1 };
  state.camera.target = { x: 0, y: 0, z: 0 };
  state.camera.yaw = CAMERA_DEFAULT_YAW;
  state.camera.pitch = CAMERA_DEFAULT_PITCH;
  state.camera.zoom = CAMERA_DEFAULT_ZOOM;
  if (state.lastData) redrawStage();
}

function bindStageInteraction() {
  const stage = document.getElementById("stage");
  let drag = null;
  stage.addEventListener("pointerdown", e => {
    // never hijack drags/clicks on the stage chrome (e.g. the recenter button)
    if (e.target && e.target.closest && e.target.closest(".chrome-btn")) return;
    if (state.view === "topdown" && e.button !== 0) return;
    const pan = e.button === 2 || e.shiftKey;   // right-drag or Shift+left-drag pans
    const rect = document.getElementById("stageSvg").getBoundingClientRect();
    drag = {
      x: e.clientX, y: e.clientY,
      view: state.view,
      yaw: state.camera.yaw, pitch: state.camera.pitch,
      pan, panX: state.camera.target.x, panY: state.camera.target.y,
      planX: state.planCamera.panX, planY: state.planCamera.panY,
      planScale: 1000 / Math.max(1, Math.min(rect.width, rect.height)),
    };
    state.dragging = true;
    stage.classList.add("is-navigating");
    if (stage.setPointerCapture) stage.setPointerCapture(e.pointerId);
    if (e.button === 2 && e.preventDefault) e.preventDefault();
  });
  // suppress the context menu so right-drag pans cleanly
  stage.addEventListener("contextmenu", e => {
    if (state.view === "3d") e.preventDefault();
  });
  stage.addEventListener("pointermove", e => {
    if (!drag || state.view !== drag.view) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (drag.view === "topdown") {
      state.planCamera.panX = drag.planX + dx * drag.planScale;
      state.planCamera.panY = drag.planY + dy * drag.planScale;
    } else if (drag.pan) {
      // pan: the scene follows the drag; the TARGET persists across recomputes
      panByScreenDelta(-dx, -dy, { x: drag.panX, y: drag.panY });
    } else {
      // rotate with the "grab the object" feel: dragging right turns the
      // horizontal direction follows the live viewer convention requested by
      // MasterYip: right-drag increases yaw; dragging down tips the model's
      // top toward the viewer.
      state.camera.yaw = drag.yaw + dx * 0.006;
      state.camera.pitch = Math.max(-0.12, Math.min(1.35, drag.pitch - dy * 0.006));
    }
    redrawStage();
  });
  const endDrag = () => {
    if (drag) {
      drag = null;
      state.dragging = false;
      stage.classList.remove("is-navigating");
      if (state.lastData) startAnimation(state.lastData);
    }
  };
  stage.addEventListener("pointerup", endDrag);
  stage.addEventListener("pointercancel", endDrag);
  stage.addEventListener("wheel", e => {
    e.preventDefault();
    if (state.view === "3d") {
      state.camera.zoom = Math.max(0.4, Math.min(4.0, state.camera.zoom * (e.deltaY < 0 ? 1.1 : 0.9)));
    } else {
      state.planCamera.zoom = Math.max(0.4, Math.min(4.0, state.planCamera.zoom * (e.deltaY < 0 ? 1.1 : 0.9)));
    }
    redrawStage();
  }, { passive: false });
}

function bindControls() {
  document.getElementById("styleSelect").addEventListener("change", e => {
    applyStyle(e.target.value);
  });
  document.getElementById("computeLive").addEventListener("click", () => applyComputeMode("live"));
  document.getElementById("computeRelease").addEventListener("click", () => applyComputeMode("release"));
  bindSlider("aMargin", v => { state.aMargin = v; });
  bindSlider("aK", v => { state.aK = Math.round(v); });
  bindSlider("bWf", v => { state.env.w_f = v; });
  bindSlider("bWm", v => { state.env.w_m = v; });
  bindSlider("bWb", v => { state.env.w_b = v; });
  bindSlider("bXf", v => { state.env.x_f = v; });
  bindSlider("bXb", v => { state.env.x_b = v; });
  bindSlider("bMargin", v => { state.bMargin = v; });
  bindSlider("bK", v => { state.bK = Math.round(v); });

  document.getElementById("aLegacy").addEventListener("change", e => {
    state.aLegacy = e.target.checked;
    if (state.mode === "A") schedule();
    else if (state.lastData) redrawStage();   // legacy affects Mode A render only
  });
  document.getElementById("aReach").addEventListener("change", e => {
    state.aReach = e.target.checked;
    if (state.mode === "A") schedule();
  });
  document.getElementById("bRejected").addEventListener("change", e => {
    state.bRejected = e.target.checked;
    toggleArcVisibility();
  });
  document.getElementById("bAccessible").addEventListener("change", e => {
    state.bAccessible = e.target.checked;
    if ((state.mode === "B" || state.mode === "C") && state.lastData) redrawStage();
  });
  document.getElementById("bAnimate").addEventListener("change", e => {
    state.bAnimate = e.target.checked;
    if (!state.bAnimate) stopAnimation();
    else if (state.mode === "B" && state.lastData) startAnimation(state.lastData);
  });

  bindSegGroup("aCapsule", ".seg[data-capsule]", b => { state.capsule = b.dataset.capsule; syncCapsules(); schedule(); });
  bindSegGroup("bCapsule", ".seg[data-capsule]", b => { state.capsule = b.dataset.capsule; syncCapsules(); schedule(); });
  bindSegGroup("cCapsule", ".seg[data-capsule]", b => { state.capsule = b.dataset.capsule; syncCapsules(); schedule(); });

  bindSegGroup("bReference", ".seg[data-ref]", b => {
    state.bReference = b.dataset.ref;
    updateJointRackState();
    schedule();
  });

  // Mode C panel: envelope sliders share Mode B state (persist between B and C).
  bindSlider("cWf", v => { state.env.w_f = v; });
  bindSlider("cWm", v => { state.env.w_m = v; });
  bindSlider("cWb", v => { state.env.w_b = v; });
  bindSlider("cXf", v => { state.env.x_f = v; });
  bindSlider("cXb", v => { state.env.x_b = v; });
  bindSlider("cMargin", v => { state.bMargin = v; });
  bindSlider("cK", v => { state.bK = Math.round(v); });
  document.getElementById("cRejected").addEventListener("change", e => {
    state.bRejected = e.target.checked;
    toggleArcVisibility();
  });
  document.getElementById("cAccessible").addEventListener("change", e => {
    state.bAccessible = e.target.checked;
    if ((state.mode === "B" || state.mode === "C") && state.lastData) redrawStage();
  });
  document.getElementById("cAnimate").addEventListener("change", e => {
    state.bAnimate = e.target.checked;
    if (!state.bAnimate) stopAnimation();
    else if (state.mode === "B" && state.lastData) startAnimation(state.lastData);
  });

  document.getElementById("viewTop").addEventListener("click", () => setView("topdown"));
  document.getElementById("view3d").addEventListener("click", () => setView("3d"));
  const recenter = document.getElementById("recenter");
  if (recenter) recenter.addEventListener("click", () => recenterCamera());

  const presetRow = document.getElementById("presetRow");
  presetRow.addEventListener("click", ev => {
    const chip = ev.target.closest(".chip[data-preset]");
    if (!chip) return;
    presetRow.querySelectorAll(".chip").forEach(c => c.classList.toggle("active", c === chip));
    const p = PRESETS[chip.dataset.preset];
    state.q = p.q.slice();
    syncJointRack();
    state.aMargin = p.margin;
    document.getElementById("aMargin").value = p.margin;
    document.getElementById("aMarginVal").textContent = p.margin.toFixed(3);
    schedule();
  });

  document.getElementById("modeA").addEventListener("click", () => setMode("A"));
  document.getElementById("modeB").addEventListener("click", () => setMode("B"));
  document.getElementById("modeC").addEventListener("click", () => setMode("C"));
  bindStageInteraction();
}

function toggleArcVisibility() {
  const svg = document.getElementById("stageSvg");
  svg.querySelectorAll(".arc-rej").forEach(el => { el.style.display = state.bRejected ? "" : "none"; });
  svg.querySelectorAll(".trajectory-line").forEach(el => { el.style.display = state.bRejected ? "" : "none"; });
}

function setMode(m) {
  if (state.mode === m) return;
  state.mode = m;
  document.body.dataset.mode = m;
  refreshStageModeLabel();
  document.getElementById("modeA").classList.toggle("active", m === "A");
  document.getElementById("modeA").setAttribute("aria-selected", m === "A");
  document.getElementById("modeB").classList.toggle("active", m === "B");
  document.getElementById("modeB").setAttribute("aria-selected", m === "B");
  document.getElementById("modeC").classList.toggle("active", m === "C");
  document.getElementById("modeC").setAttribute("aria-selected", m === "C");
  document.getElementById("ctrlA").classList.toggle("hidden", m !== "A");
  document.getElementById("ctrlB").classList.toggle("hidden", m !== "B");
  document.getElementById("ctrlC").classList.toggle("hidden", m !== "C");
  // Modes B and C share the envelope state; sync the slider visuals so the
  // positions always match state.env / bMargin / bK after a mode switch.
  if (m === "B" || m === "C") syncEnvSliders();
  updateJointRackState();
  schedule();
}

/* Reflect shared Mode B/C envelope state onto the mode-B and mode-C sliders. */
function syncEnvSliders() {
  const sets = [
    ["bWf", state.env.w_f], ["bWm", state.env.w_m], ["bWb", state.env.w_b],
    ["bXf", state.env.x_f], ["bXb", state.env.x_b],
    ["cWf", state.env.w_f], ["cWm", state.env.w_m], ["cWb", state.env.w_b],
    ["cXf", state.env.x_f], ["cXb", state.env.x_b],
    ["bMargin", state.bMargin], ["cMargin", state.bMargin],
    ["bK", state.bK], ["cK", state.bK],
  ];
  for (const [id, val] of sets) {
    const el = document.getElementById(id);
    if (!el) continue;
    el.value = String(val);
    const out = document.getElementById(id + "Val");
    if (out) out.textContent = val.toFixed(el.step >= 1 ? 0 : 3);
  }
}

/* -------------------------------------------------------------- boot */
function boot() {
  applyStyle(preferredStyle(), false);
  applyComputeMode(preferredComputeMode(), false);
  buildJointRack();
  bindControls();
  syncCapsules();
  updateJointRackState();
  document.body.dataset.mode = state.mode;
  document.body.dataset.view = state.view;

  // Keep dense numerical diagnostics available, but begin collapsed on narrow
  // screens so the visualization and joint controls establish the hierarchy.
  const diagnostics = document.getElementById("diagnostics");
  if (diagnostics && window.matchMedia && window.matchMedia("(max-width: 640px)").matches) {
    diagnostics.removeAttribute("open");
  }

  const animCb = document.getElementById("bAnimate");
  if (reducedMotion() && animCb) {
    animCb.checked = false;
    animCb.disabled = true;
    state.bAnimate = false;
    const animCc = document.getElementById("cAnimate");
    if (animCc) { animCc.checked = false; animCc.disabled = true; }
  }

  document.getElementById("statusline").textContent =
    "ENV-DESIGN-003 occupied-body support envelope · P_K(q) = { p : u_kᵀp ≤ h_kᵒᶜᶜ(q) } · 19 capsule proxies (base + 6 legs × hip/thigh/shank) · URDF el_4090 · body-yaw frame +x forward / +y left · units m / rad · leg order LB LF LM RB RF RM";

  // start with the spider (resting) preset in Mode A
  state.q = PRESETS.spider.q.slice();
  syncJointRack();
  document.querySelectorAll("#presetRow .chip").forEach(c =>
    c.classList.toggle("active", c.dataset.preset === "spider"));
  schedule();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
})();
