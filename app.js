// @ts-check
// The store, its actions, and the effects that push it into the page and the 3D stage.
import { signal, computed, effect, batch, untracked, bind, series } from 'plainpanel';
import * as net from './net.js';
import { createStage } from './stage.js';

/** @typedef {import('./stage.js').Activations} Activations */

const EMPTY = /** @type {Activations} */ ({ input: null, hidden: null, output: null });
const PHASES = ['➊ input layer', '➋ hidden layer', '➌ output layer, then update'];

// ── store ──

const state = {
  params: signal(net.init()),
  act: signal(EMPTY),
  phase: signal(0), // the layer the next step computes: 0 input, 1 hidden, 2 output
  sample: signal(0), // which XOR row is going through
  epoch: signal(0),
  epochLosses: signal(/** @type {number[]} */ ([])), // this epoch's sample losses so far
  last: signal(/** @type {{ base: net.Params, step: net.Params } | null} */ (null)), // the last update, for the α preview
  playing: signal(false),
};
const train = {
  alpha: signal(1), // learning rate
  speed: signal(1), // layer steps per frame while playing
};
const ui = {
  labels: signal(true),
  noCss: signal(false),
};
const sampleLoss = series(4000);
const epochLoss = series(1000);

const lastOf = (/** @type {readonly (number | null)[]} */ xs) => xs.at(-1) ?? null;
const fmt = (/** @type {number | null} */ v, digits = 4) => (v === null ? '–' : v.toFixed(digits));

const views = {
  epoch: computed(() => String(state.epoch())),
  next: computed(() => PHASES[state.phase()]),
  sample: computed(() => {
    const s = net.XOR[state.sample()];
    return `(${s.inputs.join(', ')}) → ${s.target}`;
  }),
  alpha: computed(() => train.alpha().toFixed(1)),
  speed: computed(() => `${train.speed()} / frame`),
  sampleLoss: computed(() => fmt(lastOf(sampleLoss.read()))),
  epochLoss: computed(() => fmt(lastOf(epochLoss.read()))),
  rows: computed(() =>
    net.XOR.map((s, k) => {
      const out = net.forward(state.params(), s.inputs).output;
      return {
        id: k,
        here: k === state.sample() ? '▶' : '',
        x1: String(s.inputs[0]),
        x2: String(s.inputs[1]),
        target: String(s.target),
        out: out.toFixed(3),
        ok: Math.round(out) === s.target ? '✓' : '✗',
      };
    }),
  ),
  solved: computed(() => {
    const right = views.rows().filter(r => r.ok === '✓').length;
    return right === 4 ? '✓ all four right' : `${right} of 4 right`;
  }),
  stepLocked: computed(() => state.playing()),
  pauseLocked: computed(() => !state.playing()),
};

// ── training ──

/** Compute one layer of the current sample; the output layer also trains on it. */
function stepLayer() {
  const k = state.sample();
  const s = net.XOR[k];
  const p = state.params();
  if (state.phase() === 0) {
    state.act({ input: s.inputs, hidden: null, output: null });
    state.phase(1);
    return;
  }
  if (state.phase() === 1) {
    state.act({ input: s.inputs, hidden: net.hiddenLayer(p, s.inputs), output: null });
    state.phase(2);
    return;
  }
  const f = net.forward(p, s.inputs);
  const l = net.loss(s.target, f.output);
  const step = net.gradients(p, s);
  state.act(f);
  sampleLoss.push(l);
  state.last({ base: p, step });
  state.params(net.apply(p, step, train.alpha()));
  state.phase(0);
  state.sample((k + 1) % net.XOR.length);
  const losses = [...state.epochLosses(), l];
  if (losses.length === net.XOR.length) {
    epochLoss.push(losses.reduce((a, b) => a + b) / losses.length);
    state.epochLosses([]);
    state.epoch(state.epoch() + 1);
  } else {
    state.epochLosses(losses);
  }
}

function stepSample() {
  do stepLayer();
  while (state.phase() !== 0);
}

const actions = {
  layer: () => batch(stepLayer),
  sample: () => batch(stepSample),
  epoch: () =>
    batch(() => {
      const start = state.epoch();
      do stepSample();
      while (state.epoch() === start);
    }),
  play: () => state.playing(true),
  pause: () => state.playing(false),
  reset: () =>
    batch(() => {
      state.playing(false);
      state.params(net.init());
      state.act(EMPTY);
      state.phase(0);
      state.sample(0);
      state.epoch(0);
      state.epochLosses([]);
      state.last(null);
      sampleLoss.clear();
      epochLoss.clear();
    }),
};

// play: `speed` layer steps per animation frame, one batch each
effect(() => {
  if (!state.playing()) return;
  let id = 0;
  const tick = () => {
    batch(() => {
      for (let k = untracked(train.speed); k > 0; k--) stepLayer();
    });
    id = requestAnimationFrame(tick);
  };
  id = requestAnimationFrame(tick);
  return () => cancelAnimationFrame(id);
});

// α preview: while paused, moving α re-applies the last update at the new rate
effect(() => {
  const alpha = train.alpha();
  untracked(() => {
    const last = state.last();
    if (last && !state.playing()) state.params(net.apply(last.base, last.step, alpha));
  });
});

// ── page ──

bind(document.body, { state, train, ui, views, actions });

const stage = createStage(/** @type {HTMLElement} */ (document.getElementById('scene')));
effect(() => stage.setModel(state.params()));
effect(() => stage.setActivations(state.act()));
effect(() => stage.setLabels(ui.labels()));

effect(() => {
  const off = ui.noCss(); // "no CSS": the same HTML, zero styling
  for (const el of document.querySelectorAll('link[rel=stylesheet], style')) {
    /** @type {HTMLLinkElement | HTMLStyleElement} */ (el).disabled = off;
  }
});

// loss chart: sample losses (thin, blue) and epoch means (thick, red), each across the full width
const chart = /** @type {HTMLCanvasElement} */ (document.getElementById('loss'));
const chartBox = /** @type {HTMLElement} */ (chart.parentElement);
const chartSize = signal({ w: 0, h: 0 });
new ResizeObserver(() => chartSize({ w: chartBox.clientWidth, h: chartBox.clientHeight })).observe(chartBox);
effect(() => {
  ui.noCss(); // redraw when the stylesheets go on or off
  const samples = sampleLoss.read();
  const epochs = epochLoss.read();
  // with the page's CSS the canvas fills its box; without it the canvas is in flow at its own pixel size
  const inFlow = getComputedStyle(chart).position !== 'absolute';
  const w = chartSize().w;
  const h = inFlow ? 120 : chartSize().h;
  const dpr = inFlow ? 1 : devicePixelRatio;
  if (!w || !h) return;
  chart.width = Math.round(w * dpr);
  chart.height = Math.round(h * dpr);
  const ctx = /** @type {CanvasRenderingContext2D} */ (chart.getContext('2d'));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.strokeStyle = 'rgba(128, 128, 128, 0.3)';
  ctx.lineWidth = 0.5;
  for (let i = 1; i < 10; i++) {
    ctx.beginPath();
    ctx.moveTo(0, (i / 10) * h);
    ctx.lineTo(w, (i / 10) * h);
    ctx.stroke();
  }
  const max = Math.min(1, Math.max(1e-6, ...samples.map(v => v ?? 0), ...epochs.map(v => v ?? 0)));
  /** @param {readonly (number | null)[]} values @param {string} color @param {number} width */
  const line = (values, color, width) => {
    if (values.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    values.forEach((v, i) => {
      const x = (i / (values.length - 1)) * w;
      const y = h - ((v ?? 0) / max) * (h - 2) - 1;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  };
  line(samples, '#2a9df4', 1);
  line(epochs, '#f0605a', 2);
});
