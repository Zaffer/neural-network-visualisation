// @ts-check
// A 2-2-1 sigmoid network learning XOR by backpropagation, as pure functions.
// Parameters are plain immutable objects: every update returns a new one, so
// the store can hold them in a signal and the α preview can re-apply the last
// gradient to the weights it started from.

/** @typedef {{ w1: number[][], b1: number[], w2: number[], b2: number }} Params  w1[j][i]: input i → hidden j */
/** @typedef {{ input: number[], hidden: number[], output: number }} Forward */
/** @typedef {{ inputs: number[], target: number }} Sample */

/** @type {readonly Sample[]} */
export const XOR = Object.freeze([
  { inputs: [0, 0], target: 0 },
  { inputs: [0, 1], target: 1 },
  { inputs: [1, 0], target: 1 },
  { inputs: [1, 1], target: 0 },
]);

const rand = () => Math.random() * 2 - 1;
const sigmoid = (/** @type {number} */ x) => 1 / (1 + Math.exp(-x));

/** @returns {Params} random weights and biases in [-1, 1] */
export function init() {
  return { w1: [[rand(), rand()], [rand(), rand()]], b1: [rand(), rand()], w2: [rand(), rand()], b2: rand() };
}

/** @param {Params} p @param {number[]} input @returns {number[]} */
export function hiddenLayer(p, input) {
  return p.w1.map((w, j) => sigmoid(input[0] * w[0] + input[1] * w[1] + p.b1[j]));
}

/** @param {Params} p @param {number[]} hidden @returns {number} */
export function outputLayer(p, hidden) {
  return sigmoid(hidden[0] * p.w2[0] + hidden[1] * p.w2[1] + p.b2);
}

/** @param {Params} p @param {number[]} input @returns {Forward} */
export function forward(p, input) {
  const hidden = hiddenLayer(p, input);
  return { input, hidden, output: outputLayer(p, hidden) };
}

/** Squared error. @param {number} target @param {number} output */
export const loss = (target, output) => (target - output) ** 2;

/**
 * The step that lowers the loss on one sample, before scaling by the learning rate.
 * @param {Params} p @param {Sample} sample @returns {Params}
 */
export function gradients(p, { inputs, target }) {
  const { hidden, output } = forward(p, inputs);
  const outDelta = (target - output) * output * (1 - output);
  const hidDelta = p.w2.map((w, j) => w * outDelta * hidden[j] * (1 - hidden[j]));
  return {
    w1: hidDelta.map(d => [d * inputs[0], d * inputs[1]]),
    b1: hidDelta,
    w2: hidden.map(h => outDelta * h),
    b2: outDelta,
  };
}

/** p + rate · step. @param {Params} p @param {Params} step @param {number} rate @returns {Params} */
export function apply(p, step, rate) {
  return {
    w1: p.w1.map((w, j) => [w[0] + rate * step.w1[j][0], w[1] + rate * step.w1[j][1]]),
    b1: p.b1.map((b, j) => b + rate * step.b1[j]),
    w2: p.w2.map((w, j) => w + rate * step.w2[j]),
    b2: p.b2 + rate * step.b2,
  };
}
