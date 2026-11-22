// @ts-check
// The 3D network: plain imperative three.js behind a narrow API, no signals in
// here. The store pushes parameters and activations in; nothing comes out.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

/** @typedef {import('./net.js').Params} Params */
/** @typedef {{ input: number[] | null, hidden: number[] | null, output: number | null }} Activations  null: not computed yet */
/**
 * @typedef {object} Stage
 * @property {(p: Params) => void} setModel  weights → connections, biases → node size and colour
 * @property {(a: Activations) => void} setActivations  activation → node opacity
 * @property {(on: boolean) => void} setLabels  numbers beside every node and connection
 * @property {() => void} dispose
 */

const POSITIVE = new THREE.Color('#2a9df4');
const NEGATIVE = new THREE.Color('#f0605a');
const INPUT = new THREE.Color('#888888');

// layout: x is the layer, y the neuron
const POS = {
  input: [new THREE.Vector3(-4, 2, 0), new THREE.Vector3(-4, -2, 0)],
  hidden: [new THREE.Vector3(0, 2, 0), new THREE.Vector3(0, -2, 0)],
  output: [new THREE.Vector3(4, 0, 0)],
};

const fmt = (/** @type {number | null} */ v) => (v === null ? '–' : v.toFixed(2));
// size and thickness grow with magnitude but level off, so a trained network (|w| ≈ 8) stays legible
const nodeSize = (/** @type {number} */ bias) => 0.15 + 0.45 * Math.tanh(Math.abs(bias) / 4);
const thickness = (/** @type {number} */ w) => 0.015 + 0.13 * Math.tanh(Math.abs(w) / 4);
const opacity = (/** @type {number | null} */ a) => (a === null ? 0.2 : a * 0.8 + 0.2);

/**
 * @param {HTMLElement} container  the stage fills it and follows its size
 * @returns {Stage}
 */
export function createStage(container) {
  /** @type {THREE.WebGLRenderer} */
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  } catch (err) {
    console.warn('stage: WebGL unavailable, running without 3D —', err instanceof Error ? err.message : err);
    container.textContent = 'The 3D view needs WebGL. The panels still train and show every number.';
    const noop = () => {};
    return { setModel: noop, setActivations: noop, setLabels: noop, dispose: noop };
  }
  renderer.setPixelRatio(devicePixelRatio);
  renderer.setClearColor(0x000000, 0); // transparent: the page's panel colour shows through, light or dark
  renderer.domElement.style.display = 'block';
  container.style.position = 'relative';
  container.append(renderer.domElement);

  // the labels are a DOM layer laid over the canvas, as three.js does it; inline, so it holds with stylesheets off
  const labelRenderer = new CSS2DRenderer();
  Object.assign(labelRenderer.domElement.style, { position: 'absolute', top: '0', left: '0', pointerEvents: 'none' });
  container.append(labelRenderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
  const controls = new OrbitControls(camera, renderer.domElement);

  let queued = false;
  const render = () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      renderer.render(scene, camera);
      labelRenderer.render(scene, camera);
    });
  };
  controls.addEventListener('change', render);

  /** @param {string} cls */
  const label = cls => {
    const div = document.createElement('span');
    div.className = cls;
    const obj = new CSS2DObject(div);
    scene.add(obj);
    return obj;
  };
  const setText = (/** @type {CSS2DObject} */ obj, /** @type {string} */ text) => {
    if (obj.element.textContent !== text) obj.element.textContent = text;
  };

  // nodes: one unit sphere each, scaled to the bias
  const sphere = new THREE.SphereGeometry(1, 24, 16);
  /** @param {THREE.Vector3} at */
  const node = at => {
    const mesh = new THREE.Mesh(sphere, new THREE.MeshBasicMaterial({ transparent: true }));
    mesh.position.copy(at);
    scene.add(mesh);
    return { mesh, value: label('lbl'), bias: label('lbl bias') };
  };
  const nodes = { input: POS.input.map(node), hidden: POS.hidden.map(node), output: POS.output.map(node) };

  // connections: one unit cylinder each, stretched between two nodes and thickened by |weight|
  const cylinder = new THREE.CylinderGeometry(1, 1, 1, 10, 1);
  const up = new THREE.Vector3(0, 1, 0);
  /** @param {THREE.Vector3} from @param {THREE.Vector3} to */
  const edge = (from, to) => {
    const mesh = new THREE.Mesh(cylinder, new THREE.MeshBasicMaterial({ transparent: true }));
    const dir = to.clone().sub(from);
    mesh.position.copy(from).lerp(to, 0.5);
    mesh.quaternion.setFromUnitVectors(up, dir.clone().normalize());
    mesh.userData.length = dir.length();
    scene.add(mesh);
    const weight = label('lbl weight');
    weight.position.copy(from).lerp(to, 0.25).add(new THREE.Vector3(0, 0.35, 0));
    return { mesh, weight };
  };
  // w1[j][i]: input i → hidden j
  const edges1 = POS.hidden.map(h => POS.input.map(i => edge(i, h)));
  const edges2 = POS.hidden.map(h => edge(h, POS.output[0]));

  /** @param {{ mesh: THREE.Mesh, weight: CSS2DObject }} e @param {number} w */
  const setEdge = (e, w) => {
    const t = thickness(w);
    e.mesh.scale.set(t, e.mesh.userData.length, t);
    const m = /** @type {THREE.MeshBasicMaterial} */ (e.mesh.material);
    m.color.copy(w >= 0 ? POSITIVE : NEGATIVE);
    m.opacity = Math.min(Math.abs(w) + 0.2, 1);
    setText(e.weight, 'w ' + w.toFixed(2));
  };
  /** @param {{ mesh: THREE.Mesh, value: CSS2DObject, bias: CSS2DObject }} n @param {number | null} bias */
  const setNode = (n, bias) => {
    const r = bias === null ? 0.2 : nodeSize(bias);
    n.mesh.scale.setScalar(r);
    /** @type {THREE.MeshBasicMaterial} */ (n.mesh.material).color.copy(bias === null ? INPUT : bias >= 0 ? POSITIVE : NEGATIVE);
    n.value.position.copy(n.mesh.position).add(new THREE.Vector3(0, r + 0.35, 0));
    n.bias.position.copy(n.mesh.position).add(new THREE.Vector3(0, -r - 0.35, 0));
    n.bias.visible = bias !== null;
    if (bias !== null) setText(n.bias, 'b ' + bias.toFixed(2));
  };

  // fit the whole network in view at any aspect ratio
  // With the page's CSS the canvas is out of flow and fills the container. Without it the canvas is in
  // flow and would grow the container it is measured from, so its height comes from the width instead.
  const fit = () => {
    const w = container.clientWidth;
    const inFlow = getComputedStyle(renderer.domElement).position !== 'absolute';
    const h = inFlow ? Math.round(w * 0.6) : container.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h);
    labelRenderer.setSize(w, h);
    camera.aspect = w / h;
    const tan = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    const dist = Math.max(3.4 / tan, 5.6 / (tan * camera.aspect));
    camera.position.set(0, 0, dist);
    camera.updateProjectionMatrix();
    controls.target.set(0, 0, 0);
    controls.update();
    render();
  };
  const observer = new ResizeObserver(fit);
  observer.observe(container);

  return {
    setModel(p) {
      nodes.input.forEach(n => setNode(n, null));
      nodes.hidden.forEach((n, j) => setNode(n, p.b1[j]));
      setNode(nodes.output[0], p.b2);
      edges1.forEach((row, j) => row.forEach((e, i) => setEdge(e, p.w1[j][i])));
      edges2.forEach((e, j) => setEdge(e, p.w2[j]));
      render();
    },
    setActivations(a) {
      const set = (/** @type {ReturnType<typeof node>} */ n, /** @type {number | null} */ v) => {
        /** @type {THREE.MeshBasicMaterial} */ (n.mesh.material).opacity = opacity(v);
        setText(n.value, fmt(v));
      };
      nodes.input.forEach((n, i) => set(n, a.input?.[i] ?? null));
      nodes.hidden.forEach((n, j) => set(n, a.hidden?.[j] ?? null));
      set(nodes.output[0], a.output);
      render();
    },
    setLabels(on) {
      labelRenderer.domElement.hidden = !on;
      render();
    },
    dispose() {
      observer.disconnect();
      controls.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      labelRenderer.domElement.remove();
    },
  };
}
