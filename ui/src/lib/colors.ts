import type { GtLayer, LayerMode, LoadedScene, PredictionLayer, SimilarityDisplay } from '../types';

const NEUTRAL = [114, 122, 132] as const;
const DIM = [42, 47, 55] as const;
const SELECTED = [255, 224, 102] as const;

function hashInt(value: number): number {
  let x = Math.imul(value + 0x9e3779b9, 0x85ebca6b);
  x ^= x >>> 13;
  x = Math.imul(x, 0xc2b2ae35);
  return (x ^ (x >>> 16)) >>> 0;
}

export function colorForLabel(label: number, salt = 0): [number, number, number] {
  if (label < 0) return [...NEUTRAL];
  const h = hashInt(label + salt * 1000003);
  return [
    40 + (h & 0x7f),
    70 + ((h >> 8) & 0x9f),
    80 + ((h >> 16) & 0x8f),
  ];
}

export function colorForScalar(value: number, min = 0, max = 1): [number, number, number] {
  const t = Math.max(0, Math.min(1, (value - min) / Math.max(max - min, 1e-6)));
  const r = Math.round(30 + 225 * t);
  const g = Math.round(70 + 150 * Math.sin(t * Math.PI));
  const b = Math.round(210 - 180 * t);
  return [r, g, b];
}

function writeColor(out: Uint8Array, dst: number, rgb: readonly number[]): void {
  out[dst] = rgb[0];
  out[dst + 1] = rgb[1];
  out[dst + 2] = rgb[2];
}

function heightRange(points: Float32Array, indices: Uint32Array): [number, number] {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < indices.length; i += 1) {
    const z = points[indices[i] * 3 + 2];
    min = Math.min(min, z);
    max = Math.max(max, z);
  }
  return [min, max];
}

export type ColorOptions = {
  mode: LayerMode;
  granularity: string;
  gtKey?: string;
  sourceIndices: Uint32Array;
  selectedSourceIndex?: number | null;
  isolate?: { kind: 'pseudo' | 'prediction' | 'gt'; key: string; label: number } | null;
  similarities?: Float32Array | null;
  similarityThreshold?: number;
  similarityDisplay?: SimilarityDisplay;
};

function labelsForMode(scene: LoadedScene, options: ColorOptions): Int32Array | null {
  if (options.mode === 'pseudo') return scene.pseudoLabels[options.granularity] ?? null;
  if (options.mode === 'prediction') return scene.predictions[options.granularity]?.labels ?? null;
  if (options.mode === 'gt' && options.gtKey) return scene.gt[options.gtKey]?.labels ?? null;
  return null;
}

function predictionForMode(scene: LoadedScene, options: ColorOptions): PredictionLayer | null {
  return options.mode === 'score' ? scene.predictions[options.granularity] ?? null : null;
}

function gtForMode(scene: LoadedScene, options: ColorOptions): GtLayer | null {
  return options.mode === 'gt' && options.gtKey ? scene.gt[options.gtKey] ?? null : null;
}

export function composePointColors(scene: LoadedScene, options: ColorOptions): Uint8Array {
  const indices = options.sourceIndices;
  const out = new Uint8Array(indices.length * 3);
  const labels = labelsForMode(scene, options);
  const pred = predictionForMode(scene, options);
  const gt = gtForMode(scene, options);
  const [zMin, zMax] = options.mode === 'height' ? heightRange(scene.points, indices) : [0, 1];

  for (let i = 0; i < indices.length; i += 1) {
    const sourceIndex = indices[i];
    const dst = i * 3;
    let rgb: readonly number[] = NEUTRAL;

    if (options.mode === 'rgb') {
      rgb = [
        scene.colors[sourceIndex * 3],
        scene.colors[sourceIndex * 3 + 1],
        scene.colors[sourceIndex * 3 + 2],
      ];
    } else if (options.mode === 'height') {
      rgb = colorForScalar(scene.points[sourceIndex * 3 + 2], zMin, zMax);
    } else if (labels) {
      rgb = colorForLabel(labels[sourceIndex], options.mode === 'prediction' ? 17 : gt ? 29 : 0);
    } else if (pred?.scores) {
      rgb = pred.labels[sourceIndex] < 0 ? DIM : colorForScalar(pred.scores[sourceIndex], 0, 1);
    } else if (options.mode === 'similarity' && options.similarities) {
      const sim = options.similarities[sourceIndex];
      const pass = sim >= (options.similarityThreshold ?? 0.5);
      rgb = options.similarityDisplay === 'binary'
        ? (pass ? SELECTED : DIM)
        : (pass ? colorForScalar(sim, options.similarityThreshold ?? 0.5, 1.0) : DIM);
    }

    if (options.isolate) {
      let isolateLabel: number | null = null;
      if (options.isolate.kind === 'pseudo') isolateLabel = scene.pseudoLabels[options.isolate.key]?.[sourceIndex] ?? null;
      if (options.isolate.kind === 'prediction') isolateLabel = scene.predictions[options.isolate.key]?.labels[sourceIndex] ?? null;
      if (options.isolate.kind === 'gt') isolateLabel = scene.gt[options.isolate.key]?.labels[sourceIndex] ?? null;
      if (isolateLabel !== options.isolate.label) rgb = DIM;
    }

    if (sourceIndex === options.selectedSourceIndex) rgb = SELECTED;
    writeColor(out, dst, rgb);
  }
  return out;
}

export function selectedInstanceFor(scene: LoadedScene, sourceIndex: number, kind: 'pseudo' | 'prediction' | 'gt', key: string) {
  const label =
    kind === 'pseudo'
      ? scene.pseudoLabels[key]?.[sourceIndex]
      : kind === 'prediction'
        ? scene.predictions[key]?.labels[sourceIndex]
        : scene.gt[key]?.labels[sourceIndex];
  if (label === undefined || label < 0) return null;
  return { kind, key, label };
}

