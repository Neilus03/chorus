import { computeCosineSimilarities } from '../lib/featureSimilarity';

type RequestMessage = {
  type: 'init';
  source: string;
  features: Float32Array;
  dim: number;
  normalized: boolean;
} | {
  type: 'compute';
  source: string;
  queryIndex: number;
};

let loadedSource: string | null = null;
let loadedFeatures: Float32Array | null = null;
let loadedDim = 0;
let loadedNormalized = false;

self.onmessage = (event: MessageEvent<RequestMessage>) => {
  const msg = event.data;
  if (msg.type === 'init') {
    loadedSource = msg.source;
    loadedFeatures = msg.features;
    loadedDim = msg.dim;
    loadedNormalized = msg.normalized;
    self.postMessage({ type: 'ready', source: loadedSource });
    return;
  }
  if (msg.type !== 'compute') return;
  try {
    if (!loadedFeatures || loadedSource !== msg.source) {
      throw new Error(`Feature source is not loaded in worker: ${msg.source}`);
    }
    const similarities = computeCosineSimilarities(
      loadedFeatures,
      loadedDim,
      msg.queryIndex,
      loadedNormalized,
    );
    self.postMessage({ type: 'result', similarities });
  } catch (err) {
    self.postMessage({ type: 'error', error: err instanceof Error ? err.message : String(err) });
  }
};
