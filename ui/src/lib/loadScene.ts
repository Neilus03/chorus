import type {
  ArtifactRef,
  FeatureRef,
  GtLayer,
  InspectionManifest,
  LoadedFeature,
  LoadedScene,
  NpyArray,
  PredictionLayer,
  QueryRow,
} from '../types';
import { parseNpy, toFloat32, toInt32, toUint8Colors } from './npy';

function fileUrl(path: string): string {
  return `/_chorus/file?path=${encodeURIComponent(path)}`;
}

function resolveUrl(path: string): string {
  return `/_chorus/resolve?path=${encodeURIComponent(path)}`;
}

function artifactPath(ref: ArtifactRef | string | undefined): string | null {
  if (!ref) return null;
  return typeof ref === 'string' ? ref : ref.path;
}

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(fileUrl(path));
  if (!res.ok) throw new Error(`Failed to fetch ${path}: ${res.status} ${await res.text()}`);
  return res.json() as Promise<T>;
}

async function fetchNpy(ref: ArtifactRef | string): Promise<NpyArray> {
  const path = artifactPath(ref);
  if (!path) throw new Error('Missing artifact path');
  const res = await fetch(fileUrl(path));
  if (!res.ok) throw new Error(`Failed to fetch ${path}: ${res.status} ${await res.text()}`);
  return parseNpy(await res.arrayBuffer());
}

function requireShape(array: NpyArray, shape: number[], name: string): void {
  if (array.shape.length !== shape.length || array.shape.some((v, i) => v !== shape[i])) {
    throw new Error(`${name} shape [${array.shape.join(', ')}], expected [${shape.join(', ')}]`);
  }
}

function defaultColors(numPoints: number): Uint8Array {
  const out = new Uint8Array(numPoints * 3);
  out.fill(178);
  return out;
}

function normalizePredLayer(layer: PredictionLayer, numPoints: number, key: string): PredictionLayer {
  if (layer.labels.length !== numPoints) {
    throw new Error(`Prediction labels for ${key} have ${layer.labels.length} points, expected ${numPoints}`);
  }
  if (layer.scores && layer.scores.length !== numPoints) {
    throw new Error(`Prediction scores for ${key} have ${layer.scores.length} points, expected ${numPoints}`);
  }
  if (layer.queryIds && layer.queryIds.length !== numPoints) {
    throw new Error(`Prediction query ids for ${key} have ${layer.queryIds.length} points, expected ${numPoints}`);
  }
  return layer;
}

export async function resolveManifest(path: string): Promise<InspectionManifest> {
  const res = await fetch(resolveUrl(path));
  if (!res.ok) throw new Error(`Failed to resolve ${path}: ${res.status} ${await res.text()}`);
  const manifest = await res.json() as InspectionManifest;
  if (manifest.schema_version !== 'chorus_inspection_bundle/v1') {
    throw new Error(`Unsupported manifest schema: ${String(manifest.schema_version)}`);
  }
  return manifest;
}

export async function loadSceneFromPath(path: string): Promise<LoadedScene> {
  const manifest = await resolveManifest(path);
  return loadSceneFromManifest(manifest);
}

export async function loadSceneFromManifest(manifest: InspectionManifest): Promise<LoadedScene> {
  const pointsNpy = await fetchNpy(manifest.arrays.points);
  if (pointsNpy.shape.length !== 2 || pointsNpy.shape[1] !== 3) {
    throw new Error(`points.npy must have shape [N, 3], got [${pointsNpy.shape.join(', ')}]`);
  }
  const numPoints = pointsNpy.shape[0];
  const points = toFloat32(pointsNpy);

  let colors = defaultColors(numPoints);
  if (manifest.arrays.colors) {
    colors = toUint8Colors(await fetchNpy(manifest.arrays.colors), numPoints);
  }

  let normals: Float32Array | undefined;
  if (manifest.arrays.normals) {
    const normalsNpy = await fetchNpy(manifest.arrays.normals);
    requireShape(normalsNpy, [numPoints, 3], 'normals.npy');
    normals = toFloat32(normalsNpy);
  }

  const pseudoLabels: LoadedScene['pseudoLabels'] = {};
  await Promise.all(
    Object.entries(manifest.labels.pseudo ?? {}).map(async ([key, ref]) => {
      const labels = toInt32(await fetchNpy(ref));
      if (labels.length !== numPoints) throw new Error(`Pseudo labels for ${key} have ${labels.length} points, expected ${numPoints}`);
      pseudoLabels[key] = labels;
    }),
  );

  const predictions: Record<string, PredictionLayer> = {};
  await Promise.all(
    Object.entries(manifest.labels.predictions ?? {}).map(async ([key, predRef]) => {
      const labels = toInt32(await fetchNpy(predRef.pred_labels));
      const scores = predRef.pred_scores ? toFloat32(await fetchNpy(predRef.pred_scores)) : undefined;
      const queryIds = predRef.pred_query_ids ? toInt32(await fetchNpy(predRef.pred_query_ids)) : undefined;
      let queryTable: QueryRow[] | undefined;
      const queryPath = artifactPath(predRef.query_table);
      if (queryPath) queryTable = await fetchJson<QueryRow[]>(queryPath);
      predictions[key] = normalizePredLayer({ labels, scores, queryIds, queryTable }, numPoints, key);
    }),
  );

  const gt: Record<string, GtLayer> = {};
  await Promise.all(
    Object.entries(manifest.labels.gt ?? {}).map(async ([key, gtRef]) => {
      const labels = toInt32(await fetchNpy(gtRef.labels));
      if (labels.length !== numPoints) throw new Error(`GT labels for ${key} have ${labels.length} points, expected ${numPoints}`);
      const classesPath = artifactPath(gtRef.instance_classes);
      gt[key] = {
        labels,
        instanceClasses: classesPath ? await fetchJson<Record<string, number>>(classesPath) : undefined,
      };
    }),
  );

  return {
    manifest,
    sceneId: manifest.scene_id,
    trainingPackDir: manifest.training_pack_dir,
    numPoints,
    points,
    colors,
    normals,
    pseudoLabels,
    predictions,
    gt,
    featureRefs: manifest.features ?? {},
    loadedFeatures: {},
    warnings: manifest.warnings ?? [],
  };
}

export async function loadFeatureSource(scene: LoadedScene, sourceName: string): Promise<LoadedFeature> {
  const ref: FeatureRef | undefined = scene.featureRefs[sourceName];
  if (!ref) throw new Error(`Feature source not found: ${sourceName}`);
  const npy = await fetchNpy(ref);
  if (npy.shape.length !== 2 || npy.shape[0] !== scene.numPoints) {
    throw new Error(`Feature source ${sourceName} shape [${npy.shape.join(', ')}], expected [${scene.numPoints}, D]`);
  }
  const values = toFloat32(npy);
  return {
    name: sourceName,
    values,
    dim: Number(ref.dim ?? npy.shape[1]),
    normalized: Boolean(ref.normalized),
  };
}
