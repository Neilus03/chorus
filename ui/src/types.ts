export type GranularityKey = string;

export type ArtifactRef = {
  path: string;
  dtype?: string;
  shape?: number[];
  [key: string]: unknown;
};

export type PredictionRef = {
  pred_labels: ArtifactRef;
  pred_scores?: ArtifactRef;
  pred_query_ids?: ArtifactRef;
  query_table?: ArtifactRef;
  query_embed?: ArtifactRef;
};

export type GtRef = {
  labels: ArtifactRef;
  instance_classes?: ArtifactRef;
};

export type FeatureRef = ArtifactRef & {
  dim?: number | null;
  normalized?: boolean;
};

export type InspectionManifest = {
  schema_version: 'chorus_inspection_bundle/v1';
  scene_id: string;
  dataset?: string | null;
  created_at?: string;
  training_pack_dir: string;
  source?: Record<string, unknown>;
  arrays: {
    points: ArtifactRef;
    colors?: ArtifactRef;
    normals?: ArtifactRef;
  };
  labels: {
    pseudo: Record<GranularityKey, ArtifactRef>;
    predictions?: Record<GranularityKey, PredictionRef>;
    gt?: Record<string, GtRef>;
  };
  features?: Record<string, FeatureRef>;
  defaults?: {
    granularity?: GranularityKey | null;
    score_threshold?: number;
    mask_threshold?: number;
    min_points?: number;
  };
  warnings?: string[];
};

export type TypedNpyData =
  | Uint8Array
  | Int8Array
  | Int16Array
  | Int32Array
  | Float32Array
  | Float64Array;

export type NpyArray = {
  descr: string;
  fortranOrder: boolean;
  shape: number[];
  data: TypedNpyData;
};

export type PredictionLayer = {
  labels: Int32Array;
  scores?: Float32Array;
  queryIds?: Int32Array;
  queryTable?: QueryRow[];
};

export type QueryRow = {
  query_id: number;
  score_probability?: number;
  mask_area?: number;
  kept?: boolean;
  instance_label?: number | null;
  [key: string]: unknown;
};

export type GtLayer = {
  labels: Int32Array;
  instanceClasses?: Record<string, number | string>;
};

export type LoadedFeature = {
  name: string;
  values: Float32Array;
  dim: number;
  normalized: boolean;
};

export type LoadedScene = {
  manifest: InspectionManifest;
  sceneId: string;
  trainingPackDir: string;
  numPoints: number;
  points: Float32Array;
  colors: Uint8Array;
  normals?: Float32Array;
  pseudoLabels: Record<GranularityKey, Int32Array>;
  predictions: Record<GranularityKey, PredictionLayer>;
  gt: Record<string, GtLayer>;
  featureRefs: Record<string, FeatureRef>;
  loadedFeatures: Record<string, LoadedFeature>;
  warnings: string[];
};

export type LayerMode = 'rgb' | 'height' | 'pseudo' | 'prediction' | 'gt' | 'score' | 'similarity';

export type SimilarityDisplay = 'heatmap' | 'binary';

export type SimilarityMetric = 'similarity' | 'distance';

export type RenderSelection = {
  sourceIndex: number;
};
