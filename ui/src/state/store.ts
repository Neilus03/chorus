import { create } from 'zustand';
import type {
  LayerMode,
  LoadedFeature,
  LoadedScene,
  RenderSelection,
  SimilarityDisplay,
  SimilarityMetric,
} from '../types';

type IsolateState = { kind: 'pseudo' | 'prediction' | 'gt'; key: string; label: number } | null;

type UiState = {
  scene: LoadedScene | null;
  loadPath: string;
  loading: boolean;
  error: string | null;
  layerMode: LayerMode;
  granularity: string;
  gtKey: string | null;
  featureSource: string | null;
  pointSize: number;
  maxRenderPoints: number;
  selected: RenderSelection | null;
  isolate: IsolateState;
  similarities: Float32Array | null;
  similarityLoading: boolean;
  similarityThreshold: number;
  similarityDisplay: SimilarityDisplay;
  similarityMetric: SimilarityMetric;
  configPath: string;
  checkpointPath: string;
  predictionRunning: boolean;
  setLoadPath: (value: string) => void;
  setLoading: (value: boolean) => void;
  setError: (value: string | null) => void;
  setScene: (scene: LoadedScene | null) => void;
  setLoadedFeature: (feature: LoadedFeature) => void;
  setLayerMode: (mode: LayerMode) => void;
  setGranularity: (key: string) => void;
  setGtKey: (key: string | null) => void;
  setFeatureSource: (key: string | null) => void;
  setPointSize: (value: number) => void;
  setMaxRenderPoints: (value: number) => void;
  setSelected: (selected: RenderSelection | null) => void;
  setIsolate: (value: IsolateState) => void;
  setSimilarities: (values: Float32Array | null) => void;
  setSimilarityLoading: (value: boolean) => void;
  setSimilarityThreshold: (value: number) => void;
  setSimilarityDisplay: (value: SimilarityDisplay) => void;
  setSimilarityMetric: (value: SimilarityMetric) => void;
  setConfigPath: (value: string) => void;
  setCheckpointPath: (value: string) => void;
  setPredictionRunning: (value: boolean) => void;
};

function defaultGranularity(scene: LoadedScene): string {
  const configured = scene.manifest.defaults?.granularity;
  if (configured && scene.pseudoLabels[configured]) return configured;
  if (configured && scene.predictions[configured]) return configured;
  const pseudo = Object.keys(scene.pseudoLabels).sort();
  if (pseudo.includes('g05')) return 'g05';
  if (pseudo.length) return pseudo[0];
  const pred = Object.keys(scene.predictions).sort();
  return pred[0] ?? 'g05';
}

export const useStore = create<UiState>((set, get) => ({
  scene: null,
  loadPath: '',
  loading: false,
  error: null,
  layerMode: 'rgb',
  granularity: 'g05',
  gtKey: null,
  featureSource: null,
  pointSize: 3,
  maxRenderPoints: 300000,
  selected: null,
  isolate: null,
  similarities: null,
  similarityLoading: false,
  similarityThreshold: 0.5,
  similarityDisplay: 'heatmap',
  similarityMetric: 'similarity',
  configPath: '/cluster/home/nedela/nedela/projects/chorus/student/configs/scannet_full_continuous_v2_eval_gt_classagnostic.yaml',
  checkpointPath: '/cluster/work/igp_psr/nedela/student_runs/scannet_full_continuous_v2_pseudo_pretrain/checkpoints/best.pt',
  predictionRunning: false,
  setLoadPath: value => set({ loadPath: value }),
  setLoading: value => set({ loading: value }),
  setError: value => set({ error: value }),
  setScene: scene => {
    if (!scene) {
      set({ scene: null, selected: null, isolate: null, similarities: null, error: null });
      return;
    }
    const granularity = defaultGranularity(scene);
    const gtKeys = Object.keys(scene.gt).sort();
    const featureKeys = Object.keys(scene.featureRefs).sort();
    const currentMode = get().layerMode;
    const hasPrediction = Object.keys(scene.predictions).length > 0;
    const hasScore = Object.values(scene.predictions).some(p => Boolean(p.scores));
    const modeAvailable =
      currentMode === 'gt' ? gtKeys.length > 0
        : currentMode === 'prediction' ? hasPrediction
          : currentMode === 'score' ? hasScore
            : currentMode === 'similarity' ? featureKeys.length > 0
              : true;
    set({
      scene,
      granularity,
      gtKey: gtKeys[0] ?? null,
      featureSource: featureKeys[0] ?? null,
      layerMode: modeAvailable ? currentMode : 'rgb',
      selected: null,
      isolate: null,
      similarities: null,
      error: null,
    });
  },
  setLoadedFeature: feature => {
    const scene = get().scene;
    if (!scene) return;
    set({
      scene: {
        ...scene,
        loadedFeatures: {
          ...scene.loadedFeatures,
          [feature.name]: feature,
        },
      },
    });
  },
  setLayerMode: mode => set({ layerMode: mode, isolate: null }),
  setGranularity: key => set({ granularity: key, isolate: null }),
  setGtKey: key => set({ gtKey: key, isolate: null }),
  setFeatureSource: key => set({ featureSource: key, similarities: null }),
  setPointSize: value => set({ pointSize: Math.max(1, Math.min(12, value)) }),
  setMaxRenderPoints: value => set({ maxRenderPoints: Math.max(1000, Math.min(2000000, Math.round(value))) }),
  setSelected: selected => set({ selected, similarities: null }),
  setIsolate: value => set({ isolate: value }),
  setSimilarities: values => set({ similarities: values }),
  setSimilarityLoading: value => set({ similarityLoading: value }),
  setSimilarityThreshold: value => set({ similarityThreshold: Math.max(-1, Math.min(2, value)) }),
  setSimilarityDisplay: value => set({ similarityDisplay: value }),
  setSimilarityMetric: value => set({ similarityMetric: value }),
  setConfigPath: value => set({ configPath: value }),
  setCheckpointPath: value => set({ checkpointPath: value }),
  setPredictionRunning: value => set({ predictionRunning: value }),
}));
