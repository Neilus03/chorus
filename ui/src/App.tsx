import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import './App.css';
import { composePointColors } from './lib/colors';
import { loadFeatureSource, loadSceneFromPath } from './lib/loadScene';
import { deterministicSubsample } from './lib/subsample';
import { useStore } from './state/store';
import InspectorPanel from './ui/InspectorPanel';
import LayerPanel from './ui/LayerPanel';
import TopToolbar from './ui/TopToolbar';
import Viewport from './ui/Viewport';

function initialPathFromUrl(): string {
  const params = new URLSearchParams(window.location.search);
  return params.get('bundle') ?? params.get('pack') ?? '';
}

function pathFromUserInput(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) return '';
  try {
    const url = new URL(trimmed);
    return url.searchParams.get('bundle') ?? url.searchParams.get('pack') ?? trimmed;
  } catch {
    return trimmed;
  }
}

type RuntimeInfo = {
  prediction_backend?: {
    available?: boolean;
    reason?: string | null;
  };
};

export default function App() {
  const scene = useStore(s => s.scene);
  const loadPath = useStore(s => s.loadPath);
  const setLoadPath = useStore(s => s.setLoadPath);
  const loading = useStore(s => s.loading);
  const setLoading = useStore(s => s.setLoading);
  const error = useStore(s => s.error);
  const setError = useStore(s => s.setError);
  const setScene = useStore(s => s.setScene);
  const setLoadedFeature = useStore(s => s.setLoadedFeature);
  const layerMode = useStore(s => s.layerMode);
  const granularity = useStore(s => s.granularity);
  const gtKey = useStore(s => s.gtKey);
  const maxRenderPoints = useStore(s => s.maxRenderPoints);
  const selected = useStore(s => s.selected);
  const isolate = useStore(s => s.isolate);
  const featureSource = useStore(s => s.featureSource);
  const similarities = useStore(s => s.similarities);
  const setSimilarities = useStore(s => s.setSimilarities);
  const setSimilarityLoading = useStore(s => s.setSimilarityLoading);
  const similarityThreshold = useStore(s => s.similarityThreshold);
  const similarityDisplay = useStore(s => s.similarityDisplay);
  const configPath = useStore(s => s.configPath);
  const checkpointPath = useStore(s => s.checkpointPath);
  const predictionRunning = useStore(s => s.predictionRunning);
  const setPredictionRunning = useStore(s => s.setPredictionRunning);
  const workerRef = useRef<Worker | null>(null);
  const workerFeatureSourceRef = useRef<string | null>(null);
  const [runtimeInfo, setRuntimeInfo] = useState<RuntimeInfo | null>(null);

  const handleLoad = useCallback(async () => {
    const path = pathFromUserInput(loadPath);
    if (!path) return;
    if (path !== loadPath) setLoadPath(path);
    setLoading(true);
    setError(null);
    try {
      const loaded = await loadSceneFromPath(path);
      setScene(loaded);
      const url = new URL(window.location.href);
      url.search = '';
      url.searchParams.set(path.endsWith('.json') ? 'bundle' : 'pack', path);
      window.history.replaceState(null, '', url);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [loadPath, setError, setLoading, setScene]);

  const handleRunPrediction = useCallback(async () => {
    if (!scene || predictionRunning) return;
    if (runtimeInfo?.prediction_backend?.available !== true) {
      setError(runtimeInfo?.prediction_backend?.reason ?? 'Live prediction export is unavailable for this server.');
      return;
    }
    setPredictionRunning(true);
    setError(null);
    try {
      const res = await fetch('/_chorus/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenePath: scene.trainingPackDir ?? scene.manifest.training_pack_dir,
          configPath,
          checkpointPath,
          granularities: [granularity],
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const payload = await res.json() as { manifestPath: string };
      const loaded = await loadSceneFromPath(payload.manifestPath);
      setScene(loaded);
      const url = new URL(window.location.href);
      url.search = '';
      url.searchParams.set('bundle', payload.manifestPath);
      window.history.replaceState(null, '', url);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setPredictionRunning(false);
    }
  }, [checkpointPath, configPath, granularity, predictionRunning, runtimeInfo, scene, setError, setPredictionRunning, setScene]);

  useEffect(() => {
    fetch('/_chorus/runtime')
      .then(res => (res.ok ? res.json() : null))
      .then((payload: RuntimeInfo | null) => {
        setRuntimeInfo(payload ?? {
          prediction_backend: {
            available: false,
            reason: 'Live prediction export is unavailable for this server.',
          },
        });
      })
      .catch(() => {
        setRuntimeInfo({
          prediction_backend: {
            available: false,
            reason: 'Live prediction export is unavailable for this server.',
          },
        });
      });
  }, []);

  useEffect(() => {
    const initial = initialPathFromUrl();
    if (!initial) return;
    setLoadPath(initial);
    setLoading(true);
    setError(null);
    void loadSceneFromPath(initial)
      .then(setScene)
      .catch(err => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false));
  }, [setError, setLoading, setLoadPath, setScene]);

  useEffect(() => {
    workerFeatureSourceRef.current = null;
  }, [scene]);

  const sourceIndices = useMemo(() => {
    if (!scene) return null;
    return deterministicSubsample(scene.numPoints, maxRenderPoints, scene.sceneId);
  }, [scene, maxRenderPoints]);

  useEffect(() => {
    if (!scene || layerMode !== 'similarity' || !featureSource || !selected) return;
    let cancelled = false;
    const current = scene.loadedFeatures[featureSource];
    const ensureFeature = async () => {
      if (current) return current;
      setSimilarityLoading(true);
      const loaded = await loadFeatureSource(scene, featureSource);
      if (!cancelled) setLoadedFeature(loaded);
      return loaded;
    };
    ensureFeature()
      .then(feature => {
        if (cancelled) return;
        if (!workerRef.current) {
          workerRef.current = new Worker(new URL('./workers/featureSimilarity.worker.ts', import.meta.url), { type: 'module' });
          workerFeatureSourceRef.current = null;
        }
        const worker = workerRef.current;
        setSimilarityLoading(true);
        worker.onmessage = (event: MessageEvent<{ type: string; similarities?: Float32Array; error?: string }>) => {
          if (cancelled) return;
          if (event.data.type === 'ready') return;
          setSimilarityLoading(false);
          if (event.data.type === 'error') {
            setError(event.data.error ?? 'Feature similarity failed');
            return;
          }
          setSimilarities(event.data.similarities ?? null);
        };
        if (workerFeatureSourceRef.current !== featureSource) {
          worker.postMessage({
            type: 'init',
            source: featureSource,
            features: feature.values,
            dim: feature.dim,
            normalized: feature.normalized,
          });
          workerFeatureSourceRef.current = featureSource;
        }
        worker.postMessage({
          type: 'compute',
          source: featureSource,
          queryIndex: selected.sourceIndex,
        });
      })
      .catch(err => {
        if (!cancelled) {
          setSimilarityLoading(false);
          setError(err instanceof Error ? err.message : String(err));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [
    featureSource,
    layerMode,
    scene,
    selected,
    setError,
    setLoadedFeature,
    setSimilarities,
    setSimilarityLoading,
  ]);

  useEffect(() => () => workerRef.current?.terminate(), []);

  const renderColors = useMemo(() => {
    if (!scene || !sourceIndices) return null;
    return composePointColors(scene, {
      mode: layerMode,
      granularity,
      gtKey: gtKey ?? undefined,
      sourceIndices,
      selectedSourceIndex: selected?.sourceIndex ?? null,
      isolate,
      similarities,
      similarityThreshold,
      similarityDisplay,
    });
  }, [scene, sourceIndices, layerMode, granularity, gtKey, selected, isolate, similarities, similarityThreshold, similarityDisplay]);

  return (
    <div className="app">
      <TopToolbar onLoad={handleLoad} />
      <main className="main">
        <LayerPanel
          onRunPrediction={handleRunPrediction}
          predictionBackendAvailable={runtimeInfo?.prediction_backend?.available ?? null}
          predictionBackendReason={runtimeInfo?.prediction_backend?.reason ?? null}
        />
        <div className="viewport-shell">
          <Viewport sourceIndices={sourceIndices} colors={renderColors} />
          {loading ? <div className="status">Loading...</div> : null}
          {predictionRunning ? <div className="status">Running prediction export...</div> : null}
          {error ? <div className="status error">{error}</div> : null}
          {scene && sourceIndices && sourceIndices.length < scene.numPoints ? (
            <div className="status">
              Rendering {sourceIndices.length.toLocaleString()} of {scene.numPoints.toLocaleString()} points.
            </div>
          ) : null}
        </div>
        <InspectorPanel />
      </main>
    </div>
  );
}
