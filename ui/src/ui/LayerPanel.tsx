import { selectedInstanceFor } from '../lib/colors';
import { useStore } from '../state/store';
import type { LayerMode, SimilarityDisplay, SimilarityMetric } from '../types';

const modes: Array<[LayerMode, string]> = [
  ['rgb', 'RGB'],
  ['height', 'Height'],
  ['pseudo', 'Pseudo'],
  ['prediction', 'Pred'],
  ['gt', 'GT'],
  ['score', 'Score'],
  ['similarity', 'Feature'],
];

type Props = {
  onRunPrediction: () => void;
  predictionBackendAvailable: boolean | null;
  predictionBackendReason?: string | null;
};

export default function LayerPanel({ onRunPrediction, predictionBackendAvailable, predictionBackendReason }: Props) {
  const scene = useStore(s => s.scene);
  const mode = useStore(s => s.layerMode);
  const setMode = useStore(s => s.setLayerMode);
  const granularity = useStore(s => s.granularity);
  const setGranularity = useStore(s => s.setGranularity);
  const gtKey = useStore(s => s.gtKey);
  const setGtKey = useStore(s => s.setGtKey);
  const featureSource = useStore(s => s.featureSource);
  const setFeatureSource = useStore(s => s.setFeatureSource);
  const selected = useStore(s => s.selected);
  const isolate = useStore(s => s.isolate);
  const setIsolate = useStore(s => s.setIsolate);
  const similarityThreshold = useStore(s => s.similarityThreshold);
  const setSimilarityThreshold = useStore(s => s.setSimilarityThreshold);
  const similarityDisplay = useStore(s => s.similarityDisplay);
  const setSimilarityDisplay = useStore(s => s.setSimilarityDisplay);
  const similarityMetric = useStore(s => s.similarityMetric);
  const setSimilarityMetric = useStore(s => s.setSimilarityMetric);
  const similarityLoading = useStore(s => s.similarityLoading);
  const configPath = useStore(s => s.configPath);
  const setConfigPath = useStore(s => s.setConfigPath);
  const checkpointPath = useStore(s => s.checkpointPath);
  const setCheckpointPath = useStore(s => s.setCheckpointPath);
  const predictionRunning = useStore(s => s.predictionRunning);

  const granularityKeys = scene
    ? [...new Set([...Object.keys(scene.pseudoLabels), ...Object.keys(scene.predictions)])].sort()
    : [];
  const gtKeys = scene ? Object.keys(scene.gt).sort() : [];
  const featureKeys = scene ? Object.keys(scene.featureRefs).sort() : [];
  const hasPredictions = scene ? Object.keys(scene.predictions).length > 0 : false;
  const hasScores = scene ? Object.values(scene.predictions).some(p => Boolean(p.scores)) : false;
  const currentFeatureLoaded = Boolean(scene && featureSource && scene.loadedFeatures[featureSource]);
  const thresholdForDisplay = similarityMetric === 'distance' ? 1 - similarityThreshold : similarityThreshold;
  const liveExportAvailable = predictionBackendAvailable === true;

  const isolateCurrent = (kind: 'pseudo' | 'prediction' | 'gt', key: string) => {
    if (!scene || !selected) return;
    const next = selectedInstanceFor(scene, selected.sourceIndex, kind, key);
    setIsolate(next);
  };

  const modeDisabled = (key: LayerMode) => {
    if (!scene) return key !== 'rgb' && key !== 'height';
    if (key === 'pseudo') return Object.keys(scene.pseudoLabels).length === 0;
    if (key === 'prediction') return !hasPredictions;
    if (key === 'score') return !hasScores;
    if (key === 'gt') return gtKeys.length === 0;
    if (key === 'similarity') return featureKeys.length === 0;
    return false;
  };

  return (
    <aside className="panel">
      <section className="section">
        <h2>Scene</h2>
        <dl className="kv">
          <dt>ID</dt>
          <dd>{scene?.sceneId ?? 'No scene'}</dd>
          <dt>Points</dt>
          <dd>{scene ? scene.numPoints.toLocaleString() : '-'}</dd>
          <dt>Dataset</dt>
          <dd>{scene?.manifest.dataset ?? '-'}</dd>
        </dl>
        {scene?.warnings.length ? (
          <ul className="warning-list">
            {scene.warnings.slice(0, 4).map(w => <li key={w}>{w}</li>)}
          </ul>
        ) : null}
      </section>

      <section className="section">
        <h2>Layer</h2>
        <div className="segmented three">
          {modes.map(([key, label]) => (
            <button
              key={key}
              className={mode === key ? 'active' : ''}
              disabled={modeDisabled(key)}
              title={modeDisabled(key) ? `${label} is not available for this scene yet` : label}
              onClick={() => setMode(key)}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="control-row">
          <label>Granularity</label>
          <select value={granularity} onChange={e => setGranularity(e.target.value)} disabled={!granularityKeys.length}>
            {granularityKeys.map(key => <option key={key} value={key}>{key}</option>)}
          </select>
        </div>
        <div className="control-row">
          <label>GT</label>
          <select value={gtKey ?? ''} onChange={e => setGtKey(e.target.value || null)} disabled={!gtKeys.length}>
            <option value="">none</option>
            {gtKeys.map(key => <option key={key} value={key}>{key}</option>)}
          </select>
        </div>
      </section>

      <section className="section">
        <h2>Prediction Cache</h2>
        <div className="control-row" style={{ gridTemplateColumns: '72px minmax(0, 1fr)' }}>
          <label>Config</label>
          <input className="path-mini" value={configPath} onChange={e => setConfigPath(e.target.value)} />
        </div>
        <div className="control-row" style={{ gridTemplateColumns: '72px minmax(0, 1fr)' }}>
          <label>Checkpoint</label>
          <input className="path-mini" value={checkpointPath} onChange={e => setCheckpointPath(e.target.value)} />
        </div>
        <button
          className="button"
          style={{ width: '100%' }}
          disabled={!scene || predictionRunning || !liveExportAvailable}
          title={!liveExportAvailable ? (predictionBackendReason ?? 'Live prediction export is unavailable') : undefined}
          onClick={onRunPrediction}
        >
          {predictionRunning ? 'Running model...' : `Run/load ${granularity}`}
        </button>
        <p className="small">
          {hasPredictions
            ? 'Cached predictions are loaded.'
            : liveExportAvailable
              ? 'Predictions and decoder features are generated into a local cache.'
              : 'Open an exported inspection bundle to enable Pred, Score, and Feature.'}
        </p>
        {!hasPredictions && !liveExportAvailable ? (
          <p className="small">
            {predictionBackendReason ?? 'Live prediction export is unavailable for this server.'}
          </p>
        ) : null}
        {!hasPredictions ? (
          <p className="small">
            Bundle path: /cluster/work/igp_psr/nedela/chorus_ui_bundles/scene0000_00/inspection_bundle.json
          </p>
        ) : null}
      </section>

      <section className="section">
        <h2>Selection</h2>
        <div className="segmented three">
          <button disabled={!selected || !scene?.pseudoLabels[granularity]} onClick={() => isolateCurrent('pseudo', granularity)}>
            Pseudo
          </button>
          <button disabled={!selected || !scene?.predictions[granularity]} onClick={() => isolateCurrent('prediction', granularity)}>
            Pred
          </button>
          <button disabled={!selected || !gtKey} onClick={() => gtKey && isolateCurrent('gt', gtKey)}>
            GT
          </button>
        </div>
        <button className="button" style={{ width: '100%', marginTop: 8 }} disabled={!isolate} onClick={() => setIsolate(null)}>
          Clear isolate
        </button>
      </section>

      <section className="section">
        <h2>Feature Similarity</h2>
        <div className="control-row">
          <label>Source</label>
          <select value={featureSource ?? ''} onChange={e => setFeatureSource(e.target.value || null)} disabled={!featureKeys.length}>
            <option value="">none</option>
            {featureKeys.map(key => <option key={key} value={key}>{key}</option>)}
          </select>
        </div>
        <div className="control-row">
          <label>Metric</label>
          <select value={similarityMetric} onChange={e => setSimilarityMetric(e.target.value as SimilarityMetric)}>
            <option value="similarity">similarity</option>
            <option value="distance">distance</option>
          </select>
        </div>
        <div className="control-row">
          <label>{similarityMetric === 'distance' ? 'Distance' : 'Similarity'}</label>
          <input
            type="range"
            min={similarityMetric === 'distance' ? 0 : -1}
            max={similarityMetric === 'distance' ? 2 : 1}
            step={0.01}
            value={thresholdForDisplay}
            onChange={e => {
              const value = Number(e.target.value);
              setSimilarityThreshold(similarityMetric === 'distance' ? 1 - value : value);
            }}
          />
        </div>
        <div className="segmented">
          {(['heatmap', 'binary'] as SimilarityDisplay[]).map(value => (
            <button key={value} className={similarityDisplay === value ? 'active' : ''} onClick={() => setSimilarityDisplay(value)}>
              {value}
            </button>
          ))}
        </div>
        <p className="small">
          {similarityLoading ? 'Computing...' : currentFeatureLoaded ? 'Ready' : featureSource ? 'Loads on use' : 'No feature source'}
        </p>
      </section>
    </aside>
  );
}
