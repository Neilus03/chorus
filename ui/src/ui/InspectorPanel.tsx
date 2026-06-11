import { useStore } from '../state/store';

function fmt(value: number | undefined, digits = 4): string {
  return value === undefined || Number.isNaN(value) ? '-' : value.toFixed(digits);
}

function instanceSize(labels: Int32Array | undefined, label: number | undefined): number | null {
  if (!labels || label === undefined || label < 0) return null;
  let count = 0;
  for (let i = 0; i < labels.length; i += 1) if (labels[i] === label) count += 1;
  return count;
}

export default function InspectorPanel() {
  const scene = useStore(s => s.scene);
  const selected = useStore(s => s.selected);
  const granularity = useStore(s => s.granularity);
  const gtKey = useStore(s => s.gtKey);
  const featureSource = useStore(s => s.featureSource);

  if (!scene) {
    return (
      <aside className="panel right">
        <section className="section"><h2>Inspector</h2><p className="small">No scene loaded.</p></section>
      </aside>
    );
  }
  if (!selected) {
    return (
      <aside className="panel right">
        <section className="section"><h2>Inspector</h2><p className="small">Click a point.</p></section>
      </aside>
    );
  }

  const i = selected.sourceIndex;
  const pseudo = scene.pseudoLabels[granularity]?.[i];
  const pred = scene.predictions[granularity];
  const gt = gtKey ? scene.gt[gtKey] : undefined;
  const rgb = [scene.colors[i * 3], scene.colors[i * 3 + 1], scene.colors[i * 3 + 2]];
  const feature = featureSource ? scene.loadedFeatures[featureSource] : undefined;
  let featureNorm: number | undefined;
  if (feature) {
    let sum = 0;
    const offset = i * feature.dim;
    for (let d = 0; d < feature.dim; d += 1) sum += feature.values[offset + d] ** 2;
    featureNorm = Math.sqrt(sum);
  }

  return (
    <aside className="panel right">
      <section className="section">
        <h2>Point</h2>
        <dl className="kv">
          <dt>Index</dt><dd>{i.toLocaleString()}</dd>
          <dt>X</dt><dd>{fmt(scene.points[i * 3])}</dd>
          <dt>Y</dt><dd>{fmt(scene.points[i * 3 + 1])}</dd>
          <dt>Z</dt><dd>{fmt(scene.points[i * 3 + 2])}</dd>
          <dt>RGB</dt><dd>{rgb.join(', ')}</dd>
        </dl>
      </section>

      <section className="section">
        <h2>Labels</h2>
        <dl className="kv">
          <dt>Pseudo</dt><dd>{pseudo ?? '-'}</dd>
          <dt>Pseudo size</dt><dd>{instanceSize(scene.pseudoLabels[granularity], pseudo)?.toLocaleString() ?? '-'}</dd>
          <dt>Pred</dt><dd>{pred?.labels[i] ?? '-'}</dd>
          <dt>Pred score</dt><dd>{pred?.scores ? fmt(pred.scores[i], 3) : '-'}</dd>
          <dt>Query</dt><dd>{pred?.queryIds ? pred.queryIds[i] : '-'}</dd>
          <dt>Pred size</dt><dd>{instanceSize(pred?.labels, pred?.labels[i])?.toLocaleString() ?? '-'}</dd>
          <dt>GT</dt><dd>{gt?.labels[i] ?? '-'}</dd>
          <dt>GT size</dt><dd>{instanceSize(gt?.labels, gt?.labels[i])?.toLocaleString() ?? '-'}</dd>
          <dt>GT class</dt><dd>{gt?.labels[i] !== undefined ? gt?.instanceClasses?.[String(gt.labels[i])] ?? '-' : '-'}</dd>
        </dl>
      </section>

      <section className="section">
        <h2>Feature</h2>
        <dl className="kv">
          <dt>Source</dt><dd>{featureSource ?? '-'}</dd>
          <dt>Loaded</dt><dd>{feature ? 'yes' : 'no'}</dd>
          <dt>Dim</dt><dd>{feature?.dim ?? '-'}</dd>
          <dt>Norm</dt><dd>{fmt(featureNorm, 3)}</dd>
        </dl>
      </section>
    </aside>
  );
}

