import { Camera, Download, FolderOpen, RotateCcw } from 'lucide-react';
import { useStore } from '../state/store';

type Props = {
  onLoad: () => void;
};

function captureCanvas(): void {
  const canvas = document.querySelector('canvas');
  if (!(canvas instanceof HTMLCanvasElement)) return;
  canvas.toBlob(blob => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'chorus-pointcloud.png';
    a.click();
    URL.revokeObjectURL(url);
  }, 'image/png');
}

export default function TopToolbar({ onLoad }: Props) {
  const loadPath = useStore(s => s.loadPath);
  const setLoadPath = useStore(s => s.setLoadPath);
  const loading = useStore(s => s.loading);
  const scene = useStore(s => s.scene);
  const pointSize = useStore(s => s.pointSize);
  const setPointSize = useStore(s => s.setPointSize);
  const maxRenderPoints = useStore(s => s.maxRenderPoints);
  const setMaxRenderPoints = useStore(s => s.setMaxRenderPoints);

  return (
    <header className="toolbar">
      <input
        className="path-input"
        value={loadPath}
        onChange={e => setLoadPath(e.target.value)}
        onKeyDown={e => {
          if (e.key === 'Enter') onLoad();
        }}
        placeholder="/path/to/scene, training_pack, or inspection_bundle.json"
      />
      <button className="button" onClick={onLoad} disabled={loading || !loadPath.trim()}>
        <FolderOpen size={16} /> {loading ? 'Loading' : 'Open'}
      </button>
      <button className="icon-button" title="Fit camera" onClick={() => window.dispatchEvent(new Event('chorus-fit-camera'))} disabled={!scene}>
        <RotateCcw size={16} />
      </button>
      <button className="icon-button" title="Screenshot" onClick={captureCanvas} disabled={!scene}>
        <Camera size={16} />
      </button>
      <div className="control-row" style={{ gridTemplateColumns: '72px 100px', margin: 0 }}>
        <label>Point</label>
        <input type="range" min={1} max={10} step={0.5} value={pointSize} onChange={e => setPointSize(Number(e.target.value))} />
      </div>
      <div className="control-row" style={{ gridTemplateColumns: '72px 110px', margin: 0 }}>
        <label>Cap</label>
        <input type="number" value={maxRenderPoints} onChange={e => setMaxRenderPoints(Number(e.target.value))} />
      </div>
      <button className="icon-button" title="Download manifest" onClick={() => {
        if (!scene) return;
        const blob = new Blob([JSON.stringify(scene.manifest, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${scene.sceneId}_inspection_manifest.json`;
        a.click();
        URL.revokeObjectURL(url);
      }} disabled={!scene}>
        <Download size={16} />
      </button>
    </header>
  );
}

