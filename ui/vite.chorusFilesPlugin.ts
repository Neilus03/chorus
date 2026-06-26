import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import type { IncomingMessage, ServerResponse } from 'node:http';
import type { Plugin } from 'vite';

const uiRoot = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(uiRoot, '..');
const allowedFileExtensions = new Set(['.json', '.npy', '.npz']);
const cacheRoot = path.join(uiRoot, '.cache');

function uniqueResolved(values: Array<string | undefined>): string[] {
  return [...new Set(values.filter((v): v is string => Boolean(v)).map(v => path.resolve(v)))];
}

function allowedRoots(): string[] {
  const configured = process.env.CHORUS_UI_ROOTS;
  if (configured) {
    return uniqueResolved(configured.split(path.delimiter));
  }
  return uniqueResolved([
    repoRoot,
    process.env.SCANNET_SCANS_ROOT,
    process.env.CHORUS_OUTPUT_ROOT,
    process.env.STUDENT_OUTPUT_ROOT,
    '/scratch2/nedela',
    '/cluster/work/igp_psr/nedela',
  ]);
}

function isInsideRoot(filePath: string, root: string): boolean {
  const rel = path.relative(root, filePath);
  return rel === '' || (!!rel && !rel.startsWith('..') && !path.isAbsolute(rel));
}

function decodeRequestedPath(raw: string): string {
  if (raw.startsWith('file://')) return fileURLToPath(raw);
  try {
    return decodeURIComponent(raw);
  } catch {
    return raw;
  }
}

function pathCandidates(raw: string): string[] {
  const requested = decodeRequestedPath(raw.split('?', 1)[0] ?? raw);
  const candidates: string[] = [];
  if (path.isAbsolute(requested)) {
    candidates.push(path.resolve(requested));
    candidates.push(path.resolve(repoRoot, requested.replace(/^\/+/, '')));
  } else {
    candidates.push(path.resolve(repoRoot, requested));
  }
  return [...new Set(candidates)];
}

function resolveAllowedPath(raw: string): { filePath?: string; status?: number; message?: string } {
  const roots = allowedRoots();
  const candidates = pathCandidates(raw);
  const filePath = candidates.find(candidate => fs.existsSync(candidate)) ?? candidates[0];
  if (!filePath) return { status: 400, message: 'Missing path' };
  if (!roots.some(root => isInsideRoot(filePath, root))) {
    return {
      status: 403,
      message: `Path is outside CHORUS_UI_ROOTS (${roots.join(path.delimiter)})`,
    };
  }
  if (!fs.existsSync(filePath)) return { status: 404, message: `Path not found: ${filePath}` };
  return { filePath };
}

function normalizeGranularityKey(raw: string): string {
  const key = String(raw).trim();
  if (/^g\d\d$/.test(key)) return key;
  const stripped = key.startsWith('g') ? key.slice(1) : key;
  const value = Number(stripped);
  if (Number.isFinite(value)) return `g${Math.round(value * 10).toString().padStart(2, '0')}`;
  return key.replace('.', '');
}

function artifact(pathValue: string) {
  return { path: pathValue };
}

function writeInt32Npy(filePath: string, values: Int32Array): void {
  const headerRaw = `{'descr': '<i4', 'fortran_order': False, 'shape': (${values.length},), }`;
  const headerBytes = Buffer.from(headerRaw, 'ascii');
  const pad = 16 - ((10 + headerBytes.length + 1) % 16);
  const header = Buffer.concat([headerBytes, Buffer.alloc(pad, ' '), Buffer.from('\n')]);
  const magic = Buffer.from([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0, header.length & 0xff, header.length >> 8]);
  const payload = Buffer.from(values.buffer, values.byteOffset, values.byteLength);
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, Buffer.concat([magic, header, payload]));
}

function safeReadJson(filePath: string): any | null {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch {
    return null;
  }
}

function scannetGtCachePaths(sceneDir: string, sceneId: string): { labelsPath: string; classesPath: string } {
  const key = crypto.createHash('sha1').update(`${sceneDir}:${sceneId}`).digest('hex').slice(0, 16);
  const dir = path.join(cacheRoot, 'gt', `${sceneId}_${key}`);
  return {
    labelsPath: path.join(dir, 'gt_instances_all.npy'),
    classesPath: path.join(dir, 'gt_instance_classes_all.json'),
  };
}

function maybeBuildScannetGt(sceneDir: string, sceneId: string, numPoints: number): Record<string, unknown> {
  const segPaths = [
    path.join(sceneDir, `${sceneId}_vh_clean_2.0.010000.segs.json`),
    path.join(sceneDir, `${sceneId}_vh_clean.segs.json`),
  ];
  const aggPaths = [
    path.join(sceneDir, `${sceneId}.aggregation.json`),
    path.join(sceneDir, `${sceneId}_vh_clean.aggregation.json`),
  ];
  const segPath = segPaths.find(p => fs.existsSync(p));
  const aggPath = aggPaths.find(p => fs.existsSync(p));
  if (!segPath || !aggPath) return {};

  const { labelsPath, classesPath } = scannetGtCachePaths(sceneDir, sceneId);
  if (!fs.existsSync(labelsPath) || !fs.existsSync(classesPath)) {
    const segJson = safeReadJson(segPath);
    const aggJson = safeReadJson(aggPath);
    const segIndices = Array.isArray(segJson?.segIndices) ? segJson.segIndices : null;
    const groups = Array.isArray(aggJson?.segGroups) ? aggJson.segGroups : null;
    if (!segIndices || !groups || segIndices.length !== numPoints) return {};

    const segToInst = new Map<number, number>();
    const instanceClasses: Record<string, string> = {};
    for (const group of groups) {
      const objectId = Number(group.objectId ?? group.id);
      if (!Number.isFinite(objectId) || objectId < 0 || !Array.isArray(group.segments)) continue;
      const instId = Math.trunc(objectId) + 1;
      instanceClasses[String(instId)] = String(group.label ?? '');
      for (const seg of group.segments) segToInst.set(Number(seg), instId);
    }

    const labels = new Int32Array(numPoints);
    labels.fill(-1);
    for (let i = 0; i < segIndices.length; i += 1) {
      const inst = segToInst.get(Number(segIndices[i]));
      if (inst !== undefined) labels[i] = inst;
    }
    writeInt32Npy(labelsPath, labels);
    fs.writeFileSync(classesPath, JSON.stringify(instanceClasses, null, 2));
  }

  return {
    all: {
      labels: artifact(labelsPath),
      instance_classes: artifact(classesPath),
    },
  };
}

function resolvePackDir(inputPath: string): string | null {
  const stat = fs.statSync(inputPath);
  if (stat.isFile() && path.basename(inputPath) === 'scene_meta.json') return path.dirname(inputPath);
  if (stat.isDirectory() && fs.existsSync(path.join(inputPath, 'scene_meta.json'))) return inputPath;
  if (stat.isDirectory()) {
    for (const name of ['training_pack', 'litept_pack']) {
      const candidate = path.join(inputPath, name);
      if (fs.existsSync(path.join(candidate, 'scene_meta.json'))) return candidate;
    }
  }
  return null;
}

function packManifest(packDir: string): Record<string, unknown> {
  const metaPath = path.join(packDir, 'scene_meta.json');
  const meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
  const labelFiles = meta.label_files && typeof meta.label_files === 'object' ? meta.label_files : {};
  const pseudo: Record<string, unknown> = {};
  for (const [rawKey, fileName] of Object.entries(labelFiles)) {
    if (typeof fileName !== 'string') continue;
    pseudo[normalizeGranularityKey(rawKey)] = {
      path: path.join(packDir, fileName),
      granularity_key: normalizeGranularityKey(rawKey),
      source_key: rawKey,
    };
  }
  const colorsPath = path.join(packDir, 'colors.npy');
  const normalsPath = path.join(packDir, 'normals.npy');
  const granularities = Object.keys(pseudo).sort();
  const sceneId = String(meta.scene_id ?? path.basename(path.dirname(packDir)));
  const numPoints = Number(meta.num_points ?? 0);
  const sceneDir = path.dirname(packDir);
  const gt = String(meta.dataset ?? '').toLowerCase() === 'scannet' && Number.isFinite(numPoints)
    ? maybeBuildScannetGt(sceneDir, sceneId, numPoints)
    : {};
  return {
    schema_version: 'chorus_inspection_bundle/v1',
    scene_id: sceneId,
    dataset: meta.dataset ?? null,
    created_at: new Date().toISOString(),
    training_pack_dir: packDir,
    source: { kind: 'training_pack', scene_meta: meta },
    arrays: {
      points: artifact(path.join(packDir, 'points.npy')),
      ...(fs.existsSync(colorsPath) ? { colors: artifact(colorsPath) } : {}),
      ...(fs.existsSync(normalsPath) ? { normals: artifact(normalsPath) } : {}),
    },
    labels: {
      pseudo,
      predictions: {},
      gt,
    },
    features: {},
    defaults: {
      granularity: granularities.includes('g05') ? 'g05' : (granularities[0] ?? null),
      score_threshold: 0.0,
      mask_threshold: 0.5,
      min_points: 30,
    },
    warnings: [],
  };
}

function readRequestBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let body = '';
    req.setEncoding('utf8');
    req.on('data', chunk => {
      body += chunk;
      if (body.length > 1024 * 1024) {
        reject(new Error('Request body too large'));
        req.destroy();
      }
    });
    req.on('end', () => resolve(body));
    req.on('error', reject);
  });
}

function defaultConfigPath(): string {
  return process.env.CHORUS_UI_DEFAULT_CONFIG || path.join(repoRoot, 'student/configs/scannet_full_continuous_v2_eval_gt_classagnostic.yaml');
}

function defaultCheckpointPath(): string {
  return process.env.CHORUS_UI_DEFAULT_CHECKPOINT || '/cluster/work/igp_psr/nedela/student_runs/scannet_full_continuous_v2_pseudo_pretrain/checkpoints/best.pt';
}

function runtimePayload(): Record<string, unknown> {
  const python = process.env.CHORUS_UI_PYTHON || 'python3';
  const liveExportRequested = process.env.CHORUS_UI_ENABLE_LIVE_EXPORT === '1';
  const hasExplicitPython = Boolean(process.env.CHORUS_UI_PYTHON);
  return {
    prediction_backend: {
      available: liveExportRequested && hasExplicitPython,
      python,
      reason: liveExportRequested
        ? (hasExplicitPython ? null : 'CHORUS_UI_PYTHON is not set')
        : 'Live export is disabled for this Vite server; export with Slurm and open the bundle.',
    },
  };
}

function hashPayload(payload: unknown): string {
  return crypto.createHash('sha1').update(JSON.stringify(payload)).digest('hex').slice(0, 16);
}

function runExporter(payload: any): Promise<{ manifestPath: string; stdout: string; stderr: string }> {
  const scenePath = String(payload.scenePath ?? '');
  if (!scenePath) throw new Error('Missing scenePath');
  const sceneResolved = resolveAllowedPath(scenePath);
  if (!sceneResolved.filePath) throw new Error(sceneResolved.message ?? 'Invalid scenePath');
  const sceneDir = resolvePackDir(sceneResolved.filePath)
    ? path.dirname(resolvePackDir(sceneResolved.filePath)!)
    : sceneResolved.filePath;

  const config = String(payload.configPath || defaultConfigPath());
  const checkpoint = String(payload.checkpointPath || defaultCheckpointPath());
  for (const p of [config, checkpoint]) {
    const resolved = resolveAllowedPath(p);
    if (!resolved.filePath) throw new Error(resolved.message ?? `Invalid path: ${p}`);
  }
  const granularities = Array.isArray(payload.granularities) && payload.granularities.length
    ? payload.granularities.map(String)
    : ['g05'];
  const python = process.env.CHORUS_UI_PYTHON || 'python3';
  const key = hashPayload({ sceneDir, config, checkpoint, granularities });
  const outRoot = path.join(cacheRoot, 'predictions', key);
  const sceneId = path.basename(sceneDir);
  const manifestPath = path.join(outRoot, sceneId, 'inspection_bundle.json');
  if (fs.existsSync(manifestPath)) {
    return Promise.resolve({ manifestPath, stdout: 'cache hit', stderr: '' });
  }
  fs.mkdirSync(outRoot, { recursive: true });

  const args = [
    path.join(uiRoot, 'scripts/export_inspection_bundle.py'),
    '--config', config,
    '--checkpoint', checkpoint,
    '--scene-dir', sceneDir,
    '--out-dir', outRoot,
    '--granularities', ...granularities,
    '--feature-sources', 'decoder_mask_feat', 'decoder_query_embed',
  ];

  return new Promise((resolve, reject) => {
    const child = spawn(python, args, {
      cwd: repoRoot,
      env: process.env,
    });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', chunk => { stdout += String(chunk); });
    child.stderr.on('data', chunk => { stderr += String(chunk); });
    child.on('error', reject);
    child.on('close', code => {
      if (code !== 0) {
        reject(new Error(
          `Prediction export failed with code ${code}.\n`
          + `Python: ${python}\n`
          + `Set CHORUS_UI_PYTHON to the environment that has numpy/torch/student installed.\n\n`
          + stderr.slice(-4000),
        ));
        return;
      }
      if (!fs.existsSync(manifestPath)) {
        reject(new Error(`Exporter finished but manifest was not written: ${manifestPath}\n${stdout}\n${stderr}`));
        return;
      }
      resolve({ manifestPath, stdout, stderr });
    });
  });
}

function resolveBundleOrPack(rawPath: string): { payload?: unknown; status?: number; message?: string } {
  const resolved = resolveAllowedPath(rawPath);
  if (!resolved.filePath) return resolved;

  const filePath = resolved.filePath;
  if (fs.statSync(filePath).isFile() && path.extname(filePath).toLowerCase() === '.json') {
    const payload = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    if (payload && payload.schema_version === 'chorus_inspection_bundle/v1') {
      return { payload };
    }
  }
  const packDir = resolvePackDir(filePath);
  if (!packDir) {
    return {
      status: 400,
      message: `Path is neither an inspection bundle nor a training pack: ${filePath}`,
    };
  }
  return { payload: packManifest(packDir) };
}

function sendText(res: ServerResponse, status: number, message: string): void {
  res.statusCode = status;
  res.setHeader('Content-Type', 'text/plain; charset=utf-8');
  res.end(message);
}

function sendJson(res: ServerResponse, payload: unknown): void {
  res.statusCode = 200;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  res.end(JSON.stringify(payload));
}

function streamFile(req: IncomingMessage, res: ServerResponse, rawPath: string): void {
  const resolved = resolveAllowedPath(rawPath);
  if (!resolved.filePath) {
    sendText(res, resolved.status ?? 500, resolved.message ?? 'Could not resolve path');
    return;
  }
  const filePath = resolved.filePath;
  if (!fs.statSync(filePath).isFile()) {
    sendText(res, 400, `Path is not a file: ${filePath}`);
    return;
  }
  const ext = path.extname(filePath).toLowerCase();
  if (!allowedFileExtensions.has(ext)) {
    sendText(res, 400, `Unsupported file extension: ${ext}`);
    return;
  }
  const stat = fs.statSync(filePath);
  res.statusCode = 200;
  res.setHeader('Content-Type', ext === '.json' ? 'application/json; charset=utf-8' : 'application/octet-stream');
  res.setHeader('Content-Length', String(stat.size));
  res.setHeader('Content-Disposition', `inline; filename="${path.basename(filePath)}"`);
  res.setHeader('Cache-Control', 'no-store');
  if (req.method === 'HEAD') {
    res.end();
    return;
  }
  fs.createReadStream(filePath).pipe(res);
}

export function chorusFilesPlugin(): Plugin {
  return {
    name: 'chorus-ui-files',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url || !['GET', 'HEAD', 'POST'].includes(req.method ?? '')) {
          next();
          return;
        }
        const url = new URL(req.url, 'http://chorus-ui.local');
        if (url.pathname === '/_chorus/predict' && req.method === 'POST') {
          void readRequestBody(req)
            .then(body => runExporter(JSON.parse(body || '{}')))
            .then(result => sendJson(res, result))
            .catch(err => sendText(res, 500, err instanceof Error ? err.message : String(err)));
          return;
        }
        if (url.pathname === '/_chorus/runtime') {
          sendJson(res, runtimePayload());
          return;
        }
        if (url.pathname === '/_chorus/resolve') {
          const rawPath = url.searchParams.get('path');
          if (!rawPath) {
            sendText(res, 400, 'Missing path query parameter');
            return;
          }
          try {
            const resolved = resolveBundleOrPack(rawPath);
            if (!resolved.payload) {
              sendText(res, resolved.status ?? 500, resolved.message ?? 'Could not resolve bundle');
              return;
            }
            sendJson(res, resolved.payload);
          } catch (err) {
            sendText(res, 500, err instanceof Error ? err.message : String(err));
          }
          return;
        }
        if (url.pathname === '/_chorus/file') {
          const rawPath = url.searchParams.get('path');
          if (!rawPath) {
            sendText(res, 400, 'Missing path query parameter');
            return;
          }
          streamFile(req, res, rawPath);
          return;
        }
        next();
      });
    },
  };
}
