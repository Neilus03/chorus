import { expect, test } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';

function headerFor(descr: string, shape: number[]): Uint8Array {
  const shapeText = shape.length === 1 ? `${shape[0]},` : shape.join(', ');
  const raw = `{'descr': '${descr}', 'fortran_order': False, 'shape': (${shapeText}), }`;
  const rawBytes = new TextEncoder().encode(raw);
  const pad = 16 - ((10 + rawBytes.length + 1) % 16);
  const header = new Uint8Array(rawBytes.length + pad + 1);
  header.set(rawBytes);
  header.fill(32, rawBytes.length, header.length - 1);
  header[header.length - 1] = 10;
  return header;
}

function writeNpy(filePath: string, descr: string, shape: number[], payload: ArrayBufferView): void {
  const header = headerFor(descr, shape);
  const data = new Uint8Array(payload.buffer, payload.byteOffset, payload.byteLength);
  const out = new Uint8Array(10 + header.length + data.length);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0], 0);
  out[8] = header.length & 0xff;
  out[9] = header.length >> 8;
  out.set(header, 10);
  out.set(data, 10 + header.length);
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, out);
}

function normalizeRows(values: Float32Array, dim: number): Float32Array {
  const out = new Float32Array(values);
  for (let row = 0; row < values.length / dim; row += 1) {
    let norm = 0;
    for (let d = 0; d < dim; d += 1) norm += out[row * dim + d] ** 2;
    norm = Math.sqrt(norm) || 1;
    for (let d = 0; d < dim; d += 1) out[row * dim + d] /= norm;
  }
  return out;
}

function createSyntheticBundle(root: string): { manifestPath: string; sceneDir: string; packDir: string } {
  const sceneDir = path.join(root, 'synthetic_scene');
  const packDir = path.join(sceneDir, 'training_pack');
  const bundleDir = path.join(root, 'bundle');
  fs.mkdirSync(packDir, { recursive: true });
  fs.mkdirSync(bundleDir, { recursive: true });

  const n = 36;
  const points = new Float32Array(n * 3);
  const colors = new Uint8Array(n * 3);
  const labels = new Int32Array(n);
  const predLabels = new Int32Array(n);
  const predScores = new Float32Array(n);
  const predQueryIds = new Int32Array(n);
  const gtLabels = new Int32Array(n);
  const features = new Float32Array(n * 4);

  for (let i = 0; i < n; i += 1) {
    const row = Math.floor(i / 6);
    const col = i % 6;
    points[i * 3] = col * 0.18;
    points[i * 3 + 1] = row * 0.18;
    points[i * 3 + 2] = (i % 3) * 0.04;
    colors[i * 3] = 50 + col * 28;
    colors[i * 3 + 1] = 70 + row * 25;
    colors[i * 3 + 2] = 180;
    labels[i] = col < 3 ? 0 : 1;
    predLabels[i] = row < 3 ? 0 : 1;
    predScores[i] = row < 3 ? 0.72 : 0.91;
    predQueryIds[i] = predLabels[i] + 4;
    gtLabels[i] = labels[i];
    features[i * 4] = labels[i] === 0 ? 1 : 0;
    features[i * 4 + 1] = labels[i] === 1 ? 1 : 0;
    features[i * 4 + 2] = row / 6;
    features[i * 4 + 3] = col / 6;
  }

  writeNpy(path.join(packDir, 'points.npy'), '<f4', [n, 3], points);
  writeNpy(path.join(packDir, 'colors.npy'), '|u1', [n, 3], colors);
  writeNpy(path.join(packDir, 'labels_g0.5.npy'), '<i4', [n], labels);
  writeNpy(path.join(packDir, 'valid_points.npy'), '|u1', [n], new Uint8Array(n).fill(1));
  writeNpy(path.join(packDir, 'seen_points.npy'), '|u1', [n], new Uint8Array(n).fill(1));
  writeNpy(path.join(packDir, 'supervision_mask.npy'), '|u1', [n], new Uint8Array(n).fill(1));
  fs.writeFileSync(path.join(packDir, 'scene_meta.json'), JSON.stringify({
    pack_version: '1.0',
    dataset: 'synthetic',
    scene_id: 'synthetic_scene',
    num_points: n,
    granularities: [0.5],
    label_files: { 'g0.5': 'labels_g0.5.npy' },
    optional_files_present: { 'colors.npy': true },
  }));

  writeNpy(path.join(bundleDir, 'pred_labels_g05.npy'), '<i4', [n], predLabels);
  writeNpy(path.join(bundleDir, 'pred_scores_g05.npy'), '<f4', [n], predScores);
  writeNpy(path.join(bundleDir, 'pred_query_ids_g05.npy'), '<i4', [n], predQueryIds);
  writeNpy(path.join(bundleDir, 'gt_instances_scannet20.npy'), '<i4', [n], gtLabels);
  writeNpy(path.join(bundleDir, 'features_decoder_mask_feat.npy'), '<f4', [n, 4], normalizeRows(features, 4));
  fs.writeFileSync(path.join(bundleDir, 'pred_query_table_g05.json'), JSON.stringify([
    { query_id: 4, score_probability: 0.72, mask_area: 18, kept: true, instance_label: 0 },
    { query_id: 5, score_probability: 0.91, mask_area: 18, kept: true, instance_label: 1 },
  ]));
  fs.writeFileSync(path.join(bundleDir, 'gt_instance_classes_scannet20.json'), JSON.stringify({ 0: 1, 1: 2 }));

  const manifest = {
    schema_version: 'chorus_inspection_bundle/v1',
    scene_id: 'synthetic_scene',
    dataset: 'synthetic',
    training_pack_dir: packDir,
    created_at: new Date().toISOString(),
    arrays: {
      points: { path: path.join(packDir, 'points.npy') },
      colors: { path: path.join(packDir, 'colors.npy') },
    },
    labels: {
      pseudo: {
        g05: { path: path.join(packDir, 'labels_g0.5.npy') },
      },
      predictions: {
        g05: {
          pred_labels: { path: path.join(bundleDir, 'pred_labels_g05.npy') },
          pred_scores: { path: path.join(bundleDir, 'pred_scores_g05.npy') },
          pred_query_ids: { path: path.join(bundleDir, 'pred_query_ids_g05.npy') },
          query_table: { path: path.join(bundleDir, 'pred_query_table_g05.json') },
        },
      },
      gt: {
        scannet20: {
          labels: { path: path.join(bundleDir, 'gt_instances_scannet20.npy') },
          instance_classes: { path: path.join(bundleDir, 'gt_instance_classes_scannet20.json') },
        },
      },
    },
    features: {
      decoder_mask_feat: {
        path: path.join(bundleDir, 'features_decoder_mask_feat.npy'),
        dim: 4,
        normalized: true,
      },
    },
    defaults: {
      granularity: 'g05',
      score_threshold: 0.0,
      mask_threshold: 0.5,
      min_points: 30,
    },
    warnings: [],
  };
  const manifestPath = path.join(bundleDir, 'inspection_bundle.json');
  fs.writeFileSync(manifestPath, JSON.stringify(manifest));
  return { manifestPath, sceneDir, packDir };
}

test('loads a synthetic inspection bundle and renders controls', async ({ page }, testInfo) => {
  const { manifestPath } = createSyntheticBundle(testInfo.outputPath('fixture'));
  await page.goto(`/?bundle=${encodeURIComponent(manifestPath)}`);
  await expect(page.getByText('synthetic_scene').first()).toBeVisible();
  await expect(page.locator('canvas')).toBeVisible();

  const layerSection = page.locator('.section').filter({ has: page.getByRole('heading', { name: 'Layer' }) });
  await layerSection.getByRole('button', { name: 'Pseudo' }).click();
  await layerSection.getByRole('button', { name: 'Pred' }).click();
  await layerSection.getByRole('button', { name: 'GT' }).click();
  await layerSection.getByRole('button', { name: 'Feature' }).click();
  await page.waitForTimeout(500);

  await expect(page.locator('.status.error')).toHaveCount(0);
  const screenshot = await page.locator('canvas').screenshot();
  expect(screenshot.length).toBeGreaterThan(5000);
});

test('loads a direct training pack path', async ({ page }, testInfo) => {
  const { sceneDir, packDir } = createSyntheticBundle(testInfo.outputPath('pack-fixture'));
  await page.goto(`/?pack=${encodeURIComponent(sceneDir)}`);
  await expect(page.getByText('synthetic_scene').first()).toBeVisible();
  await expect(page.locator('canvas')).toBeVisible();
  await expect(page.locator('.status.error')).toHaveCount(0);

  await page.goto(`/?pack=${encodeURIComponent(packDir)}`);
  await expect(page.getByText('synthetic_scene').first()).toBeVisible();
  await expect(page.locator('canvas')).toBeVisible();
  await expect(page.locator('.status.error')).toHaveCount(0);
});

test('auto-discovers ScanNet GT for a real direct pack when available', async ({ page }) => {
  const sceneDir = '/cluster/work/igp_psr/nedela/chorus_poc/scans/scene0000_00';
  test.skip(!fs.existsSync(path.join(sceneDir, 'training_pack', 'scene_meta.json')), 'local ScanNet scene0000_00 pack is not mounted');
  test.skip(!fs.existsSync(path.join(sceneDir, 'scene0000_00.aggregation.json')), 'local ScanNet GT aggregation is not mounted');

  await page.goto(`/?pack=${encodeURIComponent(sceneDir)}`);
  await expect(page.getByText('scene0000_00').first()).toBeVisible({ timeout: 60_000 });
  const layerSection = page.locator('.section').filter({ has: page.getByRole('heading', { name: 'Layer' }) });
  await expect(layerSection.getByRole('button', { name: 'GT' })).toBeEnabled();
  await expect(layerSection.locator('select').nth(1)).not.toHaveValue('');
  await layerSection.getByRole('button', { name: 'GT' }).click();
  await expect(page.locator('.status.error')).toHaveCount(0);
  const screenshot = await page.locator('canvas').screenshot();
  expect(screenshot.length).toBeGreaterThan(5000);
});
