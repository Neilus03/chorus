import { expect, it } from 'vitest';
import { computeCosineSimilarities } from './featureSimilarity';

it('computes cosine similarity for known vectors', () => {
  const features = new Float32Array([
    1, 0,
    0, 1,
    1, 1,
  ]);
  const sims = computeCosineSimilarities(features, 2, 0, false);
  expect(Array.from(sims).map(v => Number(v.toFixed(3)))).toEqual([1, 0, 0.707]);
});

