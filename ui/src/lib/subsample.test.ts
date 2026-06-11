import { expect, it } from 'vitest';
import { deterministicSubsample } from './subsample';

it('keeps all indices under the cap', () => {
  expect(Array.from(deterministicSubsample(4, 10, 'a'))).toEqual([0, 1, 2, 3]);
});

it('is deterministic and sorted over the cap', () => {
  const a = Array.from(deterministicSubsample(100, 10, 'scene'));
  const b = Array.from(deterministicSubsample(100, 10, 'scene'));
  expect(a).toEqual(b);
  expect(a).toEqual([...a].sort((x, y) => x - y));
  expect(new Set(a).size).toBe(10);
});

