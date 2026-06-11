import { expect, it } from 'vitest';
import { colorForLabel } from './colors';

it('uses neutral color for unlabeled points', () => {
  expect(colorForLabel(-1)).toEqual([114, 122, 132]);
});

it('produces stable instance colors', () => {
  expect(colorForLabel(12, 3)).toEqual(colorForLabel(12, 3));
  expect(colorForLabel(12, 3)).not.toEqual(colorForLabel(13, 3));
});

