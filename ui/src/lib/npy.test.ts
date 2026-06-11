import { describe, expect, it } from 'vitest';
import { parseNpy, toFloat32, toInt32 } from './npy';

function makeNpy(descr: string, shape: number[], payload: Uint8Array): ArrayBuffer {
  const headerRaw = `{'descr': '${descr}', 'fortran_order': False, 'shape': (${shape.join(', ')}${shape.length === 1 ? ',' : ''}), }`;
  const headerBytes = new TextEncoder().encode(headerRaw);
  const pad = 16 - ((10 + headerBytes.length + 1) % 16);
  const header = new Uint8Array(headerBytes.length + pad + 1);
  header.set(headerBytes);
  header.fill(32, headerBytes.length, header.length - 1);
  header[header.length - 1] = 10;
  const out = new Uint8Array(10 + header.length + payload.length);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0], 0);
  out[8] = header.length & 0xff;
  out[9] = header.length >> 8;
  out.set(header, 10);
  out.set(payload, 10 + header.length);
  return out.buffer;
}

it('parses little-endian float32 arrays', () => {
  const payload = new Uint8Array(new Float32Array([1.5, 2.5, 3.5, 4.5]).buffer);
  const arr = parseNpy(makeNpy('<f4', [2, 2], payload));
  expect(arr.shape).toEqual([2, 2]);
  expect(Array.from(toFloat32(arr))).toEqual([1.5, 2.5, 3.5, 4.5]);
});

it('parses int64 labels into int32 when safe', () => {
  const payload = new Uint8Array(24);
  const view = new DataView(payload.buffer);
  view.setBigInt64(0, BigInt(-1), true);
  view.setBigInt64(8, BigInt(5), true);
  view.setBigInt64(16, BigInt(8), true);
  const arr = parseNpy(makeNpy('<i8', [3], payload));
  expect(Array.from(toInt32(arr))).toEqual([-1, 5, 8]);
});

describe('float16 support', () => {
  it('converts half values to float32', () => {
    const payload = new Uint8Array([0x00, 0x3c, 0x00, 0x40]);
    const arr = parseNpy(makeNpy('<f2', [2], payload));
    expect(Array.from(toFloat32(arr))).toEqual([1, 2]);
  });
});
