import type { NpyArray, TypedNpyData } from '../types';

function ascii(bytes: Uint8Array, start: number, end: number): string {
  return new TextDecoder('latin1').decode(bytes.slice(start, end));
}

function hasNpyMagic(bytes: Uint8Array): boolean {
  return (
    bytes[0] === 0x93
    && bytes[1] === 0x4e
    && bytes[2] === 0x55
    && bytes[3] === 0x4d
    && bytes[4] === 0x50
    && bytes[5] === 0x59
  );
}

function parseHeaderShape(raw: string): number[] {
  const match = /'shape'\s*:\s*\(([^)]*)\)/.exec(raw) ?? /"shape"\s*:\s*\(([^)]*)\)/.exec(raw);
  if (!match) throw new Error(`Could not parse .npy shape from header: ${raw}`);
  return match[1]
    .split(',')
    .map(v => v.trim())
    .filter(Boolean)
    .map(v => Number(v));
}

function parseHeader(raw: string): { descr: string; fortranOrder: boolean; shape: number[] } {
  const descr = /'descr'\s*:\s*'([^']+)'/.exec(raw)?.[1] ?? /"descr"\s*:\s*"([^"]+)"/.exec(raw)?.[1];
  const fortranRaw = /'fortran_order'\s*:\s*(True|False)/.exec(raw)?.[1] ?? /"fortran_order"\s*:\s*(true|false)/.exec(raw)?.[1];
  if (!descr || !fortranRaw) throw new Error(`Invalid .npy header: ${raw}`);
  return {
    descr,
    fortranOrder: fortranRaw === 'True' || fortranRaw === 'true',
    shape: parseHeaderShape(raw),
  };
}

function elementCount(shape: number[]): number {
  return shape.reduce((acc, v) => acc * v, 1);
}

function isLittleEndian(descr: string): boolean {
  return descr[0] === '<' || descr[0] === '|' || (descr[0] !== '>' && descr[0] !== '!');
}

function dtypeCode(descr: string): string {
  return descr.replace(/^[<>=|!]/, '');
}

function halfToFloat(value: number): number {
  const sign = (value & 0x8000) ? -1 : 1;
  const exp = (value >> 10) & 0x1f;
  const frac = value & 0x03ff;
  if (exp === 0) {
    return sign * Math.pow(2, -14) * (frac / 1024);
  }
  if (exp === 0x1f) {
    return frac ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  }
  return sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

function parseData(descr: string, payload: ArrayBuffer, count: number): TypedNpyData {
  const little = isLittleEndian(descr);
  const code = dtypeCode(descr);
  const view = new DataView(payload);

  if (code === 'u1') return new Uint8Array(payload, 0, count).slice();
  if (code === 'i1') return new Int8Array(payload, 0, count).slice();
  if (code === 'b1') return new Uint8Array(payload, 0, count).slice();

  if (code === 'i2') {
    const out = new Int16Array(count);
    for (let i = 0; i < count; i += 1) out[i] = view.getInt16(i * 2, little);
    return out;
  }
  if (code === 'i4') {
    const out = new Int32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = view.getInt32(i * 4, little);
    return out;
  }
  if (code === 'i8') {
    const out = new Int32Array(count);
    for (let i = 0; i < count; i += 1) {
      const value = view.getBigInt64(i * 8, little);
      if (value < BigInt(-2147483648) || value > BigInt(2147483647)) {
        throw new Error('.npy int64 value is outside safe Int32 label range');
      }
      out[i] = Number(value);
    }
    return out;
  }
  if (code === 'f2') {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = halfToFloat(view.getUint16(i * 2, little));
    return out;
  }
  if (code === 'f4') {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = view.getFloat32(i * 4, little);
    return out;
  }
  if (code === 'f8') {
    const out = new Float64Array(count);
    for (let i = 0; i < count; i += 1) out[i] = view.getFloat64(i * 8, little);
    return out;
  }
  throw new Error(`Unsupported .npy dtype: ${descr}`);
}

export function parseNpy(buffer: ArrayBuffer): NpyArray {
  const bytes = new Uint8Array(buffer);
  if (bytes.length < 10 || !hasNpyMagic(bytes)) {
    throw new Error('Invalid .npy magic header');
  }
  const major = bytes[6];
  const minor = bytes[7];
  let headerLength = 0;
  let offset = 0;
  if (major === 1) {
    headerLength = bytes[8] | (bytes[9] << 8);
    offset = 10;
  } else if (major === 2 || major === 3) {
    headerLength = bytes[8] | (bytes[9] << 8) | (bytes[10] << 16) | (bytes[11] << 24);
    offset = 12;
  } else {
    throw new Error(`Unsupported .npy version ${major}.${minor}`);
  }
  const header = parseHeader(ascii(bytes, offset, offset + headerLength));
  if (header.fortranOrder) throw new Error('Fortran-order .npy arrays are not supported');
  const payloadOffset = offset + headerLength;
  const count = elementCount(header.shape);
  const payload = buffer.slice(payloadOffset);
  return {
    ...header,
    data: parseData(header.descr, payload, count),
  };
}

export function toFloat32(array: NpyArray): Float32Array {
  if (array.data instanceof Float32Array) return array.data;
  const out = new Float32Array(array.data.length);
  for (let i = 0; i < array.data.length; i += 1) out[i] = Number(array.data[i]);
  return out;
}

export function toInt32(array: NpyArray): Int32Array {
  if (array.data instanceof Int32Array) return array.data;
  const out = new Int32Array(array.data.length);
  for (let i = 0; i < array.data.length; i += 1) out[i] = Number(array.data[i]);
  return out;
}

export function toUint8Colors(array: NpyArray, expectedRows: number): Uint8Array {
  if (array.shape.length !== 2 || array.shape[0] !== expectedRows || array.shape[1] < 3) {
    throw new Error(`colors.npy must have shape [${expectedRows}, >=3], got [${array.shape.join(', ')}]`);
  }
  if (array.data instanceof Uint8Array && array.shape[1] === 3) return array.data;
  const values = toFloat32(array);
  let maxValue = 0;
  for (let i = 0; i < values.length; i += 1) maxValue = Math.max(maxValue, values[i]);
  const scale = maxValue <= 1.0 ? 255 : 1;
  const out = new Uint8Array(expectedRows * 3);
  const stride = array.shape[1];
  for (let row = 0; row < expectedRows; row += 1) {
    for (let c = 0; c < 3; c += 1) {
      out[row * 3 + c] = Math.max(0, Math.min(255, Math.round(values[row * stride + c] * scale)));
    }
  }
  return out;
}
