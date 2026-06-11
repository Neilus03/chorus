function hashSeed(text: string): number {
  let h = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function nextRandom(state: { value: number }): number {
  state.value = (Math.imul(state.value, 1664525) + 1013904223) >>> 0;
  return state.value / 0x100000000;
}

export function deterministicSubsample(count: number, cap: number, seedText: string): Uint32Array {
  if (count <= 0) return new Uint32Array(0);
  if (cap <= 0) return new Uint32Array(0);
  if (count <= cap) {
    const out = new Uint32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = i;
    return out;
  }
  const state = { value: hashSeed(seedText) || 1 };
  const selected = new Set<number>();
  while (selected.size < cap) {
    selected.add(Math.floor(nextRandom(state) * count));
  }
  return Uint32Array.from([...selected].sort((a, b) => a - b));
}

export function gatherFloat3(values: Float32Array, indices: Uint32Array): Float32Array {
  const out = new Float32Array(indices.length * 3);
  for (let i = 0; i < indices.length; i += 1) {
    const src = indices[i] * 3;
    const dst = i * 3;
    out[dst] = values[src];
    out[dst + 1] = values[src + 1];
    out[dst + 2] = values[src + 2];
  }
  return out;
}

export function gatherUint8_3(values: Uint8Array, indices: Uint32Array): Uint8Array {
  const out = new Uint8Array(indices.length * 3);
  for (let i = 0; i < indices.length; i += 1) {
    const src = indices[i] * 3;
    const dst = i * 3;
    out[dst] = values[src];
    out[dst + 1] = values[src + 1];
    out[dst + 2] = values[src + 2];
  }
  return out;
}

