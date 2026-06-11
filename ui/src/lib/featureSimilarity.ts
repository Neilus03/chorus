export function computeCosineSimilarities(
  features: Float32Array,
  dim: number,
  queryIndex: number,
  normalized: boolean,
): Float32Array {
  const count = Math.floor(features.length / dim);
  if (queryIndex < 0 || queryIndex >= count) {
    throw new Error(`queryIndex ${queryIndex} is outside feature count ${count}`);
  }
  const out = new Float32Array(count);
  const queryOffset = queryIndex * dim;
  let queryNorm = 0;
  for (let d = 0; d < dim; d += 1) {
    const v = features[queryOffset + d];
    queryNorm += v * v;
  }
  queryNorm = Math.sqrt(queryNorm) || 1;

  for (let i = 0; i < count; i += 1) {
    const offset = i * dim;
    let dot = 0;
    let norm = 0;
    for (let d = 0; d < dim; d += 1) {
      const a = features[offset + d];
      const b = features[queryOffset + d];
      dot += a * b;
      norm += a * a;
    }
    out[i] = normalized ? dot : dot / ((Math.sqrt(norm) || 1) * queryNorm);
  }
  return out;
}

