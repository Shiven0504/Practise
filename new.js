function mean(data) {
    const [n, m] = [data.length, data[0].length];
    const mu = new Array(m).fill(0);
    for (const row of data) for (let j = 0; j < m; j++) mu[j] += row[j];
    return mu.map(v => v / n);
  }
  
  function center(data, mu) {
    return data.map(row => row.map((v, j) => v - mu[j]));
  }
  
  function covMatrix(X) {
    const [n, m] = [X.length, X[0].length];
    const C = Array.from({ length: m }, () => new Array(m).fill(0));
    for (const row of X)
      for (let i = 0; i < m; i++)
        for (let j = i; j < m; j++) C[i][j] += row[i] * row[j];
    const scale = 1 / (n - 1);
    for (let i = 0; i < m; i++)
      for (let j = i; j < m; j++) C[j][i] = C[i][j] *= scale;
    return C;
  }
  
  function dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }
  
  function norm(v) { return Math.sqrt(dot(v, v)); }
  
  function matVec(M, v) { return M.map(row => dot(row, v)); }
  
  function powerIteration(M, maxIter = 1000, tol = 1e-9) {
    const n = M.length;
    let b = Array.from({ length: n }, () => Math.random() - 0.5);
    let bn = norm(b);
    if (bn === 0) { b[0] = 1; bn = 1; }
    b = b.map(x => x / bn);
  
    let lambda = 0;
    for (let i = 0; i < maxIter; i++) {
      const Mb = matVec(M, b);
      const MbN = norm(Mb);
      if (MbN === 0) break;
      const next = Mb.map(x => x / MbN);
      const lNext = dot(next, matVec(M, next));
      if (Math.abs(lNext - lambda) < tol) return { vec: next, val: lNext };
      b = next;
      lambda = lNext;
    }
    return { vec: b, val: lambda };
  }
  
  function deflate(M, val, vec) {
    for (let i = 0; i < M.length; i++)
      for (let j = 0; j < M.length; j++)
        M[i][j] -= val * vec[i] * vec[j];
  }
  
  /**
   * PCA via power iteration with deflation.
   * @param {number[][]} data - n samples x m features
   * @param {number} [k] - number of components (default: min(n, m))
   * @returns {{ components, eigenvalues, explainedVariance, mean, projected }}
   */
  function pca(data, k = null) {
    if (!data?.length) throw new Error("data must be non-empty array");
    const [n, m] = [data.length, data[0].length];
    k = Math.min(k ?? Math.min(n, m), m);
  
    const mu = mean(data);
    const X = center(data, mu);
    const M = covMatrix(X);
  
    const components = [], eigenvalues = [];
  
    for (let i = 0; i < k; i++) {
      const { vec, val } = powerIteration(M);
      if (val <= 0) break;
      const vn = norm(vec);
      const v = vec.map(x => x / vn);
      components.push(v);
      eigenvalues.push(val);
      deflate(M, val, v);
    }
  
    const total = eigenvalues.reduce((s, x) => s + x, 0) || 1;
    const projected = X.map(row => components.map(c => dot(row, c)));
  
    return {
      components,
      eigenvalues,
      explainedVariance: eigenvalues.map(v => v / total),
      mean: mu,
      projected,
    };
  }
  
  module.exports = { pca };