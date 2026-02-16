// ...existing code...
/**
 * Improved PCA implementation (covariance + power iteration)
 * - input validation
 * - optional standardization (z-score)
 * - deterministic RNG via seed option
 * - symmetric fixes & safety clamps
 * - explained variance computed w.r.t. total covariance trace
 * - example runs only when executed directly
 *
 * Exports: pca(data, k, opts)
 * opts:
 *   - maxIter (default 1000)
 *   - tol (default 1e-9)
 *   - standardize (default false)
 *   - seed (optional, integer for deterministic runs)
 */

function validateData(data) {
    if (!Array.isArray(data) || data.length === 0) throw new Error("data must be a non-empty array");
    const m = data[0].length;
    if (!Array.isArray(data[0]) || m === 0) throw new Error("each data row must be a non-empty array");
    for (let i = 0; i < data.length; i++) {
        if (!Array.isArray(data[i]) || data[i].length !== m) throw new Error("all rows must have the same length");
        for (let j = 0; j < m; j++) {
            const v = data[i][j];
            if (typeof v !== 'number' || !Number.isFinite(v)) throw new Error("data must contain only finite numbers");
        }
    }
}

function meanVector(data) {
    const n = data.length;
    const m = data[0].length;
    const mean = new Array(m).fill(0);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < m; j++) mean[j] += data[i][j];
    }
    for (let j = 0; j < m; j++) mean[j] /= n;
    return mean;
}

function stdVector(data, mean) {
    const n = data.length;
    const m = data[0].length;
    const std = new Array(m).fill(0);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < m; j++) {
            const d = data[i][j] - mean[j];
            std[j] += d * d;
        }
    }
    const denom = n - 1 || 1;
    for (let j = 0; j < m; j++) {
        std[j] = Math.sqrt(std[j] / denom);
        if (std[j] === 0 || !Number.isFinite(std[j])) std[j] = 1; // avoid division by zero
    }
    return std;
}

function centerData(data, mean, std = null) {
    if (!std) return data.map(row => row.map((v, j) => v - mean[j]));
    return data.map(row => row.map((v, j) => (v - mean[j]) / std[j]));
}

function transpose(A) {
    const m = A.length, n = A[0].length;
    const T = Array.from({ length: n }, () => new Array(m));
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) T[j][i] = A[i][j];
    return T;
}

function matMul(A, B) {
    const m = A.length, p = B[0].length, n = B.length;
    const C = Array.from({ length: m }, () => new Array(p).fill(0));
    for (let i = 0; i < m; i++) {
        for (let k = 0; k < n; k++) {
            const aik = A[i][k];
            for (let j = 0; j < p; j++) C[i][j] += aik * B[k][j];
        }
    }
    return C;
}

function covMatrix(centered) {
    const n = centered.length;
    const Xt = transpose(centered);
    const C = matMul(Xt, centered);
    const scale = 1 / (n - 1 || 1);
    for (let i = 0; i < C.length; i++) for (let j = 0; j < C.length; j++) C[i][j] *= scale;
    // enforce symmetry to avoid small numerical asymmetry
    for (let i = 0; i < C.length; i++) {
        for (let j = i + 1; j < C.length; j++) {
            const s = 0.5 * (C[i][j] + C[j][i]);
            C[i][j] = C[j][i] = s;
        }
    }
    return C;
}

function dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
}

function norm(a) {
    return Math.sqrt(dot(a, a));
}

function scalarVecMul(a, s) {
    return a.map(x => x * s);
}

function matVecMul(M, v) {
    const out = new Array(M.length).fill(0);
    for (let i = 0; i < M.length; i++) {
        let s = 0;
        const row = M[i];
        for (let j = 0; j < row.length; j++) s += row[j] * v[j];
        out[i] = s;
    }
    return out;
}

function subtractScaledOuter(M, scale, v) {
    // M = M - scale * (v v^T)
    for (let i = 0; i < M.length; i++) {
        for (let j = 0; j < M.length; j++) {
            M[i][j] -= scale * v[i] * v[j];
        }
    }
}

function makeRandomGenerator(seed) {
    // simple LCG for reproducibility
    let state = (seed >>> 0) || 1;
    return function() {
        state = (1664525 * state + 1013904223) >>> 0;
        return state / 4294967296;
    };
}

function powerIteration(M, opts = {}) {
    const maxIter = opts.maxIter || 1000;
    const tol = opts.tol || 1e-9;
    const n = M.length;

    // ensure symmetry
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const s = 0.5 * (M[i][j] + M[j][i]); M[i][j] = M[j][i] = s; }

    let rnd = Math.random;
    if (typeof opts.seed === 'number') rnd = makeRandomGenerator(opts.seed);

    let b = new Array(n);
    for (let i = 0; i < n; i++) b[i] = rnd() - 0.5;
    let bNorm = norm(b);
    if (bNorm === 0) { b[0] = 1; bNorm = 1; }
    b = scalarVecMul(b, 1 / bNorm);

    let lambda = dot(b, matVecMul(M, b)); // initial Rayleigh
    for (let iter = 0; iter < maxIter; iter++) {
        const Mb = matVecMul(M, b);
        const MbNorm = norm(Mb);
        if (MbNorm === 0) { lambda = 0; break; }
        const bNext = scalarVecMul(Mb, 1 / MbNorm);
        const lambdaNext = dot(bNext, matVecMul(M, bNext));
        if (!Number.isFinite(lambdaNext)) break;
        if (Math.abs(lambdaNext - lambda) < tol) {
            b = bNext;
            lambda = lambdaNext;
            break;
        }
        b = bNext;
        lambda = lambdaNext;
    }
    // clamp tiny negative due to numeric noise
    if (lambda < 0 && Math.abs(lambda) < 1e-12) lambda = 0;
    return { eigenvector: b, eigenvalue: lambda };
}

function project(centered, components) {
    // centered: n x m, components: k x m
    const n = centered.length;
    const k = components.length;
    const scores = Array.from({ length: n }, () => new Array(k).fill(0));
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < k; j++) {
            scores[i][j] = dot(centered[i], components[j]);
        }
    }
    return scores;
}

/**
 * pca(data, k = null, opts = {})
 * - data: n x m numeric array
 * - k: number of components (default = min(n, m))
 * - opts:
 *    - standardize: boolean (z-score each feature)
 *    - maxIter, tol for power iteration
 *    - seed: integer to make power iteration deterministic
 */
function pca(data, k = null, opts = {}) {
    validateData(data);
    const n = data.length;
    const m = data[0].length;
    k = k == null ? Math.min(n, m) : Math.min(k, m);

    const mean = meanVector(data);
    const std = opts.standardize ? stdVector(data, mean) : null;
    const centered = centerData(data, mean, std);

    const C = covMatrix(centered);
    const traceC = C.reduce((s, row, i) => s + (row[i] || 0), 0) || 1;

    const components = [];
    const eigenvalues = [];
    const M = C.map(row => row.slice()); // copy for deflation

    for (let i = 0; i < k; i++) {
        const piOpts = { maxIter: opts.maxIter, tol: opts.tol, seed: opts.seed != null ? (opts.seed + i) : undefined };
        const { eigenvector, eigenvalue } = powerIteration(M, piOpts);
        if (!Number.isFinite(eigenvalue) || eigenvalue <= 0 || norm(eigenvector) === 0) break;
        // normalize eigenvector to unit length
        const vnorm = norm(eigenvector) || 1;
        const v = scalarVecMul(eigenvector, 1 / vnorm);
        components.push(v);
        eigenvalues.push(eigenvalue);
        // deflate
        subtractScaledOuter(M, eigenvalue, v);
    }

    // compute explained variance ratios using covariance trace for correct proportions
    const explainedVariance = eigenvalues.map(ev => ev / traceC);

    const projected = project(centered, components);

    return {
        components,           // array of k vectors (each length m)
        eigenvalues,          // corresponding eigenvalues
        explainedVariance,    // fraction of total covariance explained by each component
        mean,                 // mean vector used for centering
        std: std,             // std used for standardization (null if not standardized)
        projected             // data projected into k-dim space (n x k)
    };
}

module.exports = { pca };

// Example usage (only runs when executed directly)
if (typeof require !== 'undefined' && require.main === module) {
    const X = [
        [2.5, 2.4],
        [0.5, 0.7],
        [2.2, 2.9],
        [1.9, 2.2],
        [3.1, 3.0],
        [2.3, 2.7],
        [2, 1.6],
        [1, 1.1],
        [1.5, 1.6],
        [1.1, 0.9],
    ];

    const res = pca(X, 2, { standardize: false, seed: 42 });
    console.log('Components:', res.components);
    console.log('Eigenvalues:', res.eigenvalues);
    console.log('Explained variance:', res.explainedVariance);
    console.log('Projected (first 3):', res.projected.slice(0, 3));
}
// ...existing code...