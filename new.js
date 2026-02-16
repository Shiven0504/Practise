
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

function centerData(data, mean) {
    return data.map(row => row.map((v, j) => v - mean[j]));
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
    const scale = 1 / (n - 1);
    for (let i = 0; i < C.length; i++) for (let j = 0; j < C.length; j++) C[i][j] *= scale;
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

function outer(v) {
    const m = v.length;
    const M = Array.from({ length: m }, () => new Array(m));
    for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) M[i][j] = v[i] * v[j];
    return M;
}

function subtractScaledOuter(M, scale, v) {
    // M = M - scale * (v v^T)
    for (let i = 0; i < M.length; i++) {
        for (let j = 0; j < M.length; j++) {
            M[i][j] -= scale * v[i] * v[j];
        }
    }
}

function powerIteration(M, opts = {}) {
    const maxIter = opts.maxIter || 1000;
    const tol = opts.tol || 1e-9;
    const n = M.length;
    let b = new Array(n);
    for (let i = 0; i < n; i++) b[i] = Math.random() - 0.5;
    let bNorm = norm(b);
    if (bNorm === 0) b[0] = 1, bNorm = 1;
    b = scalarVecMul(b, 1 / bNorm);

    let lambda = 0;
    for (let iter = 0; iter < maxIter; iter++) {
        const Mb = matVecMul(M, b);
        const MbNorm = norm(Mb);
        if (MbNorm === 0) break;
        const bNext = scalarVecMul(Mb, 1 / MbNorm);
        const lambdaNext = dot(bNext, matVecMul(M, bNext));
        if (Math.abs(lambdaNext - lambda) < tol) {
            b = bNext;
            lambda = lambdaNext;
            break;
        }
        b = bNext;
        lambda = lambdaNext;
    }
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
 * pca(data, k, opts)
 * data: array of n samples, each sample is array of m numbers
 * k: number of principal components to compute (<= m)
 * opts: { maxIter, tol }
 */
function pca(data, k = null, opts = {}) {
    if (!Array.isArray(data) || data.length === 0) throw new Error("data must be non-empty array");
    const n = data.length;
    const m = data[0].length;
    k = k == null ? Math.min(n, m) : Math.min(k, m);

    const mean = meanVector(data);
    const centered = centerData(data, mean);
    const C = covMatrix(centered);

    const components = [];
    const eigenvalues = [];
    const M = C.map(row => row.slice()); // copy for deflation

    for (let i = 0; i < k; i++) {
        const { eigenvector, eigenvalue } = powerIteration(M, opts);
        if (eigenvalue <= 0 || norm(eigenvector) === 0) break;
        // normalize eigenvector to unit length
        const vnorm = norm(eigenvector);
        const v = scalarVecMul(eigenvector, 1 / vnorm);
        components.push(v);
        eigenvalues.push(eigenvalue);
        // deflate
        subtractScaledOuter(M, eigenvalue, v);
    }

    const totalVar = eigenvalues.reduce((s, x) => s + x, 0) || 0;
    const explainedVariance = eigenvalues.map(ev => ev / (totalVar || 1));

    const projected = project(centered, components);

    return {
        components,           // array of k vectors (each length m)
        eigenvalues,          // corresponding eigenvalues
        explainedVariance,    // fraction of variance explained by each component
        mean,                 // mean vector used for centering
        projected             // data projected into k-dim space (n x k)
    };
}

module.exports = { pca };

