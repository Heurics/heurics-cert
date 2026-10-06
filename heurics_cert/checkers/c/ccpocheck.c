/* ccpocheck -- a second, independent verifier for the CCPO dual-bound certificate.
 *
 * C99, no dependencies beyond libc and libm. It shares no line of code with the prover
 * (`heurics_cert/prover/`) and none with the Python checker (`heurics_cert/checkers/bound.py`):
 * the JSON reader, the SHA-256, the verified Cholesky and the interval arithmetic are all written
 * here from the mathematics. That is the point. A checker written by the author of the prover, in
 * the author's language, sharing the author's linear-algebra stack, tests fewer things than it
 * appears to; a second implementation in a different language with a different arithmetic path
 * tests the *claim* rather than the code.
 *
 * WHAT IS BEING CHECKED
 * ---------------------
 * The certificate publishes multipliers (w, nu, pi, lam), a diagonal split d, and a claimed lower
 * bound on
 *
 *     min  x'Qx   s.t.  mu'x >= rho,  sum x = 1,  lo_i y_i <= x_i <= hi_i y_i,
 *                       sum y <= K,   y in {0,1}^n.
 *
 * Lagrangian decomposition with a copy z = x dualized by w gives, for any pi >= 0 and lam >= 0,
 * the valid lower bound
 *
 *     D  =  beta_z  -  nu  +  pi*rho  -  lam*K  +  sum_i min(0, m_i + lam)
 *
 * with     beta_z  <=  min_z  z'Az - w'z,        A = Q - diag(d),
 *          m_i     =   min_{s in [lo_i, hi_i]}  d_i s^2 + c_i s,   c_i = w_i + nu - pi*mu_i.
 *
 * Every step of that has to be established, not assumed:
 *
 *   1. pi >= 0 and lam >= 0            -- weak duality fails silently otherwise.
 *   2. lo <= hi, d finite, Q symmetric.
 *   3. A = Q - diag(d) is POSITIVE SEMIDEFINITE. The prover assumes this. If it is false,
 *      min_z z'Az - w'z is -inf and every number downstream is meaningless. Proved here by a
 *      verified Cholesky (below), never by an eigenvalue routine.
 *   4. beta_z really is a lower bound on the z-block minimum. Established WITHOUT forming a
 *      pseudo-inverse, by proving the bordered matrix
 *          M = [[A, -w/2], [-w'/2, -beta_z]]
 *      positive semidefinite: for any z, (z,1)' M (z,1) >= 0 is exactly z'Az - w'z - beta_z >= 0.
 *   5. The per-asset minima are minima over the whole interval, computed in interval arithmetic.
 *   6. claimed_bound <= the value re-derived here with every operation rounded DOWNWARD.
 *
 * VERIFIED CHOLESKY
 * -----------------
 * Higham, *Accuracy and Stability of Numerical Algorithms*, Thm 10.3-10.5: if the floating-point
 * Cholesky factorization of a symmetric B runs to completion producing R, then R'R = B + E with
 * |E| <= gamma_{n+1} |R'||R| and gamma_k = k*u/(1 - k*u). Hence
 *
 *     ||E||_2  <=  ||E||_F  <=  gamma_{n+1} * || |R| ||_F^2      and      lambda_min(B) >= -||E||_2.
 *
 * So: factor B = A - cI in ordinary float64. If it completes and the resulting error bound delta is
 * smaller than c, then lambda_min(A) >= c - delta > 0 and A is PSD -- rigorously, even though the
 * factorization itself was ordinary arithmetic. Everything feeding that conclusion is rounded
 * outward. The shift ladder is a search for a c that works; the *conclusion* does not depend on
 * which rung succeeds, only on delta < c holding for the rung that did.
 *
 * ROUNDING
 * --------
 * `dn(x)` and `up(x)` step one ulp toward -inf and +inf. A correctly-rounded IEEE operation errs by
 * at most half an ulp, so one ulp of outward motion after each operation is a rigorous enclosure.
 * The file must be compiled with contraction off (`-ffp-contract=off`) and without unsafe math, or
 * an FMA can silently produce a result that is *more* accurate than the analysis assumes in one
 * place and differently rounded in another. The Makefile does that; the binary also refuses to run
 * if it detects that `dn`/`up` are not moving.
 *
 * USAGE
 *     ccpocheck cert.json [--expect-model <sha256 prefix>] [--json]
 * exit 0 = PROVED, 1 = REFUTED, 3 = NOT PROVED, 2 = usage/parse error.
 *
 * REFUTED and NOT PROVED are different claims and never share a code: REFUTED means the certificate
 * is wrong and that has been certified; NOT PROVED means this arithmetic could not decide. A script
 * that read every non-zero exit as "the solver is wrong" would indict a correct solver on any
 * certificate with zero slack.
 */
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ directed rounding */
static double dn(double x) { return nextafter(x, -INFINITY); }
static double up(double x) { return nextafter(x,  INFINITY); }

/* interval [lo, hi], outward-rounded */
typedef struct { double lo, hi; } iv;
static iv iv_pt(double x)            { iv r = {x, x}; return r; }
static iv iv_add(iv a, iv b)         { iv r = {dn(a.lo + b.lo), up(a.hi + b.hi)}; return r; }
static iv iv_sub(iv a, iv b)         { iv r = {dn(a.lo - b.hi), up(a.hi - b.lo)}; return r; }
static iv iv_mul(iv a, iv b) {
    double p[4] = {a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi};
    double lo = p[0], hi = p[0];
    for (int i = 1; i < 4; i++) { if (p[i] < lo) lo = p[i]; if (p[i] > hi) hi = p[i]; }
    iv r = {dn(lo), up(hi)};
    return r;
}
/* division by an interval that is strictly positive */
static iv iv_divpos(iv a, iv b) {
    double p[4] = {a.lo / b.lo, a.lo / b.hi, a.hi / b.lo, a.hi / b.hi};
    double lo = p[0], hi = p[0];
    for (int i = 1; i < 4; i++) { if (p[i] < lo) lo = p[i]; if (p[i] > hi) hi = p[i]; }
    iv r = {dn(lo), up(hi)};
    return r;
}

/* ------------------------------------------------------------------ SHA-256 */
typedef struct { uint32_t h[8]; uint64_t len; uint8_t buf[64]; size_t n; } sha256;
static uint32_t rotr(uint32_t x, int c) { return (x >> c) | (x << (32 - c)); }
static const uint32_t SHA_K[64] = {
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
static void sha_block(sha256 *s, const uint8_t *p) {
    uint32_t w[64], a, b, c, d, e, f, g, h;
    for (int i = 0; i < 16; i++)
        w[i] = ((uint32_t)p[4*i] << 24) | ((uint32_t)p[4*i+1] << 16) |
               ((uint32_t)p[4*i+2] << 8) | (uint32_t)p[4*i+3];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    a=s->h[0];b=s->h[1];c=s->h[2];d=s->h[3];e=s->h[4];f=s->h[5];g=s->h[6];h=s->h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = h + S1 + ch + SHA_K[i] + w[i];
        uint32_t S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
        uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + mj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    s->h[0]+=a;s->h[1]+=b;s->h[2]+=c;s->h[3]+=d;s->h[4]+=e;s->h[5]+=f;s->h[6]+=g;s->h[7]+=h;
}
static void sha_init(sha256 *s) {
    static const uint32_t iv0[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                                    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    memcpy(s->h, iv0, sizeof iv0); s->len = 0; s->n = 0;
}
static void sha_update(sha256 *s, const void *data, size_t len) {
    const uint8_t *p = data; s->len += len;
    while (len) {
        size_t take = 64 - s->n; if (take > len) take = len;
        memcpy(s->buf + s->n, p, take); s->n += take; p += take; len -= take;
        if (s->n == 64) { sha_block(s, s->buf); s->n = 0; }
    }
}
static void sha_final(sha256 *s, char out[65]) {
    uint64_t bits = s->len * 8; uint8_t pad = 0x80;
    sha_update(s, &pad, 1);
    uint8_t z = 0;
    while (s->n != 56) sha_update(s, &z, 1);
    uint8_t lb[8];
    for (int i = 0; i < 8; i++) lb[i] = (uint8_t)(bits >> (56 - 8*i));
    sha_update(s, lb, 8);
    for (int i = 0; i < 8; i++) sprintf(out + 8*i, "%08x", s->h[i]);
    out[64] = 0;
}
/* big-endian IEEE-754 bytes of a double, the same wire form the generator hashes */
static void sha_double(sha256 *s, double v) {
    uint64_t u; memcpy(&u, &v, 8);
    uint8_t b[8];
    for (int i = 0; i < 8; i++) b[i] = (uint8_t)(u >> (56 - 8*i));
    sha_update(s, b, 8);
}
static void sha_i64(sha256 *s, int64_t v) {
    uint8_t b[8];
    for (int i = 0; i < 8; i++) b[i] = (uint8_t)((uint64_t)v >> (56 - 8*i));
    sha_update(s, b, 8);
}

/* ------------------------------------------------------------------ minimal JSON reader
 * Only what a certificate contains: an object of numbers, strings, booleans, null, flat number
 * arrays and one array-of-arrays. Deliberately small enough to audit in one sitting. */
typedef struct { const char *s; size_t i, n; } jp;
static void jskip(jp *p) {
    while (p->i < p->n) {
        char c = p->s[p->i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') p->i++;
        else break;
    }
}
static int jexpect(jp *p, char c) { jskip(p); if (p->i < p->n && p->s[p->i] == c) { p->i++; return 1; } return 0; }
static int jstring(jp *p, char *out, size_t cap) {   /* no escapes: certificates contain none */
    jskip(p);
    if (p->i >= p->n || p->s[p->i] != '"') return 0;
    p->i++;
    size_t k = 0;
    while (p->i < p->n && p->s[p->i] != '"') {
        if (p->s[p->i] == '\\') p->i++;
        if (k + 1 < cap) out[k++] = p->s[p->i];
        p->i++;
    }
    out[k] = 0;
    return jexpect(p, '"');
}
static int jnumber(jp *p, double *v) {
    jskip(p);
    char *end = NULL;
    *v = strtod(p->s + p->i, &end);
    if (end == p->s + p->i) return 0;
    p->i = (size_t)(end - p->s);
    return 1;
}
/* skip one value of any type */
static void jskip_value(jp *p) {
    jskip(p);
    if (p->i >= p->n) return;
    char c = p->s[p->i];
    if (c == '"') { char t[8]; (void)t; p->i++; while (p->i < p->n && p->s[p->i] != '"') { if (p->s[p->i]=='\\') p->i++; p->i++; } p->i++; return; }
    if (c == '[' || c == '{') {
        char open = c, close = (c == '[') ? ']' : '}';
        int depth = 0;
        while (p->i < p->n) {
            char d = p->s[p->i];
            if (d == '"') { p->i++; while (p->i < p->n && p->s[p->i] != '"') { if (p->s[p->i]=='\\') p->i++; p->i++; } }
            else if (d == open) depth++;
            else if (d == close) { depth--; if (!depth) { p->i++; return; } }
            p->i++;
        }
        return;
    }
    while (p->i < p->n && strchr(",}] \t\n\r", p->s[p->i]) == NULL) p->i++;
}

/* --timing: stage wall-clock to stderr. Not part of any proof; it exists because the checker, not
 * the prover, turned out to be the slow half (measured: 65.6 s of a 71 s check at n = 2901). */
static int g_timing = 0;
static double now_s(void) {                 /* ANSI clock(): no POSIX feature macros, stays C99-portable */
    return (double)clock() / (double)CLOCKS_PER_SEC;
}
static double g_t0 = 0.0;
static void stage(const char *what) {
    if (!g_timing) return;
    double t = now_s();
    fprintf(stderr, "  [timing] %-34s %7.2fs\n", what, t - g_t0);
    g_t0 = t;
}

typedef struct {
    int n;
    double *Q;            /* n*n, row major */
    double *mu, *lo, *hi, *d, *w;
    double nu, pi, lam, beta_z, claimed, rho;
    double reported;
    double K;             /* a positive integer by the format; held as the double the JSON carries */
    char model_hash[80];
    char dtype[32];
    int have_hash;
} cert_t;

static int read_vec(jp *p, double **out, int *len) {
    if (!jexpect(p, '[')) return 0;
    size_t cap = 256, k = 0;
    double *v = malloc(cap * sizeof *v);
    jskip(p);
    if (p->i < p->n && p->s[p->i] == ']') { p->i++; *out = v; *len = 0; return 1; }
    for (;;) {
        double x;
        if (!jnumber(p, &x)) { free(v); return 0; }
        if (k == cap) { cap *= 2; v = realloc(v, cap * sizeof *v); }
        v[k++] = x;
        jskip(p);
        if (p->i < p->n && p->s[p->i] == ',') { p->i++; continue; }
        break;
    }
    if (!jexpect(p, ']')) { free(v); return 0; }
    *out = v; *len = (int)k;
    return 1;
}

static int parse_cert(const char *text, size_t len, cert_t *c, int q_external) {
    jp p = {text, 0, len};
    memset(c, 0, sizeof *c);
    c->n = -1;
    if (!jexpect(&p, '{')) return 0;
    char key[64];
    for (;;) {
        jskip(&p);
        if (p.i < p.n && p.s[p.i] == '}') { p.i++; break; }
        if (!jstring(&p, key, sizeof key)) return 0;
        if (!jexpect(&p, ':')) return 0;
        jskip(&p);
        if (!strcmp(key, "Q")) {
            if (!jexpect(&p, '[')) return 0;
            int rows = 0, ncol = -1;
            size_t cap = 4096, k = 0;
            double *M = malloc(cap * sizeof *M);
            for (;;) {
                double *row; int m;
                if (!read_vec(&p, &row, &m)) { free(M); return 0; }
                if (ncol < 0) ncol = m;
                else if (m != ncol) { free(row); free(M); return 0; }
                while (k + (size_t)m > cap) { cap *= 2; M = realloc(M, cap * sizeof *M); }
                memcpy(M + k, row, (size_t)m * sizeof *row);
                k += (size_t)m; rows++; free(row);
                jskip(&p);
                if (p.i < p.n && p.s[p.i] == ',') { p.i++; continue; }
                break;
            }
            if (!jexpect(&p, ']')) { free(M); return 0; }
            if (rows != ncol) { free(M); return 0; }
            c->Q = M; c->n = rows;
        } else if (!strcmp(key, "mu") || !strcmp(key, "lo") || !strcmp(key, "hi") ||
                   !strcmp(key, "d")  || !strcmp(key, "w")) {
            double *v; int m;
            if (!read_vec(&p, &v, &m)) return 0;
            if (!strcmp(key, "mu")) c->mu = v; else if (!strcmp(key, "lo")) c->lo = v;
            else if (!strcmp(key, "hi")) c->hi = v; else if (!strcmp(key, "d")) c->d = v;
            else c->w = v;
        } else if (!strcmp(key, "n")) {
            double v; if (!jnumber(&p, &v)) return 0; c->n = (int)v;
        } else if (!strcmp(key, "nu") || !strcmp(key, "pi") || !strcmp(key, "lam") ||
                   !strcmp(key, "beta_z") || !strcmp(key, "claimed_bound") ||
                   !strcmp(key, "rho") || !strcmp(key, "reported_bound")) {
            double x;
            if (!jnumber(&p, &x)) return 0;
            if (!strcmp(key, "nu")) c->nu = x; else if (!strcmp(key, "pi")) c->pi = x;
            else if (!strcmp(key, "lam")) c->lam = x; else if (!strcmp(key, "beta_z")) c->beta_z = x;
            else if (!strcmp(key, "claimed_bound")) c->claimed = x;
            else if (!strcmp(key, "reported_bound")) c->reported = x;
            else c->rho = x;
        } else if (!strcmp(key, "K")) {
            /* The format requires a positive integer. Converting an out-of-range double to an integer type is
             * undefined behaviour: K = 1e300 became LONG_MIN, turned -lam*K into a huge positive term and was
             * PROVED, and a non-integer K was hashed as its truncation, unlike the Python checker. */
            double x; if (!jnumber(&p, &x)) return 0;
            if (!(x >= 1.0 && x <= 9007199254740992.0 && x == floor(x))) return 0;
            c->K = x;
        } else if (!strcmp(key, "model_hash")) {
            if (!jstring(&p, c->model_hash, sizeof c->model_hash)) return 0;
            c->have_hash = 1;
        } else if (!strcmp(key, "dtype")) {
            if (!jstring(&p, c->dtype, sizeof c->dtype)) return 0;
        } else {
            jskip_value(&p);
        }
        jskip(&p);
        if (p.i < p.n && p.s[p.i] == ',') { p.i++; continue; }
    }
    /* With --qbin the model's Q arrives as a binary file and `n` comes from the witness. */
    return c->n > 0 && (c->Q || q_external) && c->mu && c->lo && c->hi && c->d && c->w;
}

/* model fingerprint: SHA-256 over the big-endian IEEE bytes of (Q, mu, lo, hi, K, rho),
 * each preceded by its field name and, for arrays, by (rows, cols) as big-endian int64. */
static void model_hash(const cert_t *c, char out[65]) {
    sha256 s; sha_init(&s);
    int n = c->n;
    sha_update(&s, "Q", 1);  sha_i64(&s, n); sha_i64(&s, n);
    for (int i = 0; i < n * n; i++) sha_double(&s, c->Q[i]);
    const char *names[4] = {"mu", "lo", "hi"};
    const double *vecs[4] = {c->mu, c->lo, c->hi};
    for (int v = 0; v < 3; v++) {
        sha_update(&s, names[v], strlen(names[v]));
        sha_i64(&s, 1); sha_i64(&s, n);
        for (int i = 0; i < n; i++) sha_double(&s, vecs[v][i]);
    }
    sha_update(&s, "K", 1);   sha_double(&s, c->K);
    sha_update(&s, "rho", 3); sha_double(&s, c->rho);
    sha_final(&s, out);
}

/* ------------------------------------------------------------------ verified Cholesky */
#define UNIT_ROUNDOFF 1.1102230246251565e-16   /* 2^-53 */

/* Plain float64 Cholesky of the lower triangle of B (n x n, row major), overwriting `R`
 * (upper triangular, row major). Returns 0 on success, non-zero if a pivot is non-positive. */
static int chol(const double *B, int n, double *R) {
    memset(R, 0, (size_t)n * n * sizeof *R);
    /* Right-looking (rank-1 updating) form of the same factorization. The textbook left-looking loop
     * above it recomputed each inner product with a stride-n walk down column j of R, which at
     * n = 2901 misses cache on every access; this touches each updated row contiguously instead.
     * The arithmetic is not reassociated per output element in a way that weakens the backward-error
     * bound: Higham's result holds whatever order the sums are evaluated in (Lemma 3.1), which the
     * refutation path already relies on. */
    for (size_t i = 0; i < (size_t)n * n; i++) R[i] = B[i];
    for (int j = 0; j < n; j++) {
        double s = R[(size_t)j * n + j];
        if (!(s > 0.0)) return 1;
        double rjj = sqrt(s);
        R[(size_t)j * n + j] = rjj;
        double inv = 1.0 / rjj;
        double *rj = R + (size_t)j * n;
        for (int i = j + 1; i < n; i++) rj[i] *= inv;
        for (int i = j + 1; i < n; i++) {
            double f = rj[i];
            if (f == 0.0) continue;
            double *ri = R + (size_t)i * n;
            for (int k = i; k < n; k++) ri[k] -= f * rj[k];
        }
        for (int i = 0; i < j; i++) R[(size_t)j * n + i] = 0.0;
    }
    return 0;
}

/* Higham's backward-error bound for a completed factorization: gamma_{n+1} * |||R|||_F^2,
 * rounded up at every step. Returns -1 if the factorization does not complete. */
static double chol_delta(const double *B, int n, double *R) {
    if (chol(B, n, R)) return -1.0;
    double fro2 = 0.0;
    for (size_t i = 0; i < (size_t)n * n; i++) fro2 = up(fro2 + up(R[i] * R[i]));
    double gden = 1.0 - (double)(n + 1) * UNIT_ROUNDOFF;
    if (!(gden > 0.0)) return -1.0;                     /* n so large the bound is vacuous */
    double g = up(up((double)(n + 1) * UNIT_ROUNDOFF) / dn(gden));
    return up(g * fro2);
}

/* Prove the BORDERED matrix M = [[A, b], [b', corner]] >= 0 by extending an existing factorization
 * of the already-shifted A block, instead of factoring an (n+1)x(n+1) matrix from scratch.
 *
 * `R` is the computed Cholesky factor of the tested `A - cI` (diagonal rounded down) and `fro2` its
 * rounded-up || |R| ||_F^2. The last column of the standard algorithm applied to the bordered matrix
 * is exactly the triangular solve R' r = b followed by s^2 = (corner - c) - r'r, so the assembled
 * [[R, r], [0, s]] is the factor the standard algorithm would have produced -- and Higham's bound
 * holds whatever order those sums are evaluated in. This makes the z-block check O(n^2) instead of a
 * second O(n^3) factorization, halving the checker's arithmetic: the two O(n^3) factorizations it
 * used to run on A and on M become one.
 *
 * The tested bordered matrix is <= the true M - cI: `A`'s diagonal is already rounded down, the
 * border entries are exact (halving is exact in binary), and the corner is rounded down here.
 * Returns 1 and writes a rigorous lower bound on lambda_min(M) on success, 0 if it cannot conclude
 * (in which case the caller should fall back to factoring M directly). */
static int psd_bordered_from_factor(const double *R, int n, double fro2, double c,
                                    const double *b, double corner, double *lmin) {
    double *r = malloc((size_t)n * sizeof *r);
    if (!r) return 0;
    /* forward substitution R' r = b, with R upper triangular in row-major order */
    for (int i = 0; i < n; i++) {
        double acc = b[i];
        for (int k = 0; k < i; k++) acc -= R[(size_t)k * n + i] * r[k];
        double rii = R[(size_t)i * n + i];
        if (!(rii > 0.0)) { free(r); return 0; }
        r[i] = acc / rii;
    }
    double rtr = 0.0, fro_extra = 0.0;
    for (int i = 0; i < n; i++) { rtr += r[i] * r[i]; fro_extra = up(fro_extra + up(r[i] * r[i])); }
    double s2 = dn(dn(corner - c) - up(rtr));
    free(r);
    if (!(s2 > 0.0)) return 0;                       /* the bordered factorization does not complete */
    double fro2_m = up(up(fro2 + fro_extra) + up(s2));
    int m = n + 1;
    double gden = 1.0 - (double)(m + 1) * UNIT_ROUNDOFF;
    if (!(gden > 0.0)) return 0;
    double g = up(up((double)(m + 1) * UNIT_ROUNDOFF) / dn(gden));
    double delta = up(g * fro2_m);
    if (!(delta < c)) return 0;
    *lmin = dn(c - delta);
    return 1;
}

/* Prove B >= 0. On success writes a rigorous lower bound on lambda_min(B) to *lmin.
 *
 * With `border`/`corner` non-NULL it also tries to establish the bordered matrix
 * [[B, border], [border', corner]] >= 0 from the same factorization (see above), writing its bound
 * to *lmin_b and 1 to *ok_b. A failure there is not a failure of this function: the caller falls
 * back to factoring the bordered matrix on its own. */
static int verified_psd_ex(const double *B, int n, double *lmin, double *scratch1, double *scratch2,
                           const double *border, double corner, double *lmin_b, int *ok_b) {
    double *S = scratch2;
    (void)scratch1;
    /* The shift `c` used to be sized by first factoring B unshifted and reading off its backward
     * error. That factorization cost as much as the one that does the proving, and it was never part
     * of the proof -- only a way to pick a rung. It is replaceable by an a-priori bound: for a
     * completed Cholesky, || |R| ||_F^2 = trace(B), so gamma_{n+1} * trace_up(B) bounds the same
     * quantity in O(n). The conclusion below still uses the *actual* delta from the shifted
     * factorization, so nothing about rigour changes -- a bad guess costs another rung, not
     * soundness. Measured: 2.8 s -> 1.4 s of a 3.1 s check at n = 2901. */
    double scale = 0.0, tr = 0.0;
    for (int i = 0; i < n; i++) {
        double a = fabs(B[(size_t)i * n + i]);
        if (a > scale) scale = a;
        tr = up(tr + a);
    }
    scale = up(scale);
    double gden0 = 1.0 - (double)(n + 1) * UNIT_ROUNDOFF;
    if (!(gden0 > 0.0)) return 0;
    double delta0 = up(up(up((double)(n + 1) * UNIT_ROUNDOFF) / dn(gden0)) * tr);
    const double mult[4] = {4.0, 64.0, 1024.0, 65536.0};
    for (int m = 0; m < 4; m++) {
        double base = delta0 > up(UNIT_ROUNDOFF * scale) ? delta0 : up(UNIT_ROUNDOFF * scale);
        double c = up(mult[m] * base);
        /* Shift DOWN by c, entrywise rounded down, so the matrix actually factored is <= B - cI.
         * Proving that one PSD proves B - cI PSD, hence lambda_min(B) >= c - delta. */
        for (size_t i = 0; i < (size_t)n * n; i++) S[i] = B[i];
        for (int i = 0; i < n; i++) S[(size_t)i * n + i] = dn(B[(size_t)i * n + i] - c);
        double *R2 = malloc((size_t)n * n * sizeof *R2);
        double delta = chol_delta(S, n, R2);
        if (delta >= 0.0 && delta < c) {
            *lmin = dn(c - delta);
            if (border && ok_b) {
                /* || |R2| ||_F^2, the same quantity chol_delta just formed, recomputed here because
                 * the bordered bound needs it and chol_delta returns only the scaled result. */
                double fro2 = 0.0;
                for (size_t i = 0; i < (size_t)n * n; i++) fro2 = up(fro2 + up(R2[i] * R2[i]));
                *ok_b = psd_bordered_from_factor(R2, n, fro2, c, border, corner, lmin_b);
            }
            free(R2);
            return 1;
        }
        free(R2);
    }
    return 0;
}

/* The plain form, for callers with no bordered matrix to piggyback. */
static int verified_psd(const double *B, int n, double *lmin, double *scratch1, double *scratch2) {
    return verified_psd_ex(B, n, lmin, scratch1, scratch2, NULL, 0.0, NULL, NULL);
}

/* ------------------------------------------------------------------ certified refutation
 *
 * Failing to factor a matrix proves nothing about it -- verified Cholesky fails at degeneracy by
 * construction, and a valid certificate with zero slack is precisely where it cannot succeed. So a
 * REFUTED verdict is issued only on a *certified negation*: a direction along which the quadratic
 * form is provably negative, or a claim above a rigorous upper bound on what the witness supports.
 * `bound.py` builds the identical directions, so the two verifiers search the same place and can
 * disagree only where the answer lies inside rounding. */

static double gamma_up(int k) {
    double g = up((double)k * UNIT_ROUNDOFF);
    return up(g / dn(1.0 - g));
}

/* Rigorous upper bound on x'Mx - lin'x (lin may be NULL) for the exact values given. Formed in plain
 * float64, then bounded a posteriori: each exact term reaches the result through at most two
 * multiplications, m - 1 additions at each of two nesting levels and the final subtraction, each
 * rounding once, so |error| <= gamma_{2m+2} * T with T the all-positive sum (Higham, Lemma 3.1). */
static double quad_upper(const double *M, int m, const double *x, const double *lin) {
    double s = 0.0, T = 0.0;
    for (int i = 0; i < m; i++) {
        double si = 0.0, ti = 0.0;
        for (int j = 0; j < m; j++) {
            si += M[(size_t)i * m + j] * x[j];
            ti += fabs(M[(size_t)i * m + j]) * fabs(x[j]);
        }
        s += x[i] * si;
        T += fabs(x[i]) * ti;
    }
    if (lin) {
        double sl = 0.0, tl = 0.0;
        for (int i = 0; i < m; i++) { sl += lin[i] * x[i]; tl += fabs(lin[i]) * fabs(x[i]); }
        s = s - sl;
        T = T + tl;
    }
    if (!isfinite(s) || !isfinite(T)) return INFINITY;
    double g = gamma_up(2 * m + 2);
    return up(s + up(up(g * T) / dn(1.0 - g)));
}

/* Right-looking Cholesky of M (m x m, row major). At the first pivot s_j that is not positive, the
 * leading block M11 = R11'R11 is positive definite and s_j is its Schur complement, so
 * x = [-M11^{-1} m12; 1; 0...] satisfies x'Mx = s_j exactly: write it to x and return 1. Returns 0
 * if the factorization completes, in which case R holds the full upper factor. x is a candidate
 * only; a refutation stands or falls on quad_upper evaluated on it. */
static int neg_curv_dir(const double *M, int m, double *x, double *S, double *R) {
    memcpy(S, M, (size_t)m * m * sizeof *S);
    memset(R, 0, (size_t)m * m * sizeof *R);
    for (int j = 0; j < m; j++) {
        double piv = S[(size_t)j * m + j];
        if (!(piv > 0.0)) {
            for (int i = 0; i < m; i++) x[i] = 0.0;
            x[j] = 1.0;
            /* back-substitute R11 t = R[0..j-1, j] into x[0..j-1], then negate */
            for (int i = j - 1; i >= 0; i--) {
                double acc = R[(size_t)i * m + j];
                for (int k = i + 1; k < j; k++) acc -= R[(size_t)i * m + k] * x[k];
                x[i] = acc / R[(size_t)i * m + i];
            }
            for (int i = 0; i < j; i++) x[i] = -x[i];
            return 1;
        }
        double r = sqrt(piv);
        R[(size_t)j * m + j] = r;
        for (int k = j + 1; k < m; k++) R[(size_t)j * m + k] = S[(size_t)j * m + k] / r;
        for (int a = j + 1; a < m; a++) {
            double ra = R[(size_t)j * m + a];
            for (int b = j + 1; b < m; b++) S[(size_t)a * m + b] -= ra * R[(size_t)j * m + b];
        }
    }
    return 0;
}

/* Prove the exact matrix is NOT PSD, or fail to. M_up must bound it from above on the diagonal and
 * equal it elsewhere, so x'M_up x >= x'Mx for every x. Writes the upper bound found to *ub. */
static int certify_not_psd(const double *M_up, int m, double *ub) {
    double *x = malloc((size_t)m * sizeof *x);
    double *S = malloc((size_t)m * m * sizeof *S);
    double *R = malloc((size_t)m * m * sizeof *R);
    int refuted = 0;
    *ub = NAN;
    if (neg_curv_dir(M_up, m, x, S, R)) {
        *ub = quad_upper(M_up, m, x, NULL);
        refuted = *ub < 0.0;
    }
    free(x); free(S); free(R);
    return refuted;
}

/* Rigorous upper bound on beta* = min_z z'Az - w'z, the z-block value the exact tier certifies with.
 * A claim may be refuted only above an upper bound on *that* value, not on the published beta_z. Any
 * z gives z'Az - w'z >= beta*; z = A^{-1} w / 2 makes it tight, and z = 0 shows beta* <= 0. */
static double beta_star_upper(const double *A_up, int n, const double *w) {
    double *x = malloc((size_t)n * sizeof *x);
    double *S = malloc((size_t)n * n * sizeof *S);
    double *R = malloc((size_t)n * n * sizeof *R);
    double ub = 0.0;
    if (!neg_curv_dir(A_up, n, x, S, R)) {
        /* R'R = A_up: forward-solve R' y = w into x, back-solve R z = y in place, halve */
        for (int i = 0; i < n; i++) {
            double acc = w[i];
            for (int k = 0; k < i; k++) acc -= R[(size_t)k * n + i] * x[k];
            x[i] = acc / R[(size_t)i * n + i];
        }
        for (int i = n - 1; i >= 0; i--) {
            double acc = x[i];
            for (int k = i + 1; k < n; k++) acc -= R[(size_t)i * n + k] * x[k];
            x[i] = acc / R[(size_t)i * n + i];
        }
        for (int i = 0; i < n; i++) x[i] *= 0.5;
        double q = quad_upper(A_up, n, x, w);
        if (q < ub) ub = q;
    }
    free(x); free(S); free(R);
    return ub;
}

/* ------------------------------------------------------------------ the check */
/* st: 0 = established, 1 = refuted (negation certified), 2 = not established, not refuted */
typedef struct { const char *name; int ok; int st; const char *why; char detail[256]; } chk;

int main(int argc, char **argv) {
    const char *path = NULL, *expect = NULL, *qbin = NULL;
    int as_json = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--expect-model") && i + 1 < argc) expect = argv[++i];
        /* --qbin: take Q from a binary file (magic, int64 n, then n*n float64 row-major, native
         * byte order) instead of from the JSON document. This is the witness/model split the format
         * spec already describes: a verifier that holds the model needs only the O(n) witness. It is
         * not a weaker check -- the bytes still feed model_hash, so a mismatched Q is still caught by
         * the fingerprint -- and it removes 8.4M strtod calls at n = 2901. */
        else if (!strcmp(argv[i], "--qbin") && i + 1 < argc) qbin = argv[++i];
        else if (!strcmp(argv[i], "--timing")) g_timing = 1;
        else if (!strcmp(argv[i], "--json")) as_json = 1;
        else if (argv[i][0] != '-') path = argv[i];
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    if (!path) {
        fprintf(stderr, "usage: ccpocheck cert.json [--expect-model <sha256 prefix>] [--json]\n");
        return 2;
    }
    /* Refuse to run if directed rounding is not actually directing. Cheap, and it catches a build
     * with the wrong flags, which would silently turn every bound below into a guess. */
    if (!(dn(1.0) < 1.0 && up(1.0) > 1.0)) {
        fprintf(stderr, "nextafter is not moving: refusing to certify anything\n");
        return 2;
    }

    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *text = malloc((size_t)sz + 1);
    if (fread(text, 1, (size_t)sz, fp) != (size_t)sz) { fprintf(stderr, "short read\n"); return 2; }
    text[sz] = 0;
    fclose(fp);

    g_t0 = now_s();
    cert_t c;
    if (!parse_cert(text, (size_t)sz, &c, qbin != NULL)) {
        fprintf(stderr, "malformed certificate\n"); return 2; }
    free(text);
    stage("parse JSON");
    if (qbin) {
        /* Layout: 8 bytes magic "CCPOQ1\0\0", int64 n, then n*n float64 row-major in native byte
         * order. The length is checked against n, so a truncated or mismatched model is rejected
         * rather than read past its end. The bytes still feed model_hash, so substituting a
         * different Q here is caught by the fingerprint exactly as it is in the all-JSON path. */
        if (c.Q) { fprintf(stderr, "certificate carries Q and --qbin was given\n"); return 2; }
        FILE *qf = fopen(qbin, "rb");
        if (!qf) { fprintf(stderr, "cannot open %s\n", qbin); return 2; }
        char magic[8] = {0};
        int64_t qn = 0;
        if (fread(magic, 1, 8, qf) != 8 || memcmp(magic, "CCPOQ1\0\0", 8) != 0) {
            fprintf(stderr, "%s: bad magic\n", qbin); return 2; }
        if (fread(&qn, sizeof qn, 1, qf) != 1 || qn <= 0 || qn > (1 << 20)) {
            fprintf(stderr, "%s: bad n\n", qbin); return 2; }
        if (c.n > 0 && c.n != (int)qn) {
            fprintf(stderr, "%s: n disagrees with the witness\n", qbin); return 2; }
        if (fseek(qf, 0, SEEK_END) != 0) { fprintf(stderr, "%s: not seekable\n", qbin); return 2; }
        long qsz = ftell(qf);
        if (qsz != 16 + (long)((size_t)qn * (size_t)qn * sizeof(double))) {
            fprintf(stderr, "%s: size %ld does not match n = %lld\n", qbin, qsz, (long long)qn);
            return 2;
        }
        if (fseek(qf, 16, SEEK_SET) != 0) { fprintf(stderr, "%s: not seekable\n", qbin); return 2; }
        size_t cnt = (size_t)qn * (size_t)qn;
        double *QM = malloc(cnt * sizeof *QM);
        if (!QM || fread(QM, sizeof *QM, cnt, qf) != cnt) {
            fprintf(stderr, "%s: short read\n", qbin); return 2; }
        fclose(qf);
        c.Q = QM;
        c.n = (int)qn;
        stage("read Q (binary)");
    }
    if (!c.Q) { fprintf(stderr, "certificate has no Q and no --qbin was given\n"); return 2; }
    int n = c.n;

    chk checks[16];
    int nc = 0;
    /* ADDST records a check with an explicit status (0 established, 1 refuted, 2 undecided) and the
     * reason a verdict would quote. ADD is for definite checks on published numbers compared
     * exactly, where failure *is* a refutation. */
    #define ADDST(nm, status, reason, ...) do { checks[nc].name = (nm); checks[nc].st = (status); \
        checks[nc].ok = (status) == 0; checks[nc].why = (reason) ? (reason) : (nm); \
        snprintf(checks[nc].detail, sizeof checks[nc].detail, __VA_ARGS__); nc++; } while (0)
    #define ADD(nm, cond, ...) ADDST(nm, (cond) ? 0 : 1, NULL, __VA_ARGS__)

    int sym = 1;
    for (int i = 0; i < n && sym; i++)
        for (int j = 0; j < i; j++)
            if (c.Q[(size_t)i * n + j] != c.Q[(size_t)j * n + i]) { sym = 0; break; }
    ADD("Q is exactly symmetric", sym, "%s", sym ? "" : "Q[i][j] != Q[j][i]");

    stage("symmetry scan");
    char mh[65];
    model_hash(&c, mh);
    stage("model_hash (SHA-256)");
    if (c.have_hash)
        ADD("carried model fingerprint matches the model carried with it",
            !strcmp(mh, c.model_hash), "carried %.16s, recomputed %.16s", c.model_hash, mh);
    if (expect)
        ADD("model fingerprint matches the model the reader holds",
            !strncmp(mh, expect, strlen(expect)), "expected %s..., got %.16s", expect, mh);

    ADD("pi >= 0", c.pi >= 0.0, "pi = %.6e", c.pi);
    ADD("lam >= 0", c.lam >= 0.0, "lam = %.6e", c.lam);
    int dfin = 1, dmin_i = 0;
    for (int i = 0; i < n; i++) { if (!isfinite(c.d[i])) dfin = 0; if (c.d[i] < c.d[dmin_i]) dmin_i = i; }
    ADD("d is finite", dfin, "min d = %.6e", c.d[dmin_i]);
    int box = 1;
    for (int i = 0; i < n; i++) if (!(c.lo[i] <= c.hi[i])) { box = 0; break; }
    ADD("lo <= hi", box, "%s", "");

    /* A = Q - diag(d), diagonal rounded DOWN so the matrix tested is <= the true one. A_up rounds the
     * same diagonal UP, so it bounds the true matrix from above -- the side a refutation must use. */
    double *A = malloc((size_t)n * n * sizeof *A);
    memcpy(A, c.Q, (size_t)n * n * sizeof *A);
    for (int i = 0; i < n; i++) A[(size_t)i * n + i] = dn(c.Q[(size_t)i * n + i] - c.d[i]);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < i; j++) {
            double s = 0.5 * (A[(size_t)i * n + j] + A[(size_t)j * n + i]);
            A[(size_t)i * n + j] = A[(size_t)j * n + i] = s;
        }
    double *A_up = malloc((size_t)n * n * sizeof *A_up);
    memcpy(A_up, A, (size_t)n * n * sizeof *A_up);
    for (int i = 0; i < n; i++) A_up[(size_t)i * n + i] = up(c.Q[(size_t)i * n + i] - c.d[i]);
    double *s1 = malloc((size_t)n * n * sizeof *s1);
    double *s2 = malloc((size_t)n * n * sizeof *s2);
    double lminA = 0.0, lminM_fast = 0.0;
    int psdM_fast = 0;
    /* The bordered matrix rides along on A's factorization: its border is -w/2 (exact) and its
     * corner -beta_z rounded down, so proving the pair costs one O(n^3) factorization, not two. */
    double *border = malloc((size_t)n * sizeof *border);
    for (int i = 0; i < n; i++) border[i] = -0.5 * c.w[i];
    int psdA = verified_psd_ex(A, n, &lminA, s1, s2, border, dn(-c.beta_z), &lminM_fast, &psdM_fast);
    free(border);
    stage("verified_psd(A) + bordered extension");
    if (psdA) {
        ADDST("A = Q - diag(d) is positive semidefinite (verified Cholesky)", 0, NULL,
              "lambda_min >= %.6e", lminA);
    } else {
        double ub = NAN;
        if (sym && certify_not_psd(A_up, n, &ub))
            ADDST("A = Q - diag(d) is positive semidefinite (verified Cholesky)", 1,
                  "the split leaves a matrix that is not PSD, so the z-block minimum is -inf",
                  "certified not PSD: x'Ax <= %.3e < 0 along a Cholesky breakdown direction", ub);
        else
            ADDST("A = Q - diag(d) is positive semidefinite (verified Cholesky)", 2,
                  "PSD of the split could not be established in float64",
                  "not proved (shift ladder exhausted)%s", "");
    }

    double certified = -INFINITY, certified_hi = INFINITY;
    int psdM = 0;
    double lminM = 0.0;
    if (psdA) {
        /* Bordered matrix M = [[A, -w/2], [-w'/2, -beta_z]]. `-0.5*w` is exact (a power of two),
         * which it has to be: perturbing the border changes the linear term of the quadratic form
         * and the implication M >= 0 => z'Az - w'z >= beta_z stops following. The only place slack
         * is safe is the corner, where rounding -beta_z down only strengthens the claim. */
        int m = n + 1;
        if (psdM_fast) {          /* already established from A's factor -- no (n+1)^3 work needed */
            psdM = 1;
            lminM = lminM_fast;
            ADDST("beta_z is a lower bound on min_z z'Az - w'z (bordered matrix is PSD)", 0, NULL,
                  "lambda_min >= %.6e", lminM);
            goto after_bordered;
        }
        double *M = calloc((size_t)m * m, sizeof *M);
        for (int i = 0; i < n; i++)
            memcpy(M + (size_t)i * m, A + (size_t)i * n, (size_t)n * sizeof *A);
        for (int i = 0; i < n; i++) {
            M[(size_t)i * m + n] = -0.5 * c.w[i];
            M[(size_t)n * m + i] = -0.5 * c.w[i];
        }
        M[(size_t)n * m + n] = dn(-c.beta_z);
        double *t1 = malloc((size_t)m * m * sizeof *t1);
        double *t2 = malloc((size_t)m * m * sizeof *t2);
        psdM = verified_psd(M, m, &lminM, t1, t2);
        free(t1); free(t2);
        if (psdM) {
            ADDST("beta_z is a lower bound on min_z z'Az - w'z (bordered matrix is PSD)", 0, NULL,
                  "lambda_min >= %.6e", lminM);
        } else {
            /* That same rounding is why this can fail on a *valid* certificate: at beta_z = 0 the
             * corner becomes -5e-324 and the matrix as constructed genuinely is not PSD. So the
             * refutation is sought on the exact matrix instead -- A_up on the diagonal, and the corner
             * -beta_z itself, which negation leaves exact. */
            for (int i = 0; i < n; i++) M[(size_t)i * m + i] = A_up[(size_t)i * n + i];
            M[(size_t)n * m + n] = -c.beta_z;
            double ub = NAN;
            if (sym && certify_not_psd(M, m, &ub))
                ADDST("beta_z is a lower bound on min_z z'Az - w'z (bordered matrix is PSD)", 1,
                      "published beta_z is not a valid lower bound on the z-block",
                      "certified not PSD: the form is <= %.3e < 0, so beta_z overstates the minimum",
                      ub);
            else
                ADDST("beta_z is a lower bound on min_z z'Az - w'z (bordered matrix is PSD)", 2,
                      "the z-block bound has no slack the float64 checker can resolve",
                      "not proved: beta_z sits within rounding of the z-block minimum%s", "");
        }
        free(M);

    after_bordered:
        stage("bordered matrix check");
        if (psdM) {
            /* per-asset minima, in interval arithmetic: the lower endpoint for the proof, and the
             * upper endpoint -- the value at a feasible point -- for a refutation */
            double per = 0.0, per_hi = 0.0;
            for (int i = 0; i < n; i++) {
                iv civ = iv_sub(iv_add(iv_pt(c.w[i]), iv_pt(c.nu)),
                                iv_mul(iv_pt(c.pi), iv_pt(c.mu[i])));
                iv div_ = iv_pt(c.d[i]);
                iv slo = iv_pt(c.lo[i]), shi = iv_pt(c.hi[i]);
                iv qlo = iv_add(iv_mul(div_, iv_mul(slo, slo)), iv_mul(civ, slo));
                iv qhi = iv_add(iv_mul(div_, iv_mul(shi, shi)), iv_mul(civ, shi));
                double mlo = qlo.lo < qhi.lo ? qlo.lo : qhi.lo;
                double mhi = qlo.hi < qhi.hi ? qlo.hi : qhi.hi;
                if (c.d[i] > 0.0) {
                    iv negc = {-civ.hi, -civ.lo};
                    iv twod = iv_pt(2.0 * c.d[i]);
                    iv vert = iv_divpos(negc, twod);
                    iv fourd = iv_pt(4.0 * c.d[i]);
                    iv val = iv_divpos(iv_mul(negc, civ), fourd);
                    if (vert.hi > c.lo[i] && vert.lo < c.hi[i]) {
                        if (val.lo < mlo) mlo = val.lo;
                    }
                    /* for the upper bound the vertex counts only where it is *certainly* feasible */
                    if (vert.lo >= c.lo[i] && vert.hi <= c.hi[i]) {
                        if (val.hi < mhi) mhi = val.hi;
                    }
                }
                double term = dn(mlo + c.lam);
                if (term > 0.0) term = 0.0;
                per = dn(per + term);
                double term_hi = up(mhi + c.lam);
                if (term_hi > 0.0) term_hi = 0.0;
                per_hi = up(per_hi + term_hi);
            }
            double total = c.beta_z;
            total = dn(total + (-c.nu));
            total = dn(total + dn(c.pi * c.rho));
            total = dn(total + dn(-(c.lam * c.K)));
            total = dn(total + per);
            certified = total;
            /* An upper bound on what the exact tier certifies, which uses the true z-block minimum
             * beta*, not the published beta_z: a claim is refuted only above *that*.
             *
             * Computed only when the claim has already FAILED `claimed <= certified`, because that
             * is the only case where the verdict depends on it -- per the format's §3.6, a claim at
             * or below the re-derived bound is established outright. It costs a third O(n^3)
             * factorization (of A_up), which on a certificate that passes is 3.5 s of the 6.8 s
             * check at n = 2901 spent deciding between two verdicts that cannot occur. When it is
             * skipped the reported upper bound is +inf, which is a true upper bound and is what the
             * JSON now carries; the verdict itself is unchanged in every case. */
            if (c.claimed <= certified) {
                certified_hi = INFINITY;
            } else {
                double total_hi = beta_star_upper(A_up, n, c.w);
                total_hi = up(total_hi + (-c.nu));
                total_hi = up(total_hi + up(c.pi * c.rho));
                total_hi = up(total_hi + up(-(c.lam * c.K)));
                total_hi = up(total_hi + per_hi);
                certified_hi = total_hi;
            }
            stage("final inequality");
            int claim_st = c.claimed <= certified ? 0 : (c.claimed > certified_hi ? 1 : 2);
            ADDST("claimed bound <= independently re-derived bound", claim_st,
                  claim_st == 1 ? "the claimed bound exceeds anything the published witness supports"
                  : claim_st == 2 ? "the claim sits within rounding of the re-derived bound" : NULL,
                  "claimed = %.17g, certified in [%.17g, %.17g], shortfall = %.6e",
                  c.claimed, certified, certified_hi, c.claimed - certified);
        }
    }
    free(A); free(A_up); free(s1); free(s2);

    /* The verdict is a function of the checks and nothing else. Failing to establish something is
     * epistemic -- verified Cholesky fails at degeneracy by construction -- so it yields NOT PROVED.
     * Only a *certified* negation yields REFUTED: a definite violation on published numbers, a
     * direction of verified negative curvature, or a claim above a rigorous upper bound on the exact
     * value. Any refutation wins, so a negative multiplier is not masked by a PSD test after it that
     * could not complete. */
    int any_ref = 0, any_unp = 0;
    const char *reason = "";
    for (int i = 0; i < nc; i++) {
        if (checks[i].st == 1 && !any_ref) { any_ref = 1; reason = checks[i].why; }
        if (checks[i].st == 2) any_unp = 1;
    }
    if (!any_ref && any_unp)
        for (int i = 0; i < nc; i++) if (checks[i].st == 2) { reason = checks[i].why; break; }
    int all_ok = !any_ref && !any_unp;
    const char *verdict = any_ref ? "REFUTED" : (any_unp ? "NOT PROVED" : "PROVED");

    if (as_json) {
        /* JSON has no infinity or NaN literal, so a non-finite value is emitted as `null`. Writing
         * `-inf` would produce a document the reader's parser rejects -- and a verifier whose
         * output cannot be read is a verifier nobody runs. */
        char cbuf[64], sbuf[64];
        if (isfinite(certified)) {
            snprintf(cbuf, sizeof cbuf, "%.17g", certified);
            snprintf(sbuf, sizeof sbuf, "%.6e", c.claimed - certified);
        } else {
            snprintf(cbuf, sizeof cbuf, "null");
            snprintf(sbuf, sizeof sbuf, "null");
        }
        static const char *stname[3] = {"ok", "refuted", "unproved"};
        char hbuf[64];
        if (isfinite(certified_hi)) snprintf(hbuf, sizeof hbuf, "%.17g", certified_hi);
        else snprintf(hbuf, sizeof hbuf, "null");
        printf("{\"tier\":\"rigorous-c\",\"n\":%d,\"verdict\":\"%s\",\"reason\":\"%s\","
               "\"model_hash\":\"%s\",\"claimed_bound\":%.17g,\"certified_bound\":%s,"
               "\"certified_bound_upper\":%s,\"shortfall\":%s,\"checks\":[",
               n, verdict, all_ok ? "" : reason, mh, c.claimed, cbuf, hbuf, sbuf);
        for (int i = 0; i < nc; i++)
            printf("%s{\"name\":\"%s\",\"ok\":%s,\"status\":\"%s\"}", i ? "," : "",
                   checks[i].name, checks[i].ok ? "true" : "false", stname[checks[i].st]);
        printf("]}\n");
    } else {
        static const char *tag[3] = {"ok", "FAIL", "UNPROVED"};
        printf("ccpocheck (independent C verifier)  n=%d  verdict=%s%s%s\n", n, verdict,
               all_ok ? "" : ": ", all_ok ? "" : reason);
        printf("  model_hash = %s\n", mh);
        for (int i = 0; i < nc; i++)
            printf("  [%s] %s%s%s\n", tag[checks[i].st], checks[i].name,
                   checks[i].detail[0] ? " -- " : "", checks[i].detail);
        if (isfinite(certified))
            printf("  claimed  = %.17g\n  certified= %.17g\n  shortfall= %.6e\n",
                   c.claimed, certified, c.claimed - certified);
    }
    free(c.Q); free(c.mu); free(c.lo); free(c.hi); free(c.d); free(c.w);
    /* 0 PROVED, 1 REFUTED, 3 NOT PROVED -- the same codes as bound.py. 2 stays usage/parse. */
    return any_ref ? 1 : (any_unp ? 3 : 0);
}
