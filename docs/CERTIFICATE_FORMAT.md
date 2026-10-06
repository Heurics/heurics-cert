# CCPO-CERT/1 — a checkable certificate for lower bounds on cardinality-constrained MIQP

**Status:** draft specification, version 1. Two independent implementations exist
(`heurics_cert/checkers/bound.py`, Python; `heurics_cert/checkers/c/`, C99) and agree bit-for-bit on every certificate
tested.

---

## 0. What this is, and what it is not

A CCPO-CERT document is a **proof of a lower bound**, not an attestation about a solver run. It
contains no claim about who produced it, what hardware ran, how long it took, or whether the search
converged. A verifier reads it, performs a fixed finite computation, and returns one of three
verdicts. Nothing about the prover has to be trusted — not its arithmetic, not its linear algebra,
not its floating-point mode, and not the machine it ran on.

That is the property the format exists to have. A number attested by a solver is worth exactly the
reader's trust in that solver; a number accompanied by a witness that any third party can check is
worth the reader's trust in *arithmetic*. Extending VIPR-style verified results from mixed-integer
*linear* to mixed-integer *quadratic* problems appears to be open, and this format is one route.

The distinction the format cannot collapse, and does not try to: a certificate proves a bound **for
the model it carries**. It cannot prove that model is the one the reader meant. §6 is how that is
handled.

---

## 1. The problem class

CCPO-CERT/1 certifies lower bounds for

```
minimize    x' Q x
subject to  mu' x  >=  rho                        (return floor)
            sum_i x_i  =  1                       (budget)
            lo_i y_i  <=  x_i  <=  hi_i y_i       (buy-in and cap, active only where held)
            sum_i y_i  <=  K                      (cardinality)
            y in {0,1}^n,  x in R^n
```

with `Q` symmetric and positive semidefinite, `lo <= hi` entrywise, `K` a positive integer.

This is the cardinality-constrained mean-variance problem in the form Frangioni & Gentile
distribute it, and the form the OR-Library portfolio datasets are naturally cast into. Nothing in
the format is specific to finance; `Q` is any PSD matrix and `mu` any linear functional.

---

## 2. The mathematics being certified

Introduce a copy `z = x` and dualize it with `w in R^n`; dualize the budget row with `nu in R`, the
return floor with `pi >= 0`, and the cardinality row with `lam >= 0`. Split the Hessian as

```
Q = A + diag(d),        A = Q - diag(d),        A positive semidefinite
```

and write `x_i = s_i y_i` so the per-asset terms separate. For **any** `(w, nu, pi, lam, d)` with
`pi >= 0`, `lam >= 0` and `A` PSD, the value

```
D  =  beta_z  -  nu  +  pi*rho  -  lam*K  +  sum_i min(0, m_i + lam)
```

is a valid lower bound on the optimum, where

```
beta_z  <=  min_{z in R^n}  ( z'Az - w'z )
m_i      =  min_{s in [lo_i, hi_i]}  ( d_i s^2 + c_i s ),      c_i = w_i + nu - pi*mu_i.
```

Two facts make this a *format* rather than an algorithm:

* **Weak duality holds at every multiplier.** `D` is a valid bound whether the multipliers came from
  a converged ascent, one gradient step, or a guess. The verifier therefore never has to reproduce,
  or even know about, the procedure that found them.
* **Every quantity in `D` is finitely checkable.** `beta_z` is bounded below by a semidefiniteness
  statement (§3.4), the `m_i` are minima of one-dimensional quadratics on intervals, and everything
  else is arithmetic on published scalars.

The sign of `d` is deliberately unconstrained. A negative `d_i` moves curvature out of the separable
part and into `A`, where the PSD requirement still accounts for it, and forbidding it would remove
the only safe back-off available on a singular `Q`.

---

## 3. The verification procedure

A conforming verifier performs exactly these checks. Each check ends in one of three states —
**established**, **refuted**, or **undecided** — and the verdict is a function of those states and
nothing else:

* **PROVED** (exit 0) — every check established.
* **REFUTED** (exit 1) — at least one check refuted. Any refutation wins, so a definite violation is
  never masked by a later check that could not be decided.
* **NOT PROVED** (exit 3) — nothing refuted, at least one check undecided.

A check may be **refuted** only by a *certified negation*, never by the failure of a proof:

* a definite comparison on published numbers that fails — `pi < 0`, an asymmetric `Q`, `lo > hi`;
  floats compare exactly, so there is nothing to round;
* a vector along which the quadratic form of the matrix in question is **provably negative** (§3.3,
  §3.4); or
* a claim above a rigorous **upper** bound on the value the exact tier would certify (§3.6).

Everything else that fails is **undecided**. This matters because verified arithmetic cannot prove
zero slack: a verified Cholesky needs the smallest eigenvalue to clear its own backward error, so a
perfectly valid certificate sitting exactly on a boundary is a certificate it *cannot* prove. An
exact-arithmetic verifier decides every check and is therefore two-valued; a floating-point verifier
is not, and must never contradict an exact one in either direction.

The three-way outcome is required, and the direction of the error is what makes it required. A
verifier that collapses NOT PROVED into REFUTED hands its user a reproducible artifact saying a
correct solver is wrong — strictly worse than shipping no verifier. One that collapses it into PROVED
is worthless. Earlier revisions of this document defined NOT PROVED only for §3.3 and called every
other failure a refutation; both reference implementations followed it, and both refuted a valid
certificate from a real OR-Library instance whose copy multiplier was exactly zero (§3.4).

### 3.1 Well-formedness
`Q` is exactly symmetric (`Q[i][j] == Q[j][i]` as stored, not to a tolerance). `lo_i <= hi_i` for
all `i`. Every `d_i` is finite. All arrays have length `n`, `Q` is `n x n`. Every number is a finite
real, and `K` is a positive integer (an integral value at most `2^53`; a JSON `10.0` is `10`).

A document that is not well formed proves nothing, and a verifier must say so rather than fail: the
Python tiers return REFUTED with the reason, and `ccpocheck` reports "malformed certificate" with
exit 2. In particular `K` must never be converted to a machine integer unchecked: an out-of-range
conversion is undefined behaviour in C, and once turned `K = 1e300` into a large negative integer,
which made `-lam*K` a huge positive term that was PROVED.

### 3.2 Multiplier signs
`pi >= 0` and `lam >= 0`. Weak duality fails silently without these, which is why they are checked
before anything expensive.

### 3.3 The split is positive semidefinite
`A = Q - diag(d)` must be PSD. **The prover assumes this and does not check it**; if it is false,
`min_z z'Az - w'z` is `-inf` and every number downstream is meaningless. This is the check that
failure analysis found to matter most in practice: a diagonal split computed from a floating-point
`lambda_min` can overshoot the cone by a few ulps, and the resulting certificate is
self-consistent, plausible, and not a proof.

A verifier establishes it by one of:

* **exact** — a rational `LDL'` factorization. Every float in the document is a dyadic rational, so
  this is a theorem about the published numbers with no tolerance anywhere.
* **rigorous** — a verified Cholesky. Factor `A - cI` in floating point; if it completes,
  Higham's backward-error result (*Accuracy and Stability of Numerical Algorithms*, Thm 10.3–10.5)
  gives `||E||_2 <= ||E||_F <= gamma_{n+1} || |R| ||_F^2` with `gamma_k = k u/(1 - k u)`, hence
  `lambda_min(A) >= c - delta`. Every quantity feeding that conclusion is rounded outward, so the
  conclusion is rigorous although the factorization is ordinary arithmetic.

The matrix actually tested must be entrywise **less than or equal to** `Q - diag(d)` — form the
diagonal with the subtraction rounded down — so that proving the tested matrix PSD proves the true
one PSD.

If the proof does not complete, the check is **undecided**, not refuted. To refute it, a verifier
must exhibit negative curvature of the *true* matrix: bound it from above instead (the same
diagonal rounded **up**), take a candidate direction `x` — the reference implementations use the
first non-positive pivot of a right-looking Cholesky, where `x = [-M11^{-1} m12; 1; 0 ...]` gives
`x'Mx` equal to that pivot — and show a rigorous upper bound on `x'Mx` is below zero. Any `x` is
admissible; only the bound has to be rigorous. The reference implementations bound the error of the
computed form a posteriori by `gamma_{2m+2}` times the absolute-value form (Higham, Lemma 3.1), which
holds whatever order the sums are evaluated in.

### 3.4 `beta_z` is a lower bound on the z-block
Rather than form a pseudo-inverse — which would make the verifier reproduce the prover's
eigendecomposition — establish that the **bordered matrix**

```
M  =  [  A        -w/2      ]
      [ -w'/2   -beta_z     ]
```

is positive semidefinite. Then for any `z`, `(z, 1)' M (z, 1) >= 0` is exactly
`z'Az - w'z - beta_z >= 0`, which is the required statement, for every `z`, from one
semidefiniteness fact.

The border entries `-w_i/2` must be formed **exactly** — halving is exact in binary floating point.
Perturbing them changes the linear term of the quadratic form and the implication no longer
follows. The corner `-beta_z` may be rounded **down**, which only strengthens what is proved.

That rounding is also why this check cannot be proved at zero slack, and why its failure must not
be read as a refutation. When `w = 0` and `beta_z = 0` — which a Lagrangian-ascent prover emits
whenever its best iterate is its starting point, since the ascent starts at zero multipliers: a short
budget does it, and so does a step target capped below the optimum —
the corner rounds to `-5e-324` and the matrix *as constructed* genuinely is not PSD, though the claim
it stands for is exactly true. A refutation must therefore be sought on the exact matrix: the `A`
diagonal rounded up, and the corner `-beta_z` itself, which negation leaves exact (§3.3).

**Provers** should not publish at zero slack. The reference prover takes a fixed amount off `beta_z`
— a few thousand times the verifier's own Cholesky backward-error bound `gamma_{n+2} trace(A)` — and
the same amount (times `1 + margin`) off the claim, so a certificate on this boundary is proved
rather than left undecided. The multiplicative `certify_margin` alone cannot do this: it scales
`beta_z`, and a multiple of zero is zero.

### 3.5 The per-asset minima
For each `i`, compute a lower bound on `m_i = min over [lo_i, hi_i] of d_i s^2 + c_i s`, with
`c_i = w_i + nu - pi*mu_i`. The minimum of a quadratic on an interval is at an endpoint or, when
`d_i > 0`, at the clipped vertex. A rigorous verifier evaluates both endpoints and, where the vertex
interval can overlap the box, the vertex value `-c_i^2/(4 d_i)`, all in interval arithmetic, and
keeps the lower endpoint.

Taking `-c_i^2/(4 d_i)` unconditionally is *valid* — it is the unconstrained minimum, which is below
the boxed one — but on assets whose optimum sits at a bound it is loose by orders of magnitude, and
a bound that is valid and meaningless is not worth publishing.

### 3.6 The final inequality
Accumulate `D` with every operation rounded **downward**, and require

```
claimed_bound  <=  D.
```

If it holds, the check is established. If it does not, the check is **refuted only if** the claim
also exceeds a rigorous *upper* bound `D_hi` on the value the exact tier certifies; otherwise it is
**undecided**. `D_hi` is accumulated with every operation rounded **upward**, from two changes:

* the per-asset minima use the value at a *feasible* point — both endpoints, and the vertex only where
  its interval lies certainly inside the box — which bounds the boxed minimum from above; and
* the z-block term is not `beta_z` but an upper bound on the true minimum `beta* = min_z z'Az - w'z`,
  because the exact tier certifies with `beta*`. Evaluating the form at any `z` bounds `beta*` from
  above; `z = A^{-1}w/2` makes it tight, and `z = 0` shows `beta* <= 0`. Using `beta_z` here instead
  would refute a certificate that published a slack `beta_z` while the exact tier proved it.

The *shortfall* `claimed_bound - D` is reported whatever the verdict. On an exact verifier a positive
shortfall is a bound that is not a bound. On a floating-point verifier it is not, by itself: a claim
exactly at the true value has a positive shortfall against any rigorous underestimate of it.

---

## 4. Document format

A UTF-8 JSON object. Numbers are IEEE-754 binary64 written with enough digits to round-trip
(`repr` in Python, `%.17g` in C).

| field | type | meaning |
|---|---|---|
| `Q` | array of `n` arrays of `n` numbers | the Hessian, symmetric PSD, **full** matrix not a triangle |
| `mu` | array of `n` numbers | the return vector |
| `lo`, `hi` | arrays of `n` numbers | buy-in floor and cap, active only where held |
| `K` | integer | cardinality bound |
| `rho` | number | return floor |
| `d` | array of `n` numbers | the diagonal split; any sign |
| `w` | array of `n` numbers | the copy multiplier |
| `nu` | number | budget-row multiplier, unrestricted sign |
| `pi` | number | return-row multiplier, `>= 0` |
| `lam` | number | cardinality-row multiplier, `>= 0` |
| `beta_z` | number | claimed lower bound on the z-block minimum |
| `claimed_bound` | number | the number being certified |
| `model_hash` | string | SHA-256 hex over the model fields, §6 |

Optional and **never** part of the proof: `name`, `dtype`, `reported_bound`, `spectral_floor`,
`floor_wins`, `bound_seconds`, `certify_margin`. A verifier must ignore fields it does not know.

`reported_bound` deserves a note because its absence from the proof is the point. A prover may
report a number *below* `claimed_bound` — for a safety margin, or because a different construction
won on that instance. The certificate proves `claimed_bound`; anything the prover chose to report
below it is thereby also proved, and needs no separate treatment.

---

## 5. Size

The document splits into a **witness**, `O(n)`, and a **model**, `O(n^2)`:

* witness: `d`, `w`, `nu`, `pi`, `lam`, `beta_z`, `claimed_bound`, `K`, `rho`, `model_hash`
* model: `Q`, `mu`, `lo`, `hi`, `K`, `rho`

A verifier that already holds the model — normally the case, since the verifier is usually the party
that posed the problem — needs only the witness plus the fingerprint. A verifier that holds nothing
needs the whole document, which is dominated by `Q`. Implementations should report both sizes.
Quoting only the self-contained size overstates the cost by a factor of `n`; quoting only the
witness understates the case where the model must travel too.

---

## 6. `model_hash`, and the attack it exists to stop

Every verdict is **model-relative**. A certificate about a *more constrained* problem — narrower
`hi`, higher `rho`, lower `K` — is perfectly self-consistent, admits a genuinely higher bound, and
answers a question nobody asked. No self-contained document can detect this, because there is
nothing wrong with it. It is the same failure mode as a big-M value chosen too small: bounding the
wrong problem, silently.

The defence is to bind the verdict to a model the reader holds independently.

```
h = SHA-256()
for field in ("Q", "mu", "lo", "hi", "K", "rho"):
    h.update(field name as ASCII bytes)
    if field is an array:
        h.update(int64 big-endian rows, int64 big-endian cols)      # a vector has rows = 1
        h.update(float64 big-endian bytes of every entry, row-major)
    else:
        h.update(float64 big-endian bytes of the value)             # K is hashed as a float64
```

Hashed over IEEE bytes, never over decimal text, which round-trips differently across writers.
Verifiers must offer an `--expect-model <prefix>` mode that refuses a certificate whose fingerprint
does not match the one the reader supplies.

---

## 7. Conformance

An implementation conforms if, on the reference suite:

1. it returns PROVED on every valid certificate with slack and reproduces `model_hash` exactly;
2. it **never returns REFUTED on a valid certificate**, including the zero-slack boundary certificate
   (`w = 0`, `beta_z = 0`), on which an exact verifier returns PROVED and a floating-point one returns
   PROVED or NOT PROVED;
3. it **never returns PROVED** on any of the eleven adversarial mutations
   (`tools/adversarial.py`): the claim raised by one ulp above the certifiable value, by
   `1e-9`, and by 1%; `pi` or `lam` negative; `pi` or `lam` moved without being paid for in the
   objective; the split pushed past `lambda_min`; `beta_z` overstated; `w` perturbed; and the
   witness taken from the wrong ascent iterate;
4. it **returns REFUTED** on each of those mutations, except that a floating-point verifier may
   return NOT PROVED on the one-ulp mutation, which no float64 enclosure can resolve. An exact
   verifier must refute all eleven. Implementations report the **detection rate** (REFUTED) and the
   **inconclusive rate** (NOT PROVED) separately — counting every non-PROVED verdict as a detection
   scores a verifier that is undecided on everything as perfect;
5. it computes a *different* fingerprint for each of the three model substitutions (box narrowed,
   return floor raised, cardinality reduced), each of which it must otherwise accept.

Both reference implementations currently detect 10 of the 11 and leave only the one-ulp mutation
undecided; they agree with each other on every verdict, forgeries included.

The eleventh mutation is not invented. It reproduces a defect that shipped: published multipliers
taken from the last ascent iterate while the reported value came from the best one, so nothing a
reader was handed reproduced the number beside it.

### Reference implementations

| | language | dependencies | rigorous tier | exact tier |
|---|---|---|---|---|
| `checkers/bound.py` | Python 3 | numpy (rigorous tier only; the exact tier is stdlib) | yes | yes |
| `ccpocheck` | C99 | libc, libm | yes | no |

They share no code. Different JSON reader, different SHA-256, different Cholesky, different interval
arithmetic, different compiler. On the certificate corpus they agree on every verdict and on the
certified bound to within 2.7e-16 relative, and they produce identical fingerprints.

Neither substitutes for review by someone who did not write the prover. Both were written by the
same author, so a misreading of §2 would be reproduced in both, and that limitation is stated here
rather than left for a reader to notice.

---

## 8. Cost

Measured on one core of an Intel i9-14900K (`tools/cross_check.py` reproduces the agreement report,
`tools/results/BLOCK_D_CROSSCHECK.md`):

| `n` | rigorous tier | exact tier |
|---|---|---|
| 20 | 0.3 ms | 0.011 s |
| 40 | 0.5 ms | 0.17 s |
| 100 | 1.5 ms | 7.2 s |
| 200 | 2.5 ms | 171 s |
| 400 | 17 ms | — |

Verification is roughly four orders of magnitude cheaper than the search that produced the answer.
The exact tier — nothing taken on trust, not even IEEE arithmetic — reaches `n = 200` in about three
minutes, which covers a concentrated mandate outright; above that the rigorous tier is the answer
and stays in single-digit milliseconds. The witness's precision does not matter: a float32
certificate costs the same to check exactly as a float64 one.

---

## 9. What a PROVED verdict does not say

* It says nothing about the **primal**. A separate feasible point is what turns a bound into a gap.
* It says nothing about **tightness**. A trivially loose bound is perfectly provable, and a
  certificate is not a claim that the number is useful.
* It says nothing about **the prover**. Determinism, hardware and runtime are outside the format
  deliberately — the verifier does not care which machine produced the witness, only that the
  witness proves the number beside it. This is a weaker promise than cross-hardware bit-identity and
  a considerably easier one to keep: bit-identity commits a vendor to every future driver, compiler
  and dispatch path behaving identically forever.
* It is **model-relative**. See §6.
