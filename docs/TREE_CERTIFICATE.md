# CCPO-TREE/0: a checkable proof that a portfolio is optimal

A CCPO-CERT/1 certificate proves a lower bound. A tree certificate proves that a given portfolio is optimal (to a
stated relative tolerance) for the cardinality-constrained QP

    min x'Qx  s.t.  1'x = 1,  mu'x >= rho,  lo_i y_i <= x_i <= hi_i y_i,  1'y <= K,  y binary,

or, when the search did not finish, a certified lower bound. It is produced by `heurics_cert.search.BranchAndBound` and checked by
`heurics_cert.checkers.tree`, which shares no code with the search and trusts nothing it computed.

## What the reader trusts

* the model they hold (its CCPO-CERT/1 SHA-256 fingerprint must equal the record's `model_sha256`);
* the **root certificate**, a CCPO-CERT/1 tangent witness whose split `d` the record repeats as `root_d`. Proving it
  with the existing checkers establishes that `A = Q - diag(d)` is positive semidefinite, the one matrix-level fact
  every node bound relies on.

Everything else in the record is re-derived.

## The record (JSON; gzip in practice)

| field | meaning |
|---|---|
| `format` | `"CCPO-TREE/0"` |
| `model_sha256` | fingerprint of `(Q, mu, lo, hi, K, rho)`, as CCPO-CERT/1 |
| `root_d` | the root split `d` (must equal the root witness's) |
| `nodes` | list of `[parent, asset, side]`; node 0 is the root `[-1, -1, -1]`; `side` 1 = asset held, 0 = excluded |
| `split` | `{node: asset}` for every internal node |
| `term` | `{node: proof}` for every terminal node |
| `tangents` | list of sparse points `(indices, values)`: distinct integer indices in `[0, n)`, one finite value each; entry 0 is the root witness's tangent point |
| `incumbent` | sparse portfolio `(idx, val)` |
| `U`, `rel_tol` | the incumbent's objective as the search saw it (not trusted) and the search's stopping tolerance (informational; the checker applies the reader's) |

A node's held set IN and excluded set OUT are not stored: the checker rebuilds them from the parent chain.

### Proofs

* `{"kind": "bound", "at": a, "t": k, "nu", "pi", "lam", "lo0"}`: node `a` must be the terminal itself or an
  ancestor (a descendant's portfolios are a subset of an ancestor's). With `x_hat = tangents[k]`,
  `w = 2 A x_hat`, `beta = -x_hat'A x_hat`, and IN/OUT/FREE those of node `a`,

      bound = beta - nu + pi rho - lam (K - |IN|) + sum_{i in IN} m_i + sum_{i in FREE} min(m_i + lam, 0),
      m_i   = min_{s in [lo_i, hi_i]} d_i s^2 + (w_i + nu - pi mu_i) s      (lo_i replaced by 0 if `lo0`)

  is a lower bound on every portfolio of the terminal, for any `nu`, `pi >= 0`, `lam >= 0`. It is the Lagrangian of
  the model with the binaries kept binary, after the tangent inequality `x'Ax >= w'x + beta` (valid because `A` is
  PSD). `lo0` relaxes the buy-in bounds, which only lowers the bound.
* `{"kind": "card"}`: the terminal holds more than `K` assets.
* `{"kind": "support"}`: every asset is decided (IN u OUT = all) and the held support cannot meet the budget or the
  return row (checked in exact rational arithmetic).

The `val` stored with a bound proof is the search's own float value and is ignored by the checker.

## What the checker does

1. **Binding.** Format, model fingerprint, and `root_d` equal to the root witness's split; the root certificate must
   be PROVED (`check_tree` runs the rigorous checker on it).
2. **Structure.** The record has at least the root node; every non-root node's parent precedes it and splits on the
   node's asset, which is one of the model's `n` assets and not already decided on the path; every internal node has
   exactly two children, sides 1 and 0; every node is internal or terminal, never both, and every terminal proof
   names a node of the record. The terminals then partition all supports, so the minimum over terminals is a lower
   bound on the optimum. A bound proof's `at` must be a node of the record (not the root's parent `-1`) and an
   ancestor-or-self of its terminal. A record that cannot be read at all is REFUTED, never a crash.
3. **Bounds.** Each bound proof is re-evaluated as a rigorous lower bound: the tangent point must be a point
   (distinct in-range indices -- with a repeated index `w` and `beta` would come from different vectors and the
   tangent inequality would fail), then `w` and `beta` are recomputed from it with a-priori floating-point error radii (`gamma_k = k u / (1 - k u)`), each `m_i` is bounded below
   (the unconstrained minimum `-c^2/(4d)` wherever the vertex's position is ambiguous), and the final sum carries its
   own rounding radius. `card` and `support` proofs are checked combinatorially / exactly.
4. **Incumbent.** Distinct in-range indices with finite weights; budget, return row, box and cardinality within
   scale-relative tolerances; its objective bounded above rigorously.

## Verdicts

* **OPTIMAL** -- `(U - LB) / U <= rel_tol`: the incumbent is optimal to the stated tolerance. `rel_tol` is the
  reader's (`check_tree(..., rel_tol=1e-6)` by default) and is reported with the verdict; the record's own `rel_tol`
  is the search's stopping rule and is ignored, since an untrusted field must not set the acceptance rule.
* **BOUND** -- the tree is well formed and `LB` is certified, but it does not reach the incumbent (an unfinished
  search, or no valid incumbent).
* **REFUTED** -- the record is malformed or bound to another model or split.
* **NOT PROVED** -- the root certificate could not be proved.

## Cost

Checking is dominated by one matrix-vector product per distinct tangent point; each terminal then costs O(|IN| +
|FREE|). On the 189-cell factor tier (n = 1,437-2,901): median 0.3 s, max 14 s; trees median 108 KB.
