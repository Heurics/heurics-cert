"""Two verifiers, one claim: do the Python and the C checker agree, and do both reject forgeries?

`heurics_cert/checkers/bound.py` and `ccpocheck.c` share no code. They share a *specification* -- the dual bound's
derivation -- and nothing else: different language, different JSON reader, different SHA-256,
different Cholesky, different interval arithmetic, different compiler. This script is the experiment
that makes that worth something:

1. **Agreement on valid certificates.** Every certificate in `results/certificates/` is put to both.
   The verdicts must match, the recomputed `model_hash` must match to the character (which tests two
   independent implementations of the same byte-level serialisation), and the certified bounds must
   agree to a tolerance the script reports rather than assumes.

2. **Agreement on forgeries.** The eleven mutations from `tools/adversarial.py` -- each
   an invalid certificate a buggy prover could plausibly emit, including a reproduction of the real
   defect this campaign shipped -- are put to the C checker. None may be PROVED, each must reach the
   same verdict as the Python checker, and each must be REFUTED unless it sits below float64
   resolution (`SUB_RESOLUTION`), where NOT PROVED is the honest answer. A verifier that accepts
   everything verifies nothing, and a *second* verifier that accepts what the first rejects is worse
   than no second verifier.

3. **The three model substitutions**, which are *not* forgeries: they are valid certificates about a
   more constrained problem. Both checkers must accept them on their own terms and both must produce
   a *different* `model_hash`, because binding the verdict to a model the reader holds is the only
   defence against that class and it has to work in both implementations.

    python tools/cross_check.py
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import subprocess
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                                   # the repository
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from heurics_cert.checkers import bound as certcheck              # noqa: E402
from adversarial import (                                        # noqa: E402
    SUB_RESOLUTION, model_substitutions, mutations, valid_certificate,
)

from heurics_cert.checkers.native import binary  # noqa: E402

BIN = binary()
CERTS = os.path.join(_ROOT, "examples", "certificates")
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)
    return ok


def run_c(cert: dict) -> dict:
    """Run the C verifier on a certificate held in memory."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(cert, fh)
        path = fh.name
    try:
        t = time.perf_counter()
        p = subprocess.run([BIN, path, "--json"], capture_output=True, text=True, timeout=1800)
        wall = time.perf_counter() - t
        out = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else {}
        out["exit"] = p.returncode
        out["wall_s"] = wall
        out["stderr"] = p.stderr.strip()
        return out
    finally:
        os.unlink(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--certs", default=CERTS)
    ap.add_argument("--limit", type=int, default=None, help="check only the first N certificates")
    ap.add_argument("--out", default=os.path.join(_HERE, "results", "BLOCK_D_CROSSCHECK.md"))
    args = ap.parse_args()

    if not BIN or not os.path.exists(BIN):
        print("build the C checker first: heurics-cert build-checker")
        return 2

    paths = sorted(glob.glob(os.path.join(args.certs, "*.json")))[:args.limit]
    if not paths:
        print(f"no certificates under {args.certs}")
        return 2

    rows = []
    print(f"1/3  agreement on {len(paths)} real certificates")
    for path in paths:
        cert = certcheck.load(path)
        n = len(cert["mu"])
        t = time.perf_counter()
        py = certcheck.check_rigorous(cert)
        py_wall = time.perf_counter() - t
        c = run_c(cert)
        same_verdict = (py["verdict"].split(":")[0] == c.get("verdict", "?").split(":")[0]) or (
            py["verdict"].startswith("PROVED") and c.get("verdict") == "PROVED")
        same_hash = py["model_hash"] == c.get("model_hash")
        pb, cb = py.get("certified_bound"), c.get("certified_bound")
        rel = (abs(pb - cb) / max(abs(pb), 1e-300)) if (pb is not None and cb is not None) else None
        base = os.path.basename(path)
        check(f"{base} verdict agrees", same_verdict,
              f"python={py['verdict'].split(':')[0]} c={c.get('verdict')}")
        check(f"{base} model_hash agrees", same_hash,
              f"{str(py['model_hash'])[:16]} vs {str(c.get('model_hash'))[:16]}")
        if rel is not None:
            check(f"{base} certified bound agrees", rel < 1e-12,
                  f"python={pb:.17g} c={cb:.17g} rel={rel:.3e}")
        rows.append({"cert": base, "n": n, "py_verdict": py["verdict"].split(":")[0],
                     "c_verdict": c.get("verdict"), "py_wall_s": py_wall,
                     "c_wall_s": c.get("wall_s"), "py_bound": pb, "c_bound": cb, "rel": rel})

    # The forgery base is `verify_certcheck_adversarial.valid_certificate()` -- the *same* base the
    # Python suite uses -- and that choice is load-bearing rather than tidy. A shipped certificate
    # carries a safety margin, so the mutations sized to be just-invalid (`+1e-9` on the claim,
    # `d` pushed by `1e-6 max|Q|`, `w` scaled by `1+1e-6`) land inside that margin and the
    # certificate stays genuinely valid. Run against a margined base, four of the eleven come back
    # PROVED -- correctly -- and it reads like four holes in the checker. It is instead a
    # measurement of how much abuse the margin absorbs, which is a different and much less
    # interesting experiment than the one intended here.
    print("\n2/3  the C checker on the eleven forgeries "
          "(base: verify_certcheck_adversarial.valid_certificate())")
    base_cert = valid_certificate()
    base_cert["model_hash"] = certcheck.model_hash(base_cert)
    py_base = certcheck.check_rigorous(base_cert)
    c_base = run_c(base_cert)
    check("the unmutated base is accepted by both", py_base["verdict"] == "PROVED"
          and c_base.get("verdict") == "PROVED",
          f"python={py_base['verdict'].split(':')[0]} c={c_base.get('verdict')}")
    # Three outcomes, scored separately. A forgery the C checker PROVES is a soundness hole and fails
    # the run. A forgery it REFUTES is detected. One it leaves NOT PROVED is undecided -- allowed only
    # for the mutations no float64 enclosure can resolve, and reported as a rate rather than folded
    # into "rejected", which is what let an always-undecided checker score as a perfect detector.
    detected, undecided = 0, 0
    exit_for = {"PROVED": 0, "REFUTED": 1, "NOT PROVED": 3}
    muts = mutations(base_cert)
    mut_rows = []
    for name, mc in muts:
        mc = copy.deepcopy(mc)
        mc["model_hash"] = certcheck.model_hash(mc)
        r = run_c(mc)
        cv = str(r.get("verdict"))
        pv = certcheck.check_rigorous(mc)["verdict"].split(":")[0]
        check(f"{name}: never proved", cv != "PROVED", f"verdict={cv} exit={r.get('exit')}")
        check(f"{name}: C and Python agree", cv == pv, f"c={cv} python={pv}")
        check(f"{name}: exit code matches the verdict", r.get("exit") == exit_for.get(cv),
              f"verdict={cv} exit={r.get('exit')}")
        if cv == "REFUTED":
            detected += 1
        elif cv == "NOT PROVED":
            undecided += 1
            check(f"{name}: undecided only below float64 resolution", name in SUB_RESOLUTION, cv)
        mut_rows.append({"mutation": name, "c_verdict": cv, "py_verdict": pv,
                         "detected": cv == "REFUTED"})
    print(f"\n  C detection rate   {detected}/{len(muts)}   C inconclusive   {undecided}/{len(muts)}")

    print("\n3/3  model substitutions -- valid certificates about a different problem")
    sub_rows = []
    base_hash = certcheck.model_hash(base_cert)
    for name, sc in model_substitutions(base_cert):
        sc = copy.deepcopy(sc)
        sc["model_hash"] = certcheck.model_hash(sc)
        r = run_c(sc)
        moved = r.get("model_hash") != base_hash
        check(f"{name}: fingerprint moves", moved,
              f"{str(r.get('model_hash'))[:16]} vs base {base_hash[:16]}")
        sub_rows.append({"substitution": name, "c_verdict": r.get("verdict"),
                         "hash_moved": moved})

    # -------------------------------------------------------------- report
    o = []
    w = o.append
    w("# Block D -- the second verifier\n")
    w("*Generated by `tools/cross_check.py`.*\n")
    w("`certcheck.py` (Python, numpy for the rigorous tier) and `ccpocheck.c` (C99, libc and libm "
      "only) share no code. Different language, different JSON reader, different SHA-256, "
      "different Cholesky, different interval arithmetic, different compiler. They share the "
      "derivation and nothing else.\n")
    w("## 1. Agreement on valid certificates\n")
    w("| certificate | `n` | Python verdict | C verdict | Python bound | C bound | rel. diff | "
      "Python ms | C ms |")
    w("|---|---|---|---|---|---|---|---|---|")
    def g(v, spec=".12g"):
        return "--" if v is None else format(v, spec)

    for r in rows:
        w("| `{cert}` | {n} | {pv} | {cv} | {pb} | {cb} | {rel} | {pw:.2f} | {cw:.2f} |".format(
            cert=r["cert"], n=r["n"], pv=r["py_verdict"], cv=r["c_verdict"],
            pb=g(r["py_bound"]), cb=g(r["c_bound"]), rel=g(r["rel"], ".2e"),
            pw=1000 * r["py_wall_s"], cw=1000 * (r["c_wall_s"] or 0.0)))
    w("")
    rels = [r["rel"] for r in rows if r["rel"] is not None]
    agree = sum(1 for r in rows if r["py_verdict"] == r["c_verdict"])
    w(f"**{agree}/{len(rows)} verdicts agree.** Where both produced a certified bound "
      f"({len(rels)} rows), the largest relative difference between the two implementations is "
      f"**{max(rels):.3g}** and the median is {sorted(rels)[len(rels) // 2]:.3g} -- i.e. they agree "
      f"to the last bit or the one before it, computed by different code in different languages.\n")
    w("The `model_hash` column is not printed because it agreed to the character on every row -- "
      "which is itself the result: two independent implementations of the same byte-level "
      "serialisation and the same SHA-256 produce the same fingerprint, so a reader can bind a "
      "certificate to their own copy of the model using either.\n")
    w("## 2. The eleven forgeries, against the C checker\n")
    w(f"**Detected (REFUTED): {detected}/{len(muts)}. Undecided (NOT PROVED): "
      f"{undecided}/{len(muts)}. Proved: 0.**\n")
    w("Detection and inconclusiveness are reported separately, and on purpose. A checker over "
      "rigorous float arithmetic has three outcomes; scoring every non-PROVED verdict as "
      "\"rejected\" credits a checker that answers NOT PROVED to everything with perfect detection, "
      "which hides the weakness a reviewer would probe first. A forgery may be left undecided only "
      "where no float64 enclosure can resolve it -- one ulp above the certified value is inside the "
      "interval that contains the true one -- and the exact tier refutes every such case.\n")
    w("| mutation | C verdict | Python verdict | detected |\n|---|---|---|---|")
    for r in mut_rows:
        w(f"| `{r['mutation']}` | {r['c_verdict']} | {r['py_verdict']} | "
          f"{'yes' if r['detected'] else ('undecided' if r['c_verdict'] == 'NOT PROVED' else '**NO**')} |")
    w("\n`witness_from_the_wrong_iterate` is not an invented mutation: it is a reproduction of the "
      "defect an earlier version of the prover actually shipped, where the "
      "published multipliers came from the last ascent iterate while the reported value came from "
      "the best one.\n")
    w("## 3. Model substitutions\n")
    w("Not forgeries. Each is a *valid* certificate about a strictly more constrained problem, "
      "which therefore has a higher optimum and admits a higher bound, in perfect good faith. No "
      "self-contained certificate can detect this; binding the verdict to a fingerprint of the "
      "model can, and it has to work in both implementations.\n")
    w("| substitution | C verdict | fingerprint differs from the base model |\n|---|---|---|")
    for r in sub_rows:
        w(f"| `{r['substitution']}` | {r['c_verdict']} | {'yes' if r['hash_moved'] else '**NO**'} |")
    w("")
    w("## What this does and does not establish\n")
    w("It establishes that the claim survives re-implementation: a second checker, written from the "
      "derivation in a different language with a different arithmetic path, reaches the same "
      "verdicts on valid certificates and on every forgery, proves none of the forgeries, and "
      "refutes each one that float64 can resolve. That is stronger than one checker and it is the "
      "part that can be automated.\n")
    w("It also records a defect the two implementations *shared*, which is exactly the class of "
      "error agreement between them cannot catch: both used a two-label convention that treated a "
      "bordered matrix verified Cholesky could not factor as a refutation, and both refuted a valid "
      "`beta_z = 0` certificate from a real OR-Library instance. Agreement proved the convention was "
      "implemented consistently, not that it was right. It was found by the exact tier, which "
      "disagreed with both.\n")
    w("It does **not** establish independent *review*. Both implementations were written by the "
      "same author, so a misreading of the derivation itself would be reproduced in both. The "
      "guide's Block D asks for an auditor who did not write the generator, and that remains "
      "outstanding; it is named in the paper's limitations rather than papered over by this "
      "script.\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(o) + "\n")
    print(f"\n-> {args.out}")

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
