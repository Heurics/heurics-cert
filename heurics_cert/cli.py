"""The ``heurics-cert`` command.

    heurics-cert prove PROBLEM.npz -o cert.json [--qbin model.qbin] [--solver fista|clarabel] [--no-check]
    heurics-cert verify cert.json [--qbin model.qbin] [--method rigorous|exact|c] [--expect-model PREFIX]
    heurics-cert check-tree tree.json.gz --certificate cert.json [--qbin model.qbin] [--rel-tol 1e-6]
    heurics-cert build-checker [--dest DIR] [--compiler CC]
    heurics-cert info

Exit codes: 0 proved (or optimal), 1 refuted, 2 model mismatch, 3 not proved or unavailable.
"""
from __future__ import annotations

import argparse
import json
import sys

from heurics_cert.status import Status, exit_code


def _prove(a) -> int:
    from heurics_cert.models.problem import Problem
    from heurics_cert.params import Params
    from heurics_cert.prover.pipeline import prove
    problem = Problem.from_npz(a.problem)
    params = Params(solver=a.solver, ascent_steps=a.ascent_steps, check=() if a.no_check else ("rigorous", "c"))
    res = prove(problem, params=params, qbin=a.qbin)
    res.certificate.save(a.out, qbin=a.qbin)
    print(json.dumps({"status": str(res.status), "bound": res.bound, "split": res.split,
                      "fingerprint": problem.fingerprint, "spread": res.spread,
                      "verdicts": {k: str(v.status) for k, v in res.verdicts.items()},
                      "timings_s": {k: round(t, 3) for k, t in res.timings.items()}}, indent=1))
    return 0 if (res.proved or a.no_check) else exit_code(res.status)


def _verify(a) -> int:
    from heurics_cert.checkers.api import verify
    v = verify(a.cert, a.method, qbin=a.qbin, expect_model=a.expect_model)
    print(json.dumps({"status": str(v.status), "method": v.method, "certified_bound": v.certified_bound,
                      "fingerprint": v.model_hash, "reason": v.reason}, indent=1))
    return exit_code(v.status)


def _check_tree(a) -> int:
    from heurics_cert.certificates.certificate import load
    from heurics_cert.certificates.tree import TreeCertificate
    cert = load(a.certificate, qbin=a.qbin)
    r = TreeCertificate.load(a.tree).check(cert.problem, cert.witness, method=a.method, rel_tol=a.rel_tol)
    print(json.dumps({k: r.get(k) for k in ("verdict", "LB", "U", "gap_pct", "terminals", "reason")}, indent=1))
    return exit_code(Status.parse(r["verdict"]))


def _build_checker(a) -> int:
    from heurics_cert.checkers.build import build
    try:
        path = build(a.dest, a.compiler)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 3
    print(json.dumps({"built": path, "self_test": "passed"}, indent=1))
    return 0


def _info(_a) -> int:
    from heurics_cert import __version__
    from heurics_cert.checkers.native import binary
    print(json.dumps({"version": __version__, "c_checker": binary()}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="heurics-cert", description="Checkable optimality certificates.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prove", help="certify a lower bound for a problem stored as .npz")
    p.add_argument("problem")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--qbin", help="write the split layout: witness JSON plus this binary model file")
    p.add_argument("--solver", default="fista", choices=("fista", "clarabel"))
    p.add_argument("--ascent-steps", type=int, default=8)
    p.add_argument("--no-check", action="store_true")
    p.set_defaults(fn=_prove)
    v = sub.add_parser("verify", help="check a bound certificate (CCPO-CERT/1 or /2)")
    v.add_argument("cert")
    v.add_argument("--qbin")
    v.add_argument("--method", default="rigorous", choices=("rigorous", "exact", "c"))
    v.add_argument("--expect-model")
    v.set_defaults(fn=_verify)
    t = sub.add_parser("check-tree", help="check a search-tree certificate (CCPO-TREE/0)")
    t.add_argument("tree")
    t.add_argument("--certificate", required=True, help="the root certificate the tree builds on")
    t.add_argument("--qbin")
    t.add_argument("--method", default="rigorous", choices=("rigorous", "exact", "c"))
    t.add_argument("--rel-tol", type=float, default=1e-6)
    t.set_defaults(fn=_check_tree)
    b = sub.add_parser("build-checker", help="compile the C checker from its bundled source")
    b.add_argument("--dest", help="directory for the binary (default ~/.heurics-cert/bin)")
    b.add_argument("--compiler", help="C compiler (default $CC, cc, gcc, clang, or MSVC on Windows)")
    b.set_defaults(fn=_build_checker)
    i = sub.add_parser("info", help="version and checker availability")
    i.set_defaults(fn=_info)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
