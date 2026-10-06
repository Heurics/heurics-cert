"""Certificates for other portfolio problems (``jax`` extra).

* ``heurics_cert.extras.cvar`` -- cardinality-constrained mean-CVaR: a support certifier over a pool of duals
  (``certify_cvar``) and the convex ceiling that rules out relaxation bounds (``cvar_convex_ceiling``);
* ``heurics_cert.extras.index_tracking`` -- lot-constrained index tracking, certified by enumerating supports
  (``certify``).

These modules import JAX at import time; import them explicitly.
"""
