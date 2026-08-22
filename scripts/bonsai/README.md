# Bonsai Hadamard experiments (2026-08-21)

Rotation + recovery pipeline applied to PrismML Ternary-Bonsai-1.7B.
All measurements, gates, and the final NO-SHIP verdict are in
`docs/RUN_NOTES.md`; the experiment plan in `docs/ROADMAP-BONSAI.md`.

- `hstack.py` — Hadamard machinery (Sylvester H2048, Paley-12 kron
  H512, nonzero-mean ternary STE, HLinear, fold-back export).
  `hstack.self_test()` verifies every structural claim numerically.
- `c0c1_gate.py` — identity + admission-price gates.
- `h_cpt.py` — CPT arms (rotated vs control; `--data wikipedia`).
- `h_kd.py` — multi-level KD (T=1 logits + last-2-layer attention).
- `export_folded.py` — fold-back to stock-HF-loadable fp16.
- `bonsai_tune.py` — earlier PV scale-tune transfer test (negative).

Provenance: clean single-driver re-runs after a two-session
collision; audit trail in RUN_NOTES (2026-08-21 entries).
