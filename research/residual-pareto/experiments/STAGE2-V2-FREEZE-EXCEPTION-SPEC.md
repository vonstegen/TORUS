# Stage 2 v2 — Driver Extension (Freeze Exception Spec)

**Date:** 2026-08-24
**Driver SHA at baseline:** `e7b2442` (Stage 2 v1 frozen)
**Authority:** OPERATING-PLAN §6 (feature freeze) and §7 (driver changes require a documented exception).

## Why this exception is needed

Stage 2 v1 showed the TWN damage recipe is degenerate on layers 8 and 15 down_proj. To satisfy RPM-006's "≥2 layer categories" PASS+ rule, **Stage 2 v2 needs at least one non-down_proj site** (e.g., attention `q_proj`/`v_proj`).

The current driver has two hardcoded assumptions that block non-down_proj sites:

1. **`build_site_adapter(arm_id, ..., hidden_size, intermediate_size)`** hardcodes
   `in_features=intermediate_size, out_features=hidden_size`. For attention
   `q_proj`, the geometry is `(hidden_size, hidden_size)` — the input/output
   dim is `hidden_size`, not `intermediate_size`.

2. **`damage_target_module`** implements only TWN-style ternary damage.
   Gaussian weight noise (the Stage 2 v2 recipe) is a separate damage
   mode and must coexist with the existing one without changing its
   defaults.

This spec defines a narrow, documented exception to the feature freeze
that adds path-aware dim resolution + a Gaussian noise damage mode,
without altering any existing TWN behavior.

## Scope of the exception

### In-scope changes to `examples/af2_storage_tournament.py`

1. **New helper: `resolve_site_dims(target_module) -> tuple[int, int]`**
   - Returns `(in_features, out_features)` for any `nn.Linear` (or
     Conv1D) target module.
   - For `nn.Linear`: `in_features = weight.shape[1]`, `out_features = weight.shape[0]`.
   - For Conv1D: `in_features = weight.shape[0]`, `out_features = weight.shape[1]`
     (Conv1D stores weight transposed: (in, out)).
   - This is purely a query function; it does not modify any state.

2. **Refactor `build_site_adapter`** to accept `site_dims` instead of
   `hidden_size, intermediate_size`:
   ```python
   def build_site_adapter(arm_id: str, *, target_module, site_dims):
       in_features, out_features = site_dims
       ...
   ```
   - Default behavior is unchanged: when called with a down_proj module
     it produces the same adapter dimensions as before.
   - Caller is updated to pass `site_dims = resolve_site_dims(target_module)`
     instead of `(model.config.hidden_size, intermediate_size)`.

3. **New helper: `damage_target_module_gaussian(target_module, *,
   sigma, seed) -> dict`**
   - Deterministic, seeded Gaussian weight noise:
     `W' = W + sigma * std(W) * eps`, where `eps ~ N(0, 1)` from a
     `torch.Generator(seed=seed)`.
   - The damage is **in-place and frozen** (sets
     `weight.requires_grad_(False)`).
   - Records metadata: `sigma, seed, fro_norm_before, fro_norm_after, fro_ratio`.
   - **No interaction with the TWN damage path** — this is a
     separate function called when `--damage-gaussian` is set.

4. **New CLI flag: `--damage-gaussian`** (separate from `--damage-ptq`):
   ```
   --damage-gaussian          Apply Gaussian weight noise to the
                              target module's weight BEFORE adapter
                              construction. Stage 2 v2 damage mode.
                              Sigma is set via --damage-sigma.
                              Required for Stage 2 v2 EXP-RPM-LQ/LV
                              (attention projection sites) and as a
                              co-registered damage mode on down_proj
                              sites for cross-region comparison.
   --damage-sigma FLOAT       Sigma multiplier for --damage-gaussian.
                              Default 1.0 (i.e., W' = W + 1.0 * std(W) * eps).
   --damage-seed INT          Seed for the Gaussian noise generator.
                              Default 0 (deterministic per-σ).
   ```

5. **Wire `--damage-gaussian` into `run_one_seed`** alongside the
   existing `--damage-ptq` path. The two are mutually exclusive
   (explicit check: error if both flags are set).

6. **Update `pre_train_eval_if_damaged`** to a polymorphic version
   that records the damage mode name in `summary["base_state"]`.

### Out-of-scope (forbidden by freeze exception)

- No changes to arm dimensions, adapter constructors, or tournament
   protocol. The seven trained arms and two random controls are
   unchanged.
- No changes to `resolve_target_module`. It already handles arbitrary
   paths via `getattr`. Tests confirm this.
- No changes to `--damage-ptq` behavior, defaults, or metadata format.
- No changes to the Stage 1 / Stage 1.5 driver SHAs (`692e8ee`).
- No new arm types or cost vector entries.

## Construction + restoration tests

Two new tests in `tests/test_af2_driver_extension.py` (freeze exception
proof):

1. **`test_resolve_site_dims_down_proj`**: `resolve_site_dims` on a
   linear layer with shape `(hidden_size, intermediate_size)` returns
   `(intermediate_size, hidden_size)` — the same as the old hardcoded
   call.
2. **`test_resolve_site_dims_attention_proj`**: `resolve_site_dims`
   on a linear layer with shape `(hidden_size, hidden_size)` returns
   `(hidden_size, hidden_size)`.
3. **`test_damage_gaussian_seeded_reproducible`**: Two consecutive
   calls with the same `sigma, seed` produce bit-identical noise.
4. **`test_damage_gaussian_different_seed_differs`**: Two calls with
   same `sigma` but different `seed` produce different noise.
5. **`test_damage_gaussian_does_not_touch_other_modules`**: Other
   weights in the model are unchanged after `damage_target_module_gaussian`.
6. **`test_damage_gaussian_freezes_weight`**: After damage,
   `target_module.weight.requires_grad` is False.
7. **`test_damage_gaussian_records_metadata`**: The returned dict has
   `sigma, seed, fro_norm_before, fro_norm_after, fro_ratio`.
8. **`test_damage_modes_mutually_exclusive`**: Setting both
   `--damage-ptq` and `--damage-gaussian` errors out before any model
   is loaded.

## Cross-mode equivalence

For `model.layers.0.mlp.down_proj` (the AF2-D site), the new path
should produce the **same site_dims** as the old hardcoded path:
- Old: `(hidden_size=hidden, intermediate_size=intermediate)`.
- New: `resolve_site_dims(target_module).weight.shape[1]` and `[0]`
  which is `(intermediate, hidden)`.

The adapter construction is order-dependent (in_features first,
out_features second), so the new path must pass them in the same order
as the old path. This is enforced by the construction tests.

## Commit boundary

- One commit: `research: Stage 2 v2 driver extension (Gaussian damage + path-aware dims)`.
- Manifest SHA: `e7b2442` → new SHA after extension.
- Stage 1 / Stage 1.5 driver references (`692e8ee`) untouched.
- All existing tests must still pass (228/233 baseline; 5 kernel-load
  failures pre-existing).

## Reversibility

A single `git revert` of the extension commit restores the pre-v2
behavior. The TWN damage path is unmodified, so Stage 1 / Stage 1.5
results are unaffected.

## Authorized by

- Operating plan §6 (driver changes documented; mutation range bounded)
- Roadmap §2.17 (Stage 2 v2 required by RPM-006 PASS+)
- User direction 2026-08-24: "attention driver support → Gaussian
  calibration pilot → freeze qualifying sites and σ values → preregister
  Stage 2 v2 tournaments → execute"