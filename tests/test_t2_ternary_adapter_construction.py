"""Integration test: T2TernaryAdapter constructs and sets is_untrained.

This is the regression test for the 7383b57 driver bug where
T2TernaryAdapter.__init__ lost the self.latent creation. The
synthetic-aggregate tests (test_audit_*) don't exercise adapter
construction, so this regression went undetected.

The test:
  1. Constructs a T2TernaryAdapter (must not raise NameError).
  2. Verifies self.latent exists and is a torch.nn.Parameter.
  3. Verifies self.is_untrained is False for train=True, True
     for train=False.

The full driver end-to-end test (including forward pass) is
covered by the EXP-RPM-000 reproduction itself on Legion. A unit
test that constructs the adapter and inspects its attributes is
sufficient to catch the 7383b57 regression.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _run_snippet(snippet: str) -> str:
    """Run a python snippet in a subprocess that imports the driver
    via runpy (which sets up the module's __name__ correctly so
    dataclass introspection works)."""
    driver_path = EXAMPLES / "af2_storage_tournament.py"
    wrapper = (
        "import runpy\n"
        f"ns = runpy.run_path(r'{driver_path}', run_name='_af2_driver')\n"
        "import torch\n"
        + snippet
    )
    r = subprocess.run([sys.executable, "-c", wrapper],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.fail(f"subprocess failed: stdout={r.stdout!r} stderr={r.stderr!r}")
    return r.stdout.strip()


def test_t2_ternary_constructs_with_latent():
    """The 7383b57 regression: __init__ did not create self.latent."""
    snippet = (
        "ad = ns['T2TernaryAdapter'](in_features=16, out_features=8, train=True)\n"
        "assert ad.latent is not None, 'no self.latent'\n"
        "assert isinstance(ad.latent, torch.nn.Parameter)\n"
        "assert tuple(ad.latent.shape) == (8, 16), f'wrong shape {ad.latent.shape}'\n"
        "assert ad.latent.requires_grad is True\n"
        "assert ad.is_untrained is False, f'is_untrained={ad.is_untrained}'\n"
        "print('OK')\n"
    )
    assert _run_snippet(snippet) == "OK"


def test_t2_ternary_untrained_sets_is_untrained_and_no_grad():
    """The actual intended fix from CHANGELOG 0.16.5: is_untrained
    now correctly reflects train=False."""
    snippet = (
        "ad = ns['T2TernaryAdapter'](in_features=16, out_features=8, train=False)\n"
        "assert ad.is_untrained is True, f'is_untrained={ad.is_untrained}'\n"
        "assert ad.latent.requires_grad is False\n"
        "print('OK')\n"
    )
    assert _run_snippet(snippet) == "OK"



def test_t2_ternary_patch_replaces_target_forward():
    """Regression for the 4cf3860 bug: T2TernaryAdapter.patch defined
    `residual` but did not call _patch_module_forward, so the
    target_module.forward was never replaced. Verify the patch DOES
    replace target_module.forward and DOES contribute to the output."""
    snippet = (
        "import torch\n"
        "import torch.nn as nn\n"
        "ad = ns['T2TernaryAdapter'](in_features=16, out_features=8, train=True, init_seed=42)\n"
        "base = nn.Linear(16, 8, bias=False)\n"
        "base.weight.data = torch.zeros_like(base.weight.data)\n"
        "ad.patch(base)\n"
        "import inspect\n"
        "assert 'patched_forward' in str(base.forward), f'forward not patched: {base.forward}'\n"
        "with torch.no_grad():\n"
        "    ad.latent.fill_(10.0)\n"
        "x = torch.randn(2, 16)\n"
        "y = base(x)\n"
        "assert y.abs().sum() > 0, f'patched forward produced zero: {y}'\n"
        "print('OK')\n"
    )
    assert _run_snippet(snippet) == "OK"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])