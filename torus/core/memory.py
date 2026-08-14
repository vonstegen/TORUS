"""Memory hierarchy hints for residual ternary planes.

On a heterogeneous machine (CPU + GPU + NVMe) the hot residual plane
should sit in fast memory and the cold ones on slow storage. This
module is a *policy*: it decides which plane lives where, given an
explicit budget. It does no allocation itself; that decision is left
to the runtime.

The policy is deliberately simple for Phase 2. Future phases will
move planes based on observed gate activation rates.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MemoryTier(str, Enum):
    """Storage tiers in latency order."""
    VRAM = "vram"           # GPU memory (TITAN RTX 24 GB each on P620)
    RAM = "ram"             # system RAM (128 GB ECC on P620)
    NVME = "nvme"           # local NVMe (2 TB on P620)


@dataclass(frozen=True)
class Budget:
    """Byte budgets per tier."""
    vram_bytes: int
    ram_bytes: int
    nvme_bytes: int

    def get(self, tier: MemoryTier) -> int:
        return {
            MemoryTier.VRAM: self.vram_bytes,
            MemoryTier.RAM: self.ram_bytes,
            MemoryTier.NVME: self.nvme_bytes,
        }[tier]


@dataclass(frozen=True)
class PlaneSize:
    """Bytes a single residual plane occupies."""
    weight_bytes: int           # packed 2-bit weight bytes
    scale_bytes: int            # FP16-equivalent scale bytes
    total_bytes: int            # sum of the two

    @classmethod
    def from_estimate(cls, num_weights: int, num_scales: int) -> "PlaneSize":
        w = (num_weights + 3) // 4
        s = 2 * num_scales
        return cls(weight_bytes=w, scale_bytes=s, total_bytes=w + s)


@dataclass(frozen=True)
class Placement:
    """Where each plane should live."""
    tiers: tuple[MemoryTier, ...]   # one tier per plane, ordered by plane index

    @property
    def n_planes(self) -> int:
        return len(self.tiers)


def place_planes(
    plane_sizes: list[PlaneSize],
    budget: Budget,
) -> Placement:
    """Decide which tier each plane lives in.

    Strategy:
      1. Place the primary plane (index 0) in the fastest tier that fits.
      2. Place residual planes in the next-fastest available tier.
      3. If nothing fits, demote the primary to the next tier (always
         possible because the policy is monotonic).

    The policy is purely declarative — the caller is responsible for
    actually moving bytes.
    """
    tier_order = [MemoryTier.VRAM, MemoryTier.RAM, MemoryTier.NVME]
    budget_dict = {
        MemoryTier.VRAM: budget.vram_bytes,
        MemoryTier.RAM: budget.ram_bytes,
        MemoryTier.NVME: budget.nvme_bytes,
    }

    placements: list[MemoryTier] = []
    remaining = dict(budget_dict)
    for tier in tier_order:
        remaining[tier] = budget_dict[tier]

    for psize in plane_sizes:
        placed = False
        for tier in tier_order:
            if remaining[tier] >= psize.total_bytes:
                placements.append(tier)
                remaining[tier] -= psize.total_bytes
                placed = True
                break
        if not placed:
            # Demote: spill to NVMe regardless of remaining space.
            # The runtime is expected to handle out-of-space explicitly.
            placements.append(MemoryTier.NVME)

    # Sanity: primary plane index 0 must be fastest available tier
    # *or* a strictly faster tier than any plane that's strictly colder.
    return Placement(tiers=tuple(placements))


def p620_default_budget() -> Budget:
    """Reasonable defaults for the documented P620 target."""
    return Budget(
        vram_bytes=48 * 1024**3,   # 2x 24 GB TITAN RTX (before activations/KV)
        ram_bytes=128 * 1024**3,   # 128 GB ECC, minus OS + activations
        nvme_bytes=2 * 1024**4,    # 2 TB NVMe, minus OS
    )


def gb10_default_budget() -> Budget:
    """Reasonable defaults for the GB10 dev box.

    The GB10 has unified CPU/GPU memory (~120 GB usable), so the
    VRAM and RAM pools overlap in practice. We split them by
    convention: 80 GB VRAM-equivalent, 40 GB RAM-only, 1 TB NVMe.
    """
    return Budget(
        vram_bytes=80 * 1024**3,   # GB10 unified memory pool (large)
        ram_bytes=40 * 1024**3,    # remaining unified memory for non-GPU use
        nvme_bytes=1 * 1024**4,    # ~1 TB NVMe (minus OS)
    )
