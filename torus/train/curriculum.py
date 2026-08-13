"""Progressive curriculum schedule for residual-plane training.

Rather than training all `k` residual planes from the start, TORUS
uses a curriculum:

- Stage 1: the *primary* plane alone. Train until a quality plateau.
- Stage 2: freeze the primary, introduce plane #2. Train the residual
  to fix what plane #1 got wrong.
- Stage 3: optionally introduce plane #3 the same way.
- (etc.)

This is directly informed by the *residual* structure: plane #i is
trained to capture the approximation error of plane #(i-1). Freezing
the earlier planes is the cleanest way to enforce that.

The schedule is purely declarative; the trainer consults `stage_at(step)`
to decide which planes are learnable and which are frozen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class CurriculumStage:
    """One stage in the curriculum."""
    n_planes_active: int        # how many planes can be trained / used
    threshold: float = 0.7     # ternary sparsity threshold for this stage
    min_steps: int = 1         # minimum steps before this stage may advance


@dataclass
class CurriculumSchedule:
    """A list of `CurriculumStage`s consulted by step index."""
    stages: list[CurriculumStage] = field(default_factory=list)

    @classmethod
    def progressive(cls, steps_per_stage: Iterable[int],
                    thresholds: Iterable[float] | None = None) -> "CurriculumSchedule":
        """Build a default progressive schedule.

        Args:
            steps_per_stage: how many steps to spend in each stage.
                Length determines the number of planes: a 3-stage
                schedule yields up to 3 active planes.
            thresholds: matching-length threshold per stage. Defaults
                to constant 0.7.
        """
        steps = list(steps_per_stage)
        ths = list(thresholds) if thresholds is not None else [0.7] * len(steps)
        if len(ths) != len(steps):
            raise ValueError("steps_per_stage and thresholds must match length")
        stages = [
            CurriculumStage(n_planes_active=i + 1, threshold=ths[i], min_steps=s)
            for i, s in enumerate(steps)
        ]
        return cls(stages=stages)

    def stage_at(self, step: int) -> CurriculumStage:
        """Return the active stage for the given training step.

        Stages are cumulative: the *i*-th stage becomes active once the
        *i*-th minimum is reached. Earlier stages are not necessarily
        *inactive* (a residual plane trained in stage 2 stays active
        in stage 3), but the active plane count grows monotonically.
        """
        cumulative = 0
        for stage in self.stages:
            cumulative += stage.min_steps
            if step < cumulative:
                return stage
        return self.stages[-1]

    def n_planes_active_at(self, step: int) -> int:
        return self.stage_at(step).n_planes_active

    def max_planes(self) -> int:
        if not self.stages:
            return 0
        return max(s.n_planes_active for s in self.stages)
