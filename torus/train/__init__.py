"""Training-time primitives: distillation loss, STE, curriculum, loop."""
from torus.train.losses import (
    DistillationConfig,
    combined_distillation_loss,
    expert_route_loss,
    intermediate_alignment_loss,
    kl_divergence,
)
from torus.train.ste import TernarySTE, ternary_quantize_with_ste
from torus.train.curriculum import CurriculumSchedule, CurriculumStage
from torus.train.loop import (
    DistillationBatch,
    DistillationTrainer,
    TrainingConfig,
    TrainingStats,
)

__all__ = [
    "DistillationConfig",
    "kl_divergence",
    "intermediate_alignment_loss",
    "expert_route_loss",
    "combined_distillation_loss",
    "TernarySTE",
    "ternary_quantize_with_ste",
    "CurriculumStage",
    "CurriculumSchedule",
    "DistillationBatch",
    "TrainingConfig",
    "TrainingStats",
    "DistillationTrainer",
]
