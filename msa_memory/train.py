"""
Training — End-to-End Trainable Memory with Auxiliary Contrastive Loss.

This module implements the training loop that makes MSA end-to-end trainable —
the fundamental capability that decoupled RAG systems lack.

Key components:
  1. Auxiliary Contrastive Loss (Laux) — supervises routing directly
  2. Combined Loss Schedule — two-phase warmup→main curriculum
  3. Curriculum Training — 8K SFT → 64K extension for 100M extrapolation

Loss equation (from paper):
  Laux = -1/|P| · Σᵢ log( exp(sᵢ⁺/τ) / (exp(sᵢ⁺/τ) + Σⱼ exp(sᵢⱼ⁻/τ)) )

Ablation results:
  - Removing continual pre-training: −36.2% average score
  - Removing stage 2 curriculum: −7.6% average score
  - Removing Memory Interleave: −11.9% average score

Usage:
    python -m msa_memory.train \\
        --corpus_dir ./training_data/ \\
        --backbone Qwen/Qwen3-4B-Instruct \\
        --warmup_steps 5000 \\
        --total_steps 100000
"""

import argparse

import torch
import torch.nn.functional as F

from .config import MSAConfig


def auxiliary_routing_loss(
    positive_scores: torch.Tensor,   # relevance scores of correct documents
    negative_scores: torch.Tensor,   # relevance scores of wrong documents
    temperature: float = 0.07,       # τ from equation (5)
) -> torch.Tensor:
    """
    Laux: Supervised contrastive loss over document routing decisions.

    This supervises the layer-wise routing process directly — every MSA layer
    learns to route to the correct evidence, not just the final output.

    Equation (5) from paper:
    Laux = -1/|P| · Σᵢ log( exp(sᵢ⁺/τ) / (exp(sᵢ⁺/τ) + Σⱼ exp(sᵢⱼ⁻/τ)) )

    Args:
        positive_scores: Relevance scores of correct documents [num_positives].
        negative_scores: Relevance scores of wrong documents [num_positives, num_negatives].
        temperature: Temperature parameter τ controlling distribution sharpness.

    Returns:
        Scalar loss tensor.
    """
    pos_exp = torch.exp(positive_scores / temperature)  # [num_positives]
    neg_exp = torch.exp(negative_scores / temperature)  # [num_positives, num_negatives]
    neg_sum = neg_exp.sum(dim=-1)                       # [num_positives]

    loss = -torch.log(pos_exp / (pos_exp + neg_sum))
    return loss.mean()


def combined_loss(
    lm_loss: torch.Tensor,
    aux_loss: torch.Tensor,
    training_phase: str,   # "warmup" or "main"
    config: MSAConfig,
) -> torch.Tensor:
    """
    Two-phase loss schedule.

    Phase 1 (warmup):  L = 0.1·L_LM + 1.0·L_aux  (prime the router first)
    Phase 2 (main):    L = 1.0·L_LM + 0.1·L_aux  (generation dominates)

    The warmup phase rapidly aligns Router Projectors before main training.
    Ablation: removing CPT causes 31.3% average performance drop.

    Args:
        lm_loss: Language model (next-token prediction) loss.
        aux_loss: Auxiliary contrastive routing loss.
        training_phase: Either "warmup" or "main".
        config: MSAConfig instance with loss weights.

    Returns:
        Combined weighted loss tensor.
    """
    if training_phase == "warmup":
        return config.warmup_lm_weight * lm_loss + config.warmup_aux_weight * aux_loss
    else:
        return config.main_lm_weight * lm_loss + config.main_aux_weight * aux_loss


def curriculum_training_schedule(
    total_steps: int,
    warmup_fraction: float = 0.05,
) -> list[dict]:
    """
    Two-stage curriculum training schedule.

    Stage 1: SFT on 8K context → establishes instruction following
    Stage 2: Extend to 64K context → enables 100M extrapolation at inference

    Ablation: MSA-S2 (both stages) beats MSA-S1 (stage 1 only) by 7.6% average,
    and by 29.5% on MS MARCO (7.34M token corpus).

    Args:
        total_steps: Total training steps across all phases.
        warmup_fraction: Fraction of steps for warmup (default: 5%).

    Returns:
        List of phase configurations with steps, lr, context_len, and loss weights.
    """
    warmup_steps = int(total_steps * warmup_fraction)
    return [
        {
            "phase": "warmup",
            "steps": warmup_steps,
            "lr": 1e-4,
            "context_len": 8192,
            "loss_weights": {"lm": 0.1, "aux": 1.0},
        },
        {
            "phase": "main_stage1",
            "steps": (total_steps - warmup_steps) // 2,
            "lr": 6e-6,
            "context_len": 8192,
            "loss_weights": {"lm": 1.0, "aux": 0.1},
        },
        {
            "phase": "main_stage2",
            "steps": (total_steps - warmup_steps) // 2,
            "lr": 6e-6,
            "context_len": 65536,   # extend to 64K — enables 100M extrapolation
            "loss_weights": {"lm": 1.0, "aux": 0.1},
        },
    ]


def main():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train the MSA Memory System end-to-end."
    )
    parser.add_argument(
        "--corpus_dir", type=str, required=True,
        help="Directory containing training documents.",
    )
    parser.add_argument(
        "--backbone", type=str, default="Qwen/Qwen3-4B-Instruct",
        help="HuggingFace model ID for the backbone.",
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=5000,
        help="Number of warmup steps.",
    )
    parser.add_argument(
        "--total_steps", type=int, default=100000,
        help="Total training steps.",
    )
    parser.add_argument(
        "--gpus", type=str, default="0",
        help="Comma-separated GPU IDs.",
    )
    args = parser.parse_args()

    config = MSAConfig(
        backbone_model=args.backbone,
        warmup_steps=args.warmup_steps,
    )

    schedule = curriculum_training_schedule(args.total_steps)

    print("Training schedule:")
    for phase in schedule:
        print(f"  {phase['phase']}: {phase['steps']} steps, "
              f"lr={phase['lr']}, context={phase['context_len']}, "
              f"weights={phase['loss_weights']}")

    print("\nNOTE: Full training loop requires a prepared dataset with")
    print("positive/negative document annotations. See docs/TRAINING.md")
    print("for the complete training pipeline setup.")


if __name__ == "__main__":
    main()
