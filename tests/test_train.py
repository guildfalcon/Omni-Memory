"""
Tests for the training module (auxiliary loss and curriculum schedule).
"""

import torch

from msa_memory.config import MSAConfig
from msa_memory.train import (
    auxiliary_routing_loss,
    combined_loss,
    curriculum_training_schedule,
)


class TestAuxiliaryRoutingLoss:
    """Test suite for the contrastive routing loss (Laux)."""

    def test_perfect_routing(self):
        """Loss should be low when positives score higher than negatives."""
        positive_scores = torch.tensor([1.0, 1.0, 1.0])
        negative_scores = torch.tensor([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ])
        loss = auxiliary_routing_loss(positive_scores, negative_scores)
        assert loss.item() < 0.1  # should be very small

    def test_bad_routing(self):
        """Loss should be high when negatives score higher than positives."""
        positive_scores = torch.tensor([0.0, 0.0, 0.0])
        negative_scores = torch.tensor([
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ])
        loss = auxiliary_routing_loss(positive_scores, negative_scores)
        assert loss.item() > 1.0  # should be large

    def test_temperature_effect(self):
        """Higher temperature should produce lower loss gradient (softer)."""
        pos = torch.tensor([0.5])
        neg = torch.tensor([[0.3]])

        loss_low_t = auxiliary_routing_loss(pos, neg, temperature=0.01)
        loss_high_t = auxiliary_routing_loss(pos, neg, temperature=1.0)

        # Lower temperature → sharper distribution → different loss
        assert not torch.isclose(loss_low_t, loss_high_t)

    def test_output_is_scalar(self):
        """Loss should be a scalar tensor."""
        pos = torch.tensor([1.0, 0.5])
        neg = torch.tensor([[0.2, 0.1], [0.3, 0.4]])
        loss = auxiliary_routing_loss(pos, neg)
        assert loss.dim() == 0


class TestCombinedLoss:
    """Test suite for the two-phase combined loss schedule."""

    def test_warmup_phase_weights(self):
        """Warmup: L_aux dominates (weight=1.0), L_LM suppressed (weight=0.1)."""
        config = MSAConfig()
        lm = torch.tensor(10.0)
        aux = torch.tensor(10.0)

        total = combined_loss(lm, aux, "warmup", config)
        # 0.1 * 10 + 1.0 * 10 = 11.0
        assert torch.isclose(total, torch.tensor(11.0))

    def test_main_phase_weights(self):
        """Main: L_LM dominates (weight=1.0), L_aux suppressed (weight=0.1)."""
        config = MSAConfig()
        lm = torch.tensor(10.0)
        aux = torch.tensor(10.0)

        total = combined_loss(lm, aux, "main", config)
        # 1.0 * 10 + 0.1 * 10 = 11.0 (same total, different arrangement)
        assert torch.isclose(total, torch.tensor(11.0))


class TestCurriculumSchedule:
    """Test suite for the curriculum training schedule."""

    def test_three_phases(self):
        """Schedule should have exactly 3 phases."""
        schedule = curriculum_training_schedule(100000)
        assert len(schedule) == 3

    def test_phase_names(self):
        """Phases should be warmup, main_stage1, main_stage2."""
        schedule = curriculum_training_schedule(100000)
        assert schedule[0]["phase"] == "warmup"
        assert schedule[1]["phase"] == "main_stage1"
        assert schedule[2]["phase"] == "main_stage2"

    def test_total_steps(self):
        """Total steps across phases should equal input."""
        total = 100000
        schedule = curriculum_training_schedule(total)
        total_scheduled = sum(p["steps"] for p in schedule)
        assert total_scheduled == total

    def test_context_extension(self):
        """Stage 2 should extend context from 8K to 64K."""
        schedule = curriculum_training_schedule(100000)
        assert schedule[0]["context_len"] == 8192
        assert schedule[1]["context_len"] == 8192
        assert schedule[2]["context_len"] == 65536

    def test_warmup_fraction(self):
        """Warmup should be the specified fraction of total steps."""
        total = 100000
        fraction = 0.1
        schedule = curriculum_training_schedule(total, warmup_fraction=fraction)
        assert schedule[0]["steps"] == int(total * fraction)
