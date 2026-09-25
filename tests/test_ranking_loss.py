import torch

from evaluate import compute_score_metrics
import train


def test_pairwise_ranking_loss_rewards_the_same_order_as_target_quality():
    targets = torch.tensor([0.9, 0.5, 0.1])
    ordered_predictions = torch.tensor([0.8, 0.4, 0.2])
    inverted_predictions = torch.tensor([0.2, 0.4, 0.8])

    assert hasattr(train, "compute_pairwise_ranking_loss")
    assert train.compute_pairwise_ranking_loss(ordered_predictions, targets) == 0.0
    assert train.compute_pairwise_ranking_loss(inverted_predictions, targets) > 0.0


def test_pairwise_ordering_accuracy_reports_the_target_order():
    targets = torch.tensor([0.9, 0.5, 0.1])

    assert compute_score_metrics(
        torch.tensor([0.8, 0.4, 0.2]), targets
    )["pairwise_ordering_accuracy"] == 1.0
    assert compute_score_metrics(
        torch.tensor([0.2, 0.4, 0.8]), targets
    )["pairwise_ordering_accuracy"] == 0.0
