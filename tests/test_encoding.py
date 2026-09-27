from __future__ import annotations

import math

import torch

from scope.encoding import SightlineCoordinatePE


def test_tied_init_copies_values_without_sharing_parameters() -> None:
    encoding = SightlineCoordinatePE(
        dim=16,
        plucker_init="small",
        plucker_mlp_hidden=8,
        tied_init=True,
    )

    for query, key in zip(encoding.eq.parameters(), encoding.ek.parameters(), strict=True):
        assert torch.equal(query, key)
        assert query is not key


def test_checkpoint_loading_overrides_tied_initialization() -> None:
    source = SightlineCoordinatePE(
        dim=16,
        plucker_init="small",
        plucker_mlp_hidden=8,
        tied_init=False,
    )
    target = SightlineCoordinatePE(
        dim=16,
        plucker_init="small",
        plucker_mlp_hidden=8,
        tied_init=True,
    )

    target.load_state_dict(source.state_dict())

    for source_parameter, target_parameter in zip(
        source.parameters(), target.parameters(), strict=True
    ):
        assert torch.equal(source_parameter, target_parameter)


def test_flip_and_plucker_eps_are_applied_during_decomposition() -> None:
    encoding = SightlineCoordinatePE(dim=8, plucker_eps=1e-3)
    plucker = torch.tensor([[[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]]])

    feat_q, feat_k, log_scale = encoding.decompose_plucker(plucker)

    assert torch.equal(feat_q[..., :3], plucker[..., :3])
    assert torch.equal(feat_k[..., 3:6], plucker[..., :3])
    assert torch.allclose(log_scale, torch.full_like(log_scale, math.log(1e-3)))


def test_log_scale_augmentation_only_runs_in_training() -> None:
    encoding = SightlineCoordinatePE(
        dim=8,
        log_scale_aug_prob=1.0,
        log_scale_aug_range=(0.5, 0.5),
    )
    log_scale = torch.zeros(2, 4, 1)

    encoding.train()
    assert torch.equal(
        encoding._maybe_perturb_log_scale(log_scale),
        torch.full_like(log_scale, 0.5),
    )

    encoding.eval()
    assert torch.equal(encoding._maybe_perturb_log_scale(log_scale), log_scale)
