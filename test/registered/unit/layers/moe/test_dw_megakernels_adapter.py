import torch

from sglang.srt.layers.moe.dw_megakernels import (
    _interleave_w31_rows,
    _undo_per_rank_shared_slot_remap,
    _weight_sf_physical,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.managers.io_struct import GenerateReqInput


def test_interleave_w31_rows_uses_up8_gate8_stripes():
    up = torch.arange(16).reshape(1, 16, 1)
    gate = 100 + torch.arange(16).reshape(1, 16, 1)
    result = _interleave_w31_rows(torch.cat((up, gate), dim=1), 16).flatten()
    assert result.tolist() == (
        list(range(8))
        + list(range(100, 108))
        + list(range(8, 16))
        + list(range(108, 116))
    )


def test_weight_scale_physical_layout_preserves_all_bytes():
    scale = torch.arange(128 * 8, dtype=torch.uint8).view(1, 128, 8)
    scale = scale.view(torch.float8_e4m3fn)
    result = _weight_sf_physical(scale)
    assert result.shape == (1, 1, 2, 128)
    assert result.view(torch.uint8).numel() == scale.numel()
    assert torch.equal(
        result.view(torch.uint8).flatten().sort().values,
        scale.view(torch.uint8).flatten().sort().values,
    )


def test_undo_per_rank_shared_slot_remap():
    # Four routed experts + one shared slot per rank:
    # physical routed IDs [0..3, 5..8] -> logical [0..7].
    physical = torch.tensor([[0, 3, 5, 8]], dtype=torch.int64)
    logical = _undo_per_rank_shared_slot_remap(physical, 4, 1)
    assert logical.tolist() == [[0, 3, 4, 7]]


def test_dw_megamoe_allows_cutedsl_weight_layout_runner(monkeypatch):
    from sglang.srt.layers.moe.moe_runner import runner as runner_module

    monkeypatch.setenv("SGLANG_MEGAMOE_KERNEL_BACKEND", "dw_nvfp4")
    monkeypatch.setattr(
        runner_module, "get_moe_a2a_backend", lambda: MoeA2ABackend.MEGAMOE
    )
    runner = runner_module.MoeRunner(
        MoeRunnerBackend.FLASHINFER_CUTEDSL, MoeRunnerConfig()
    )
    assert runner.runner_core is None
    assert runner.fused_func is None


def test_native_batch_preserves_per_item_dp_routes():
    request = GenerateReqInput(
        input_ids=[[1, 2], [3, 4]],
        sampling_params={"max_new_tokens": 1},
        routed_dp_rank=[1, 0],
    )
    request.normalize_batch_and_arguments()
    assert request[0].routed_dp_rank == 1
    assert request[1].routed_dp_rank == 0
