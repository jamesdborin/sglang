"""SGLang adapter for the external ``dw-megakernels`` MegaMoE package.

The external kernel deliberately exposes its native launch ABI instead of
owning framework policy. This module translates SGLang's ModelOpt NVFP4
weights, routing layout, activation scales, and expert-parallel process group
into that ABI. It is imported only when ``SGLANG_MEGAMOE_KERNEL_BACKEND`` is
set to ``dw_nvfp4``; stock SGLang paths therefore remain dependency-free.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Any

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_DW_NVFP4_ALIASES = {"dw", "dw-megakernels", "dw_megakernels", "dw_nvfp4"}


def use_dw_nvfp4_mega_moe() -> bool:
    """Whether MegaMoE should use the external NVFP4 implementation."""

    return envs.SGLANG_MEGAMOE_KERNEL_BACKEND.get().strip().lower() in _DW_NVFP4_ALIASES


def _interleave_w31_rows(tensor: torch.Tensor, intermediate: int) -> torch.Tensor:
    """Convert canonical ``[all up, all gate]`` rows to 8-row stripes."""

    if tensor.ndim != 3 or tensor.shape[1] != 2 * intermediate:
        raise ValueError("dw NVFP4 W31 interleave expects [E,2I,K]")
    granularity = 8
    if intermediate % granularity:
        raise ValueError("dw NVFP4 W31 interleave requires I divisible by 8")
    experts, _, columns = tensor.shape
    up = tensor[:, :intermediate].reshape(
        experts, intermediate // granularity, granularity, columns
    )
    gate = tensor[:, intermediate:].reshape(
        experts, intermediate // granularity, granularity, columns
    )
    return (
        torch.stack((up, gate), dim=2)
        .reshape(experts, 2 * intermediate, columns)
        .contiguous()
    )


def _weight_sf_physical(scale: torch.Tensor) -> torch.Tensor:
    """Build the kernel's ``[E,N/128,K/64,128]`` Int32 scale storage."""

    if scale.ndim != 3 or scale.dtype != torch.float8_e4m3fn:
        raise ValueError("dw NVFP4 weight scales must be E4M3 [E,N,K/16]")
    experts, rows, columns = scale.shape
    if rows % 128 or columns % 4:
        raise ValueError("dw NVFP4 scales require N%128==0 and (K/16)%4==0")
    return (
        scale.reshape(experts, rows // 128, 4, 32, columns // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .view(torch.int32)
        .reshape(experts, rows // 128, columns // 4, 128)
    )


def _undo_per_rank_shared_slot_remap(
    routed_ids: torch.Tensor, num_local_routed: int, num_shared: int
) -> torch.Tensor:
    """Map SGLang's per-rank shared-slot IDs back to logical routed IDs."""

    if num_shared == 0:
        return routed_ids
    physical_stride = num_local_routed + num_shared
    owner = torch.div(routed_ids, physical_stride, rounding_mode="floor")
    local = routed_ids - owner * physical_stride
    return owner * num_local_routed + local


def _local_scale_rows(scale: torch.Tensor, layer: Any) -> torch.Tensor:
    """Select the current rank from a local or global expert-scale tensor."""

    local = layer.num_local_experts
    if scale.shape[0] == local:
        return scale
    start = layer.moe_ep_rank * local
    stop = start + local
    if scale.shape[0] < stop:
        raise ValueError(
            f"cannot select local NVFP4 scales [{start}:{stop}] from {tuple(scale.shape)}"
        )
    return scale[start:stop]


def build_dw_nvfp4_experts_weights(layer: Any) -> None:
    """Convert one loaded ModelOpt NVFP4 FusedMoE layer in-place."""

    if getattr(layer, "_dw_nvfp4_weights_built", False):
        return

    num_shared = int(layer.num_fused_shared_experts)
    if num_shared not in (0, 1):
        raise ValueError("dw NVFP4 MegaMoE supports zero or one fused shared expert")
    num_local_routed = int(layer._num_local_routed)
    if layer.num_local_experts != num_local_routed + num_shared:
        raise ValueError("unexpected SGLang local routed/shared expert layout")

    intermediate = int(layer.intermediate_size_per_partition)
    hidden = int(layer.hidden_size)
    w13 = layer.w13_weight.data
    w13_sf = layer.w13_weight_scale.data
    if w13.shape[1] != 2 * intermediate:
        raise ValueError(f"unexpected NVFP4 W13 shape {tuple(w13.shape)}")

    # SGLang's non-TRTLLM ModelOpt loader materializes [gate, up]. The external
    # kernel consumes canonical [up, gate] before its granularity-8 row stripe.
    gate, up = w13.chunk(2, dim=1)
    gate_sf, up_sf = w13_sf.chunk(2, dim=1)
    w31 = _interleave_w31_rows(torch.cat((up, gate), dim=1), intermediate)
    w31_sf = _weight_sf_physical(
        _interleave_w31_rows(torch.cat((up_sf, gate_sf), dim=1), intermediate)
    )
    w2_sf = _weight_sf_physical(layer.w2_weight_scale.data)

    # Reuse the registered parameter storage names so model memory accounting
    # remains accurate and the unneeded stock layout is released.
    layer.w13_weight.data = w31
    layer.w13_weight_scale.data = w31_sf.view(torch.float8_e4m3fn).reshape_as(w13_sf)
    layer.w2_weight.data = layer.w2_weight.data.contiguous()
    layer.w2_weight_scale.data = w2_sf.view(torch.float8_e4m3fn).reshape_as(
        layer.w2_weight_scale.data
    )

    routed = slice(0, num_local_routed)
    shared = slice(num_local_routed, num_local_routed + num_shared)
    layer._dw_nvfp4_l1_weight = layer.w13_weight.data[routed].reshape(
        num_local_routed * 2 * intermediate, hidden // 2
    )
    layer._dw_nvfp4_l1_weight_sf = (
        layer.w13_weight_scale.data[routed]
        .view(torch.int32)
        .reshape(num_local_routed * 2 * intermediate, hidden // 64)
    )
    layer._dw_nvfp4_l2_weight = layer.w2_weight.data[routed].reshape(
        num_local_routed * hidden, intermediate // 2
    )
    layer._dw_nvfp4_l2_weight_sf = (
        layer.w2_weight_scale.data[routed]
        .view(torch.int32)
        .reshape(num_local_routed * hidden, intermediate // 64)
    )

    # One common input quantizer feeds every routed expert. Expert-specific
    # ModelOpt dequantization remains in the per-expert GEMM alpha vectors.
    input_dequant = layer.w13_input_scale.max().to(torch.float32).reshape(1)
    w13_scale2 = layer.w13_weight_scale_2.to(torch.float32)
    if w13_scale2.ndim == 2 and w13_scale2.shape[1] > 1:
        gate_scale2 = w13_scale2[:, 0]
        up_scale2 = w13_scale2[:, 1]
    else:
        gate_scale2 = w13_scale2.reshape(-1)
        up_scale2 = gate_scale2
    l1_gate_alpha = input_dequant * gate_scale2
    l1_up_alpha = input_dequant * up_scale2

    w2_input_dequant = _local_scale_rows(
        layer.w2_input_scale.to(torch.float32), layer
    ).reshape(-1)
    l2_alpha = w2_input_dequant * layer.w2_weight_scale_2.to(torch.float32).reshape(-1)
    intermediate_global_scale = w2_input_dequant.reciprocal()

    if num_shared:
        layer._dw_nvfp4_shared_l1_weight = layer.w13_weight.data[shared].reshape(
            2 * intermediate, hidden // 2
        )
        layer._dw_nvfp4_shared_l1_weight_sf = (
            layer.w13_weight_scale.data[shared]
            .view(torch.int32)
            .reshape(2 * intermediate, hidden // 64)
        )
        layer._dw_nvfp4_shared_l2_weight = layer.w2_weight.data[shared].reshape(
            hidden, intermediate // 2
        )
        layer._dw_nvfp4_shared_l2_weight_sf = (
            layer.w2_weight_scale.data[shared]
            .view(torch.int32)
            .reshape(hidden, intermediate // 64)
        )
        layer._dw_nvfp4_shared_l1_gate_alpha = l1_gate_alpha[shared]
        layer._dw_nvfp4_shared_l1_up_alpha = l1_up_alpha[shared]
        layer._dw_nvfp4_shared_l2_alpha = l2_alpha[shared]
    else:
        # The CuTe host wrapper constructs all descriptors even when their
        # consumers compile out, so supply non-empty aliases of owned storage.
        layer._dw_nvfp4_shared_l1_weight = layer._dw_nvfp4_l1_weight[:1, :1]
        layer._dw_nvfp4_shared_l1_weight_sf = layer._dw_nvfp4_l1_weight_sf[:1, :1]
        layer._dw_nvfp4_shared_l2_weight = layer._dw_nvfp4_l2_weight[:1, :1]
        layer._dw_nvfp4_shared_l2_weight_sf = layer._dw_nvfp4_l2_weight_sf[:1, :1]
        layer._dw_nvfp4_shared_l1_gate_alpha = l1_gate_alpha[:1]
        layer._dw_nvfp4_shared_l1_up_alpha = l1_up_alpha[:1]
        layer._dw_nvfp4_shared_l2_alpha = l2_alpha[:1]

    layer._dw_nvfp4_l1_gate_alpha = l1_gate_alpha[routed].contiguous()
    layer._dw_nvfp4_l1_up_alpha = l1_up_alpha[routed].contiguous()
    layer._dw_nvfp4_l2_alpha = l2_alpha[routed].contiguous()
    layer._dw_nvfp4_intermediate_global_scale = intermediate_global_scale.contiguous()
    layer._dw_nvfp4_input_global_scale = input_dequant.reciprocal().contiguous()
    layer._dw_nvfp4_weights_built = True
    layer._mega_moe_weights_built = True


def _fake_arguments(module: Any, cutlass: Any, cute: Any) -> list[Any]:
    """Construct the specialization's fixed CuTeDSL compilation signature."""

    fake = cute.runtime.make_fake_compact_tensor
    u8, i32, f32, bf16 = (
        cutlass.Uint8,
        cutlass.Int32,
        cutlass.Float32,
        cutlass.BFloat16,
    )
    max_tokens = module.MAX_TOKENS_PER_RANK
    hidden = module.HIDDEN
    intermediate = module.INTERMEDIATE
    local_experts = module.EXPERTS_PER_RANK
    shared_experts = module.NUM_SHARED_EXPERTS
    shared_descriptor_experts = module.SHARED_DESCRIPTOR_EXPERTS
    ring_tokens = module.DATA_RING_TOKENS
    scale_ring_tokens = module.SCALE_RING_TOKENS
    shared_scale_tokens = module.NVFP4_SHARED_SCALE_TOKENS

    def matrix(dtype: Any, shape: tuple[int, int], alignment: int = 128) -> Any:
        return fake(dtype, shape, stride_order=(1, 0), assumed_align=alignment)

    def vector(dtype: Any, extent: int, alignment: int = 16) -> Any:
        return fake(dtype, (extent,), stride_order=(0,), assumed_align=alignment)

    dummy_shared = shared_experts == 0
    shared_shape = (1, 1)
    return [
        matrix(bf16, (max_tokens, hidden)),
        vector(i32, local_experts),
        cutlass.Uint32(min(max_tokens, 8192)),
        vector(u8, module.NVFP4_SYMMETRIC_BYTES, alignment=128),
        tuple(cutlass.Int64(0) for _ in range(72)),
        cutlass.Uint32(0),
        matrix(u8, (ring_tokens, hidden // 2)),
        matrix(i32, (scale_ring_tokens, hidden // 64), alignment=16),
        matrix(u8, (local_experts * 2 * intermediate, hidden // 2)),
        matrix(i32, (local_experts * 2 * intermediate, hidden // 64), alignment=16),
        matrix(u8, (ring_tokens, intermediate // 2)),
        matrix(u8, (ring_tokens, intermediate // 2)),
        matrix(i32, (scale_ring_tokens, intermediate // 64), alignment=16),
        matrix(u8, (local_experts * hidden, intermediate // 2)),
        matrix(i32, (local_experts * hidden, intermediate // 64), alignment=16),
        matrix(u8, shared_shape if dummy_shared else (max_tokens, hidden // 2)),
        matrix(
            i32,
            shared_shape if dummy_shared else (shared_scale_tokens, hidden // 64),
            alignment=16,
        ),
        matrix(
            u8,
            shared_shape
            if dummy_shared
            else (shared_descriptor_experts * 2 * intermediate, hidden // 2),
        ),
        matrix(
            i32,
            shared_shape
            if dummy_shared
            else (shared_descriptor_experts * 2 * intermediate, hidden // 64),
            alignment=16,
        ),
        matrix(u8, shared_shape if dummy_shared else (max_tokens, intermediate // 2)),
        matrix(u8, shared_shape if dummy_shared else (max_tokens, intermediate // 2)),
        matrix(
            i32,
            shared_shape if dummy_shared else (shared_scale_tokens, intermediate // 64),
            alignment=16,
        ),
        matrix(
            u8,
            shared_shape
            if dummy_shared
            else (shared_descriptor_experts * hidden, intermediate // 2),
        ),
        matrix(
            i32,
            shared_shape
            if dummy_shared
            else (shared_descriptor_experts * hidden, intermediate // 64),
            alignment=16,
        ),
        vector(f32, local_experts),
        vector(f32, local_experts),
        vector(f32, local_experts),
        vector(f32, max(shared_experts, 1)),
        vector(f32, max(shared_experts, 1)),
        vector(f32, max(shared_experts, 1)),
        vector(f32, local_experts + shared_experts),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    ]


def _symmetric_slice(
    symmetric: torch.Tensor,
    offset: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    itemsize = torch.empty((), dtype=dtype).element_size()
    size = math.prod(shape) * itemsize
    if offset < 0 or offset + size > symmetric.numel():
        raise RuntimeError(
            f"dw NVFP4 symmetric view {shape} at {offset}+{size} exceeds allocation"
        )
    return symmetric[offset : offset + size].view(dtype).view(*shape)


@dataclass
class _RuntimeState:
    module: Any
    compiled: Any
    symmetric: torch.Tensor
    symmetric_handle: Any
    peer_buffers: tuple[torch.Tensor, ...]
    rank_offsets: tuple[int, ...]
    rank_idx: int
    tensors: dict[str, torch.Tensor]
    output: torch.Tensor
    stats: torch.Tensor


_RUNTIME_STATES: dict[tuple[Any, ...], _RuntimeState] = {}
_LOGGED_PHASES: set[str] = set()


def _get_runtime_state(moe: Any, hidden_states: torch.Tensor) -> _RuntimeState:
    from dw_megakernels import (
        get_mega_moe_kernel,
        get_mega_moe_module,
        shape_from_hf_config,
    )
    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    ep = get_moe_ep_group()
    max_tokens = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    shape = shape_from_hf_config(
        moe.config.to_dict(),
        name=moe.config.model_type,
        num_ranks=ep.world_size,
        max_tokens_per_rank=max_tokens,
    )
    # Some hybrid checkpoints quantize routed experts to NVFP4 but retain the
    # shared expert at FP8/BF16. SGLang then executes that shared MLP separately,
    # so specialize the external kernel to the layer's actual fused count.
    shape = replace(shape, num_shared_experts=int(moe.num_fused_shared_experts))
    shape.validate()
    key = (ep.unique_name, hidden_states.device.index, shape)
    cached = _RUNTIME_STATES.get(key)
    if cached is not None:
        return cached

    import cutlass
    import cutlass.cute as cute
    import torch.distributed as dist
    from torch._C._distributed_c10d import _SymmetricMemory

    module = get_mega_moe_module("nvfp4", shape=shape)
    kernel = get_mega_moe_kernel("nvfp4", shape=shape)
    compiled = None
    # Serial compilation avoids eight concurrent MLIR/ptxas processes fighting
    # over the serving container's CPUs. Every rank still compiles its own FFI
    # callable because generated wrappers are process-local.
    for owner in range(ep.world_size):
        if ep.rank_in_group == owner:
            logger.info("Compiling dw-megakernels NVFP4 on EP rank %d", owner)
            compiled = cute.compile(
                kernel,
                *_fake_arguments(module, cutlass, cute),
                options="--enable-tvm-ffi",
            )
        dist.barrier(group=ep.cpu_group)
    if compiled is None:
        raise RuntimeError("dw NVFP4 rank-serialized compile did not run locally")

    symmetric = _SymmetricMemory.empty_strided_p2p(
        (module.NVFP4_SYMMETRIC_BYTES,),
        [1],
        torch.uint8,
        hidden_states.device,
        ep.cpu_group.group_name,
    )
    handle = _SymmetricMemory.rendezvous(symmetric)
    peer_buffers = tuple(
        handle.get_buffer(peer, [module.NVFP4_SYMMETRIC_BYTES], torch.uint8)
        for peer in range(ep.world_size)
    )
    base = symmetric.data_ptr()
    offsets = [peer.data_ptr() - base for peer in peer_buffers]
    if any(peer.data_ptr() == 0 for peer in peer_buffers):
        raise RuntimeError("dw NVFP4 symmetric rendezvous returned a null peer mapping")
    rank_offsets = tuple(offsets + [0] * (72 - len(offsets)))
    symmetric.zero_()

    hidden = module.HIDDEN
    intermediate = module.INTERMEDIATE
    ring = module.DATA_RING_TOKENS
    scale_ring = module.SCALE_RING_TOKENS
    shared_scale_tokens = module.NVFP4_SHARED_SCALE_TOKENS
    input_tokens = _symmetric_slice(
        symmetric,
        module.NVFP4_INPUT_TOKENS,
        (module.MAX_TOKENS_PER_RANK, hidden // 2),
        torch.uint8,
    )
    tensors = {
        "input_tokens": input_tokens,
        "input_sf": _symmetric_slice(
            symmetric,
            module.NVFP4_INPUT_SF,
            (module.MAX_TOKENS_PER_RANK, hidden // 16),
            torch.uint8,
        ),
        "topk_ids": _symmetric_slice(
            symmetric,
            module.NVFP4_INPUT_TOPK_INDICES,
            (module.MAX_TOKENS_PER_RANK, module.TOPK),
            torch.int64,
        ),
        "topk_weights": _symmetric_slice(
            symmetric,
            module.NVFP4_INPUT_TOPK_WEIGHTS,
            (module.MAX_TOKENS_PER_RANK, module.TOPK),
            torch.float32,
        ),
        "l1_acts": _symmetric_slice(
            symmetric, module.NVFP4_L1_ACTIVATIONS, (ring, hidden // 2), torch.uint8
        ),
        "l1_acts_sf": _symmetric_slice(
            symmetric,
            module.NVFP4_L1_SF,
            (scale_ring, hidden // 64),
            torch.int32,
        ),
        "l1_output": _symmetric_slice(
            symmetric,
            module.NVFP4_L2_ACTIVATIONS,
            (ring, intermediate // 2),
            torch.uint8,
        ),
        "l2_acts": _symmetric_slice(
            symmetric,
            module.NVFP4_L2_ACTIVATIONS,
            (ring, intermediate // 2),
            torch.uint8,
        ),
        "l2_acts_sf": _symmetric_slice(
            symmetric,
            module.NVFP4_L2_SF,
            (scale_ring, intermediate // 64),
            torch.int32,
        ),
    }
    if module.NUM_SHARED_EXPERTS:
        tensors.update(
            {
                "shared_l1_acts": input_tokens,
                "shared_l1_acts_sf": _symmetric_slice(
                    symmetric,
                    module.NVFP4_SHARED_L1_SF,
                    (shared_scale_tokens, hidden // 64),
                    torch.int32,
                ),
                "shared_l1_output": _symmetric_slice(
                    symmetric,
                    module.NVFP4_SHARED_L2_ACTIVATIONS,
                    (module.MAX_TOKENS_PER_RANK, intermediate // 2),
                    torch.uint8,
                ),
                "shared_l2_acts": _symmetric_slice(
                    symmetric,
                    module.NVFP4_SHARED_L2_ACTIVATIONS,
                    (module.MAX_TOKENS_PER_RANK, intermediate // 2),
                    torch.uint8,
                ),
                "shared_l2_acts_sf": _symmetric_slice(
                    symmetric,
                    module.NVFP4_SHARED_L2_SF,
                    (shared_scale_tokens, intermediate // 64),
                    torch.int32,
                ),
            }
        )
    else:
        tensors.update(
            {
                "shared_l1_acts": input_tokens[:1, :1],
                "shared_l1_acts_sf": tensors["l1_acts_sf"][:1, :1],
                "shared_l1_output": tensors["l1_output"][:1, :1],
                "shared_l2_acts": tensors["l2_acts"][:1, :1],
                "shared_l2_acts_sf": tensors["l2_acts_sf"][:1, :1],
            }
        )

    state = _RuntimeState(
        module=module,
        compiled=compiled,
        symmetric=symmetric,
        symmetric_handle=handle,
        peer_buffers=peer_buffers,
        rank_offsets=rank_offsets,
        rank_idx=ep.rank_in_group,
        tensors=tensors,
        # Layers execute serially on the model stream, so one stable output
        # and stats allocation can be reused by every MoE layer. This also
        # keeps all pointers fixed for decode CUDA graph capture without
        # reserving max-token output storage once per layer.
        output=torch.empty(
            (module.MAX_TOKENS_PER_RANK, module.HIDDEN),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        ),
        stats=torch.zeros(
            (module.EXPERTS_PER_RANK,),
            dtype=torch.int32,
            device=hidden_states.device,
        ),
    )
    _RUNTIME_STATES[key] = state
    logger.info(
        "Initialized dw NVFP4 MegaMoE: shape=%s symmetric_bytes=%d",
        shape,
        module.NVFP4_SYMMETRIC_BYTES,
    )
    return state


def run_dw_nvfp4_mega_moe(
    moe: Any,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_tokens: int,
    phase: str = "unknown",
) -> torch.Tensor:
    """Quantize, dispatch, execute, combine, and return one fused NVFP4 MoE."""

    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize

    layer = moe.experts
    if not getattr(layer, "_dw_nvfp4_weights_built", False):
        raise RuntimeError("dw NVFP4 weights were not prepared during model loading")
    if fp4_quantize is None:
        raise RuntimeError("FlashInfer fp4_quantize is required by dw NVFP4 MegaMoE")

    state = _get_runtime_state(moe, hidden_states)
    module = state.module

    if num_tokens:
        x_q, x_sf = fp4_quantize(
            hidden_states,
            layer._dw_nvfp4_input_global_scale,
            sf_vec_size=16,
            is_sf_swizzled_layout=False,
        )
        state.tensors["input_tokens"][:num_tokens].copy_(x_q)
        state.tensors["input_sf"][:num_tokens].copy_(x_sf)

        routed_ids = topk_ids[:, : module.TOPK].to(torch.int64)
        routed_ids = _undo_per_rank_shared_slot_remap(
            routed_ids, module.EXPERTS_PER_RANK, module.NUM_SHARED_EXPERTS
        )
        state.tensors["topk_ids"][:num_tokens].copy_(routed_ids)
        state.tensors["topk_weights"][:num_tokens].copy_(
            topk_weights[:, : module.TOPK].to(torch.float32)
        )

        if module.NUM_SHARED_EXPERTS:
            token = torch.arange(
                num_tokens, dtype=torch.int64, device=hidden_states.device
            )
            block = token // module.BLOCK_M
            idx = token % module.BLOCK_M
            transformed = (
                block * module.NVFP4_SF_BLOCK_M
                + (idx & ~127)
                + (idx & 31) * 4
                + ((idx >> 5) & 3)
            )
            words = (
                x_sf.contiguous()
                .view(torch.int32)
                .reshape(num_tokens, module.HIDDEN // 64)
            )
            storage = state.tensors["shared_l1_acts_sf"].view(torch.int32).flatten()
            columns = torch.arange(
                module.HIDDEN // 64, dtype=torch.int64, device=hidden_states.device
            )
            physical = (
                columns[:, None] * module.NVFP4_SHARED_SCALE_TOKENS
                + transformed[None, :]
            )
            storage[physical] = words.transpose(0, 1)

    state.stats.zero_()
    # The TVM-FFI wrapper owns the current CUDA stream; the compile signature's
    # final fake stream is therefore omitted from the runtime invocation.
    state.compiled(
        state.output,
        state.stats,
        num_tokens,
        state.symmetric,
        state.rank_offsets,
        state.rank_idx,
        state.tensors["l1_acts"],
        state.tensors["l1_acts_sf"],
        layer._dw_nvfp4_l1_weight,
        layer._dw_nvfp4_l1_weight_sf,
        state.tensors["l1_output"],
        state.tensors["l2_acts"],
        state.tensors["l2_acts_sf"],
        layer._dw_nvfp4_l2_weight,
        layer._dw_nvfp4_l2_weight_sf,
        state.tensors["shared_l1_acts"],
        state.tensors["shared_l1_acts_sf"],
        layer._dw_nvfp4_shared_l1_weight,
        layer._dw_nvfp4_shared_l1_weight_sf,
        state.tensors["shared_l1_output"],
        state.tensors["shared_l2_acts"],
        state.tensors["shared_l2_acts_sf"],
        layer._dw_nvfp4_shared_l2_weight,
        layer._dw_nvfp4_shared_l2_weight_sf,
        layer._dw_nvfp4_l1_gate_alpha,
        layer._dw_nvfp4_l1_up_alpha,
        layer._dw_nvfp4_l2_alpha,
        layer._dw_nvfp4_shared_l1_gate_alpha,
        layer._dw_nvfp4_shared_l1_up_alpha,
        layer._dw_nvfp4_shared_l2_alpha,
        layer._dw_nvfp4_intermediate_global_scale,
    )
    if phase not in _LOGGED_PHASES:
        logger.info(
            "Executed dw NVFP4 MegaMoE phase=%s num_tokens=%d", phase, num_tokens
        )
        _LOGGED_PHASES.add(phase)
    return state.output[:num_tokens]


__all__ = [
    "build_dw_nvfp4_experts_weights",
    "run_dw_nvfp4_mega_moe",
    "use_dw_nvfp4_mega_moe",
]
