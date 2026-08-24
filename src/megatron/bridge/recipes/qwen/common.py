# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared building blocks for convergence-oriented Qwen recipes."""

from megatron.bridge.recipes.qwen.qwen3_moe import qwen3_235b_a22b_pretrain_config
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_with_mxfp8_mixed


def _enable_hybridep_full_iteration_mxfp8(cfg: ConfigContainer) -> None:
    """Enable the measured HybridEP MXFP8 graph and overlap configuration.

    These settings are intended to preserve training math, but full-iteration
    graphs and reduced-precision collectives still require convergence
    verification with natural expert routing before becoming a default.
    """
    cfg.model.cuda_graph_impl = "full_iteration"
    cfg.model.cuda_graph_scope = []
    cfg.rng.te_rng_tracker = True
    cfg.model.use_te_rng_tracker = True

    cfg.model.offload_modules = []
    cfg.model.moe_pad_experts_for_cuda_graph_inference = True
    cfg.model.moe_paged_stash = True
    cfg.model.moe_expert_rank_capacity_factor = 1.5
    cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.2
    cfg.model.moe_paged_stash_buffer_size_factor_cpu = 1.0

    cfg.model.moe_shared_expert_overlap = False
    cfg.model.high_priority_a2a_comm_stream = True
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.moe_mlp_glu_interleave_size = 32
    cfg.model.moe_hybridep_num_sms_preprocessing = 32

    cfg.mixed_precision.fp8_dot_product_attention = True
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=True,
        overlap_moe_expert_parallel_comm=True,
        delay_wgrad_compute=True,
    )


def _qwen3_235b_a22b_pretrain_256gpu_blackwell_fp8mx_config(
    *, pipeline_parallel_size: int, virtual_pipeline_parallel_size: int, micro_batch_size: int
) -> ConfigContainer:
    """Build the shared natural-routing 256-GPU Blackwell MXFP8 candidate."""
    cfg = qwen3_235b_a22b_pretrain_config()

    # Precision and kernels copied from the measured flat recipes. Reduced-
    # precision gradient reduction remains an explicit convergence gate.
    cfg.mixed_precision = bf16_with_mxfp8_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.bias_activation_fusion = True
    cfg.model.apply_rope_fusion = True
    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096

    # Measured 256-GPU topology. The GB200 and GB300 recipes supply their
    # hardware-specific PP/VPP/MBS values.
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = pipeline_parallel_size
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = virtual_pipeline_parallel_size
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 8192
    cfg.train.micro_batch_size = micro_batch_size
    cfg.scheduler.lr_warmup_iters = 40
    cfg.scheduler.lr_decay_iters = 100
    cfg.checkpoint.save_interval = 100

    # Natural routing is the defining difference from a throughput benchmark.
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_hybridep_num_sms = 32
    _enable_hybridep_full_iteration_mxfp8(cfg)

    # Retain convergence guards while keeping validation outside the bounded
    # mock-data verification window.
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.check_for_large_grads = True
    cfg.rerun_state_machine.check_for_nan_in_loss = True
    cfg.validation.eval_iters = 0
    cfg.validation.eval_interval = 0
    cfg.logger.log_interval = 1
    return cfg


__all__ = [
    "_enable_hybridep_full_iteration_mxfp8",
    "_qwen3_235b_a22b_pretrain_256gpu_blackwell_fp8mx_config",
]
