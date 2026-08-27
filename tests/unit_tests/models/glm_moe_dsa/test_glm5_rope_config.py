from types import SimpleNamespace

import pytest

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.glm_moe_dsa.glm5_bridge import GLM5Bridge


pytestmark = pytest.mark.unit


def test_provider_recovers_glm52_rope_width_from_qk_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        first_k_dense_replace=3,
        num_hidden_layers=78,
        moe_intermediate_size=2048,
        n_shared_experts=1,
        qk_head_dim=256,
        qk_nope_head_dim=192,
        qk_rope_head_dim=192,
        rope_parameters={"rope_theta": 1_000_000},
        index_head_dim=128,
        index_n_heads=32,
        index_topk=2048,
    )
    monkeypatch.setattr(
        MegatronModelBridge,
        "provider_bridge",
        lambda _self, _hf_pretrained: SimpleNamespace(),
    )

    provider = GLM5Bridge().provider_bridge(SimpleNamespace(config=config))

    assert provider.qk_pos_emb_head_dim == 64
