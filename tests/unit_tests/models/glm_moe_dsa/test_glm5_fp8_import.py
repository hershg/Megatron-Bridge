import torch

from megatron.bridge.models.glm_moe_dsa.glm5_bridge import GLM5Bridge


def test_fp8_checkpoint_weight_is_dequantized_with_its_block_scale() -> None:
    weight_name = "model.layers.0.self_attn.q_a_proj.weight"
    weight = torch.tensor([[1.0, -2.0], [4.0, -8.0]], dtype=torch.float8_e4m3fn)
    scale_inv = torch.tensor([[0.25]], dtype=torch.float32)

    converted = GLM5Bridge().maybe_modify_loaded_hf_weight(
        weight_name,
        {weight_name: weight, weight_name + "_scale_inv": scale_inv},
    )

    assert converted.dtype == torch.bfloat16
    torch.testing.assert_close(converted, weight.to(torch.bfloat16) * scale_inv.to(torch.bfloat16))


def test_compound_fp8_checkpoint_weights_use_their_own_scales() -> None:
    names = {"gate": "gate.weight", "up": "up.weight"}
    state_dict = {
        "gate.weight": torch.tensor([[2.0]], dtype=torch.float8_e4m3fn),
        "gate.weight_scale_inv": torch.tensor([[0.5]], dtype=torch.float32),
        "up.weight": torch.tensor([[4.0]], dtype=torch.float8_e4m3fn),
        "up.weight_scale_inv": torch.tensor([[0.25]], dtype=torch.float32),
    }

    converted = GLM5Bridge().maybe_modify_loaded_hf_weight(names, state_dict)

    torch.testing.assert_close(converted["gate"], torch.tensor([[1.0]], dtype=torch.bfloat16))
    torch.testing.assert_close(converted["up"], torch.tensor([[1.0]], dtype=torch.bfloat16))
