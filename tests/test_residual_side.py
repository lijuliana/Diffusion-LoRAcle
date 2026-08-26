"""klein's module names match none of the FLUX patterns, so the name-based rule picked the
non-residual side for every output-side module and 42% of modules were then silently dropped."""
import torch

from ditloracle.reader.dataset import pick_residual_side

D = 3072
# (name, d_out, d_in) taken from the klein checkpoint
KLEIN = [
    ("double_blocks.0.attn.to_q", 3072, 3072),
    ("double_blocks.0.img_attn.proj", 3072, 3072),
    ("double_blocks.0.img_mlp.0", 18432, 3072),
    ("double_blocks.0.img_mlp.2", 3072, 9216),
    ("double_blocks.0.txt_mlp.2", 3072, 9216),
    ("single_transformer_blocks.0.attn.to_out", 3072, 12288),
    ("single_transformer_blocks.0.attn.to_qkv_mlp_proj", 27648, 3072),
]


def test_every_klein_module_yields_a_residual_width_direction():
    for name, do, di in KLEIN:
        U, V = torch.zeros(do, 4), torch.zeros(di, 4)
        M, side = pick_residual_side(U, V, D, name)
        assert M.shape[0] == D, f"{name}: picked {side} with width {M.shape[0]}, expected {D}"


def test_picks_output_side_for_down_projections():
    U, V = torch.zeros(3072, 4), torch.zeros(9216, 4)
    _, side = pick_residual_side(U, V, D, "double_blocks.0.img_mlp.2")
    assert side == "U"


def test_picks_input_side_for_up_projections():
    U, V = torch.zeros(18432, 4), torch.zeros(3072, 4)
    _, side = pick_residual_side(U, V, D, "double_blocks.0.img_mlp.0")
    assert side == "V"
