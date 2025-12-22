import torch
from mcoplib.op import scale_dynamic_quant

def scale_dynamic_quant_cpu(hidden_states, smooth_scale):
    bs, sl, hd = hidden_states.shape
    hidden_states_flat = hidden_states.view(bs * sl, hd).float()
    quant_ref = torch.zeros_like(hidden_states_flat, dtype=torch.int8)
    scale_ref = torch.zeros(bs * sl, device="cuda", dtype=torch.float32)
    
    for token_idx in range(bs * sl):
        token_data = hidden_states_flat[token_idx] * smooth_scale

        token_max = torch.abs(token_data).max().item()
        scale = token_max / 127.0 if token_max > 0 else 1.0
        scale_ref[token_idx] = scale

        quantized = torch.round(token_data / scale).clamp(-128, 127).to(torch.int8)
        quant_ref[token_idx] = quantized
    
    return quant_ref.view(bs, sl, hd), scale_ref

def do_test(test_name, batch_size, seq_len, hidden_dim, do_verify, need_random=True, manual_hidden=None, manual_scales=None):
    def make_hidden_states_smooth_scales(batch_size, seq_len, hidden_dim, need_random):
        if need_random:
            torch.manual_seed(13)
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16)
        smooth_scales = torch.ones(hidden_dim, device="cuda", dtype=torch.float32)
        return hidden_states, smooth_scales
    if manual_hidden == None and manual_scales == None:
        hidden_states, smooth_scales = make_hidden_states_smooth_scales(batch_size, seq_len, hidden_dim, need_random)
    else:
        hidden_states = manual_hidden
        smooth_scales = manual_scales
    quant_tokens, per_token_scale = scale_dynamic_quant(
        hidden_states,
        smooth_scales,
        torch.int8
    )
    do_verify(test_name, hidden_states, smooth_scales, quant_tokens, per_token_scale)

def test_scale_dynamic_quant():
    def basic_test_verify(test_name, hidden_states, smooth_scales, quant_tokens, per_token_scale):
        assert quant_tokens.shape == (2, 128, 768)
        assert per_token_scale.shape == (2 * 128,)
    do_test(test_name='Basic Test', batch_size=2, seq_len=128, hidden_dim=768, do_verify=basic_test_verify, need_random=False)

    def correct_test_verify(test_name, hidden_states, smooth_scales, quant_tokens, per_token_scale):
        quant_ref, scale_ref = scale_dynamic_quant_cpu(hidden_states, smooth_scales)
        scale_tolerance = 1e-5
        scale_match = torch.allclose(per_token_scale.cpu(), scale_ref.cpu(), rtol=scale_tolerance)
        quant_match = torch.allclose(quant_tokens.cpu(), quant_ref.cpu(), rtol=0, atol=1)
        assert quant_match
        assert scale_match
    do_test(test_name='Correctness Test', batch_size=2, seq_len=128, hidden_dim=768, do_verify=correct_test_verify)

    def specific_values_test_verify(test_name, hidden_states, smooth_scales, quant_tokens, per_token_scale):
        assert abs(per_token_scale[0].item() - (8.0 / 127)) < 1e-5
        assert abs(per_token_scale[1].item() - (4.0 / 127)) < 1e-5
    do_test(test_name='Specific value Test', batch_size=None, seq_len=None, hidden_dim=None, do_verify=specific_values_test_verify, need_random=False, manual_hidden=torch.tensor([[[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]], device="cuda", dtype=torch.bfloat16),
                                                                                                                                              manual_scales=torch.tensor([1.0, 1.0, 2.0, 2.0], device="cuda", dtype=torch.float32))

    hidden_states = torch.zeros(1, 4, 8, device="cuda", dtype=torch.bfloat16)
    smooth_scales = torch.ones(8, device="cuda", dtype=torch.float32)
    def edge_case_0_test_verify(test_name, hidden_states, smooth_scales, quant_tokens, per_token_scale):
        assert torch.all(quant_tokens == 0)
    do_test(test_name='Edge case 0 Test', batch_size=None, seq_len=None, hidden_dim=None, do_verify=edge_case_0_test_verify, need_random=False, manual_hidden=hidden_states,
                                                                                                                                           manual_scales=smooth_scales)
    hidden_states = torch.ones(1, 4, 8, device="cuda", dtype=torch.bfloat16) * 1000.0
    smooth_scales = torch.ones(8, device="cuda", dtype=torch.float32)
    def edge_case_1_test_verify(test_name, hidden_states, smooth_scales, quant_tokens, per_token_scale):
        assert torch.all(torch.abs(quant_tokens) <= 127)
    do_test(test_name='Edge case 1 Test', batch_size=None, seq_len=None, hidden_dim=None, do_verify=edge_case_1_test_verify, need_random=False, manual_hidden=hidden_states,
                                                                                                                                           manual_scales=smooth_scales)
   
    hidden_states = torch.randn(1, 1, 768, device="cuda", dtype=torch.bfloat16)
    smooth_scale = torch.ones(768, device="cuda", dtype=torch.float32)
    def shape_1_1_768_test_verify(test_name, hidden_states, smooth_scales, quant_tokens, per_token_scale):
        assert quant_tokens.shape == (1, 1, 768)
        assert per_token_scale.shape == (1 * 1,)
    do_test(test_name='Shape_1_1_768 Test', batch_size=None, seq_len=None, hidden_dim=None, do_verify=shape_1_1_768_test_verify, need_random=False, manual_hidden=hidden_states,
                                                                                                                                           manual_scales=smooth_scales)
    
    hidden_states = torch.randn(4, 512, 768, device="cuda", dtype=torch.bfloat16)
    smooth_scale = torch.ones(768, device="cuda", dtype=torch.float32)
    def shape_4_512_768_test_verify(test_name, hidden_states, smooth_scales, quant_tokens, per_token_scale):
        assert quant_tokens.shape == (4, 512, 768)
        assert per_token_scale.shape == (4 * 512,)
    do_test(test_name='Shape_4_512_768 Test', batch_size=None, seq_len=None, hidden_dim=None, do_verify=shape_4_512_768_test_verify, need_random=False, manual_hidden=hidden_states,
                                                                                                                                           manual_scales=smooth_scales)
    
    hidden_states = torch.randn(1, 2048, 4096, device="cuda", dtype=torch.bfloat16)
    smooth_scale = torch.ones(4096, device="cuda", dtype=torch.float32)
    def shape_1_2048_4096_test_verify(test_name, hidden_states, smooth_scales, quant_tokens, per_token_scale):
        assert quant_tokens.shape == (1, 2048, 4096)
        assert per_token_scale.shape == (1 * 2048,)
    do_test(test_name='Shape_1_2048_4096 Test', batch_size=None, seq_len=None, hidden_dim=None, do_verify=shape_1_2048_4096_test_verify, need_random=False, manual_hidden=hidden_states,
                                                                                                                                           manual_scales=smooth_scales)
    
    hidden_states = torch.randn(8, 128, 1024, device="cuda", dtype=torch.bfloat16)
    smooth_scale = torch.ones(1024, device="cuda", dtype=torch.float32)
    def shape_8_128_1024_test_verify(test_name, hidden_states, smooth_scales, quant_tokens, per_token_scale):
        assert quant_tokens.shape == (8, 128, 1024)
        assert per_token_scale.shape == (8 * 128,)
    do_test(test_name='Shape_8_128_1024 Test', batch_size=None, seq_len=None, hidden_dim=None, do_verify=shape_8_128_1024_test_verify, need_random=False, manual_hidden=hidden_states,
                                                                                                                                           manual_scales=smooth_scales)
    
    hidden_states = torch.randn(2, 64, 256, device="cuda", dtype=torch.bfloat16) * 1e-7
    smooth_scale = torch.ones(256, device="cuda", dtype=torch.float32)
    def stablity_test_verify(test_name, hidden_states, smooth_scales, quant_tokens, per_token_scale):
        assert torch.all(torch.isfinite(per_token_scale))
        assert torch.all(per_token_scale > 0)
    do_test(test_name='Stability Test', batch_size=None, seq_len=None, hidden_dim=None, do_verify=stablity_test_verify, need_random=False, manual_hidden=hidden_states,
                                                                                                                                           manual_scales=smooth_scales)
if __name__ == "__main__":
    test_scale_dynamic_quant()