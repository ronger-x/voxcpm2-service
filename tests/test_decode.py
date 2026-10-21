import unittest

import torch
from voxcpm.modules.minicpm4.config import MiniCPM4Config
from voxcpm.modules.minicpm4.model import MiniCPMModel

from inference_accel import BoundedDecode


def tiny_model(device):
    config = MiniCPM4Config(
        bos_token_id=0, eos_token_id=1, hidden_size=32, intermediate_size=64,
        max_position_embeddings=1024, num_attention_heads=4, num_hidden_layers=2,
        num_key_value_heads=2, rms_norm_eps=1e-5, vocab_size=0, scale_emb=1,
        dim_model_base=32, scale_depth=1, rope_theta=10000,
        rope_scaling={'type': 'longrope', 'long_factor': [1.] * 4, 'short_factor': [1.] * 4,
                      'original_max_position_embeddings': 1024},
    )
    model = MiniCPMModel(config).to(device).eval()
    model.setup_cache(1, 1024, device, torch.float32)
    return model


class DecodeTests(unittest.TestCase):
    @torch.inference_mode()
    def compare(self, device, graph):
        torch.manual_seed(11)
        model = tiny_model(device)
        reference = tiny_model(device)
        reference.load_state_dict(model.state_dict())
        decode = BoundedDecode(model, use_graph=graph, bucket_size=256, graph_limit=512)
        # Reusing cached graphs across reset, crossing buckets, exact capacity,
        # and the eager fallback must all preserve the reference computation.
        for position in (0, 1, 254, 255, 256, 257, 511, 512, 1023, 3):
            cache = model.kv_cache.kv_cache.normal_()
            reference.kv_cache.kv_cache.copy_(cache)
            before = cache.clone()
            model.kv_cache.current_length = position + 1
            x = torch.randn(1, 32, device=device)
            pos = torch.tensor([position], device=device)
            expected = reference.forward_step(x, pos)
            actual = decode(x, pos)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(cache[..., :position, :], before[..., :position, :], rtol=0, atol=0)
            torch.testing.assert_close(cache[..., position + 1:, :], before[..., position + 1:, :], rtol=0, atol=0)
        if graph:
            self.assertTrue(decode.use_graph)
            self.assertEqual(len(decode.graphs), 2)
            # A new cache allocation invalidates captured addresses.
            model.setup_cache(1, 1024, device, torch.float32)
            model.kv_cache.current_length = 4
            decode(x, torch.tensor([3], device=device))
            self.assertEqual(len(decode.graphs), 1)

    def test_cpu_matches_full_attention(self):
        self.compare('cpu', False)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
    def test_cuda_graph_matches_full_attention(self):
        self.compare('cuda', True)


if __name__ == '__main__':
    unittest.main()
