"""Bounded KV attention and optional CUDA graphs for VoxCPM2 2.0.3."""

import importlib.util
import logging
from types import MethodType

import torch
from voxcpm.modules.minicpm4.model import apply_rotary_pos_emb

log = logging.getLogger(__name__)


def attention_step(self, hidden_states, position_emb, position_id, kv_cache):
    batch = hidden_states.size(0)
    query = self.q_proj(hidden_states).view(batch, 1, self.num_heads, self.head_dim).transpose(1, 2)
    key = self.k_proj(hidden_states).view(batch, 1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value = self.v_proj(hidden_states).view(batch, 1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    if position_emb is not None:
        query, key = apply_rotary_pos_emb(query, key, *position_emb)
    key_cache, value_cache = kv_cache
    # index_copy_ accepts a device position without a host synchronization.
    key_cache.index_copy_(2, position_id, key)
    value_cache.index_copy_(2, position_id, value)
    mask = (torch.arange(key_cache.size(2), device=query.device) <= position_id).view(1, 1, 1, -1)
    output = torch.nn.functional.scaled_dot_product_attention(
        query, key_cache, value_cache, attn_mask=mask, enable_gqa=True,
    )
    return self.o_proj(output.transpose(1, 2).reshape(batch, self.num_heads * self.head_dim))


class BoundedDecode:
    """One serial GPU worker owns this object and the model's static KV cache.

    Graphs cover 256-position buckets up to 2048 positions by default. Longer
    contexts use eager attention on exactly the used prefix, bounding graph
    memory independently of request length. Each graph owns its memory pool.
    """

    def __init__(self, model, use_graph=True, bucket_size=256, graph_limit=2048):
        if bucket_size < 1 or graph_limit < 0:
            raise ValueError('Invalid CUDA graph bucket/limit')
        self.model = model
        self.use_graph = use_graph
        self.bucket_size = bucket_size
        self.graph_limit = graph_limit
        self.graphs = {}
        self.cache = model.kv_cache
        for layer in model.layers:
            layer.self_attn.forward_step = MethodType(attention_step, layer.self_attn)

    def run(self, inputs, position, length):
        model = self.model
        rope = model.rope_emb(position) if model.rope_emb is not None else None
        hidden = inputs
        for index, layer in enumerate(model.layers):
            key, value = model.kv_cache.get_layer_cache(index)
            hidden = layer.forward_step(hidden, rope, position, (key[:, :, :length], value[:, :, :length]))
        return model.norm(hidden)

    def __call__(self, inputs, position):
        cache = self.model.kv_cache
        if cache is not self.cache:
            # Captured kernels retain addresses of the old cache.
            self.graphs.clear()
            self.cache = cache
        length = cache.current_length  # step() increments this before forward_step.
        if not 0 < length <= cache.max_length:
            raise ValueError('KV cache length must include the current decode position')
        if not self.use_graph or not inputs.is_cuda or length > self.graph_limit:
            return self.run(inputs, position, length)
        bucket = min(((length + self.bucket_size - 1) // self.bucket_size) * self.bucket_size, cache.max_length)
        graph_key = (bucket, inputs.shape, inputs.dtype, inputs.device)
        entry = self.graphs.get(graph_key)
        if entry is None:
            try:
                static_input, static_position = inputs.clone(), position.clone()
                stream = torch.cuda.Stream(device=inputs.device)
                stream.wait_stream(torch.cuda.current_stream(inputs.device))
                with torch.cuda.stream(stream):
                    self.run(static_input, static_position, bucket)
                torch.cuda.current_stream(inputs.device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                # Warmup/capture/replay overwrite only this step's KV slot with
                # the same deterministic values. Earlier positions are untouched.
                with torch.cuda.graph(graph, stream=stream):
                    output = self.run(static_input, static_position, bucket)
                entry = (graph, static_input, static_position, output)
                self.graphs[graph_key] = entry
            except RuntimeError:
                log.exception('CUDA graph capture failed; using bounded eager attention')
                self.use_graph = False
                self.graphs.clear()
                return self.run(inputs, position, length)
        graph, static_input, static_position, output = entry
        static_input.copy_(inputs)
        static_position.copy_(position)
        graph.replay()
        # The caller may keep an output across subsequent replays.
        return output.clone()


def optimize_model(model, use_graph=True, compile_features=True):
    for lm in (model.base_lm, model.residual_lm):
        lm.forward_step = BoundedDecode(lm, use_graph=use_graph)
    if compile_features and model.device == 'cuda' and importlib.util.find_spec('triton'):
        model._feat_encoder_raw = model.feat_encoder
        model.feat_encoder = torch.compile(model.feat_encoder, mode='reduce-overhead', fullgraph=True)
        model.feat_decoder.estimator = torch.compile(
            model.feat_decoder.estimator, mode='reduce-overhead', fullgraph=True,
        )
    return model
