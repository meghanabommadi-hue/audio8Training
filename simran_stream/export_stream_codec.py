"""Export a stateful streaming split of the Audio8 0.1B codec decoder to ONNX.

The bundled ``codec_decoder_fp16.onnx`` is stateless, so streaming has to
re-decode ~128 frames of history for every chunk. The codec is fully causal:

    codes -> embedding sum -> post transformer (8 layers, 128-frame window)
          -> upsample + conv decoder (receptive field ~8 frames) -> audio

so it splits into two graphs:

* ``codec_post_step.onnx``: embedding + post transformer for ``n`` new frames,
  carrying a rotated K/V cache of the last 127 frames per layer (exact).
* ``codec_conv.onnx``: upsample + conv decoder on ``n + CONV_CONTEXT`` frames
  of transformer output; the first ``CONV_CONTEXT`` frames of audio are dropped.

Run with the export venv:
    .venv-export/bin/python export_stream_codec.py
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize

TORCH_MODEL = Path.home() / "models/Edge0-Audio8-TTS-Preview-0.1b"
OUT_DIR = Path(__file__).resolve().parent / "artifacts"
WINDOW = 128
MAX_POSITIONS = 4096
CONV_CONTEXT = 8
HOP = 2048


def load_codec():
    spec = importlib.util.spec_from_file_location("arktts_codec", TORCH_MODEL / "modeling_arktts_codec.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["arktts_codec"] = module
    spec.loader.exec_module(module)
    config = type("Config", (), json.loads((TORCH_MODEL / "config.json").read_text()))
    codec = module.ArkttsCodec(config)
    state = torch.load(TORCH_MODEL / "codec.pth", map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    if any("generator." in key for key in state):
        state = {k.replace("generator.", ""): v for k, v in state.items() if "generator." in k}
    state = {k: v for k, v in state.items() if not k.endswith(("freqs_cis", "causal_mask"))}
    codec.load_state_dict(state, strict=True)
    codec.eval()
    return codec, module


def make_traceable(module) -> None:
    """Replace shape-dependent Python in the decode path with length-independent ops.

    Tracing would otherwise bake the example input length into slices. In the
    decode path every causal conv has stride 1 (no right padding) and residual
    branches keep their length, so these are exact equivalents.
    """

    def conv_forward(self, x):
        assert self.stride == 1, "decode path only uses stride-1 causal convs"
        return self.conv(F.pad(x, (self.padding, 0)))

    def transpose_forward(self, x):
        crop = self.kernel_size - self.stride
        x = self.conv(x)
        return x[..., :-crop] if crop else x

    def residual_forward(self, x):
        return x + self.block(x)

    module.ArkttsCausalConv1d.forward = conv_forward
    module.ArkttsCausalConvTranspose1d.forward = transpose_forward
    module.ArkttsResidualUnit.forward = residual_forward


def remove_weight_norms(module: nn.Module) -> None:
    for sub in module.modules():
        if parametrize.is_parametrized(sub, "weight"):
            parametrize.remove_parametrizations(sub, "weight", leave_parametrized=True)
        elif hasattr(sub, "weight_g"):
            torch.nn.utils.remove_weight_norm(sub)


class PostStep(nn.Module):
    """Codes for ``n`` new frames -> post-transformer output, with a K/V cache."""

    def __init__(self, codec, module):
        super().__init__()
        q = codec.quantizer
        post = q.post_module
        assert isinstance(post.input_proj, nn.Identity) and isinstance(post.output_proj, nn.Identity)
        assert post.window_size == WINDOW and post.causal

        def table(vq):
            # out_proj(codebook[i]) for every entry: a [K, 1024] lookup table.
            weight = vq.out_proj.weight.squeeze(-1)  # [1024, 8]
            return vq.codebook.weight @ weight.T + vq.out_proj.bias

        with torch.no_grad():
            self.register_buffer("semantic_table", table(q.semantic_quantizer.quantizers[0]).clone())
            self.register_buffer("residual_tables", torch.stack([table(v) for v in q.quantizer.quantizers]).clone())
            # Same bf16-rounded rotary values as the reference decode, by absolute position.
            rope = module._rope(MAX_POSITIONS, post.head_dim, post.rope_base).float()
            self.register_buffer("rope", rope)  # [P, head_dim/2, 2]
        self.layers = post.layers
        self.norm = post.norm
        attn = self.layers[0].attention
        self.n_head, self.n_kv, self.head_dim = attn.n_head, attn.n_local_heads, attn.head_dim
        self.semantic_size = self.semantic_table.shape[0]
        self.residual_size = self.residual_tables.shape[1]

    @staticmethod
    def _rotate(x, values):
        # x: [B, n, H, D]; values: [n, D/2, 2]
        shaped = x.reshape(*x.shape[:-1], -1, 2)
        cos = values[None, :, None, :, 0]
        sin = values[None, :, None, :, 1]
        real = shaped[..., 0] * cos - shaped[..., 1] * sin
        imag = shaped[..., 1] * cos + shaped[..., 0] * sin
        return torch.stack((real, imag), dim=-1).flatten(3)

    def forward(self, codes, start, cache_k, cache_v):
        # codes [1, 10, n] int64; start [1] int64; cache_* [L, 1, n_kv, W-1, D]
        n = codes.shape[2]
        sem = codes[0, 0].clamp(0, self.semantic_size - 1)
        res = codes[0, 1:].clamp(0, self.residual_size - 1)  # [9, n]
        x = self.semantic_table[sem]
        books = torch.arange(res.shape[0], device=codes.device)[:, None]
        x = x + self.residual_tables[books, res].sum(0)
        x = x[None]  # [1, n, 1024]

        q_pos = start + torch.arange(n, device=codes.device)
        k_pos = torch.cat([start - (WINDOW - 1) + torch.arange(WINDOW - 1, device=codes.device), q_pos])
        allowed = (k_pos[None, :] >= 0) & (k_pos[None, :] <= q_pos[:, None]) & (k_pos[None, :] >= q_pos[:, None] - WINDOW + 1)
        bias = torch.zeros(allowed.shape, dtype=x.dtype).masked_fill(~allowed, float("-inf"))[None, None]
        values = self.rope[q_pos]
        repeat = self.n_head // self.n_kv
        scale = 1.0 / float(self.head_dim) ** 0.5

        new_k, new_v = [], []
        for index, layer in enumerate(self.layers):
            attn = layer.attention
            hidden = layer.attention_norm(x)
            query, key, value = attn.wqkv(hidden).split(
                (self.n_head * self.head_dim, self.n_kv * self.head_dim, self.n_kv * self.head_dim), dim=-1
            )
            query = self._rotate(query.view(1, n, self.n_head, self.head_dim), values).transpose(1, 2)
            key = self._rotate(key.view(1, n, self.n_kv, self.head_dim), values).transpose(1, 2)
            value = value.view(1, n, self.n_kv, self.head_dim).transpose(1, 2)
            keys = torch.cat([cache_k[index], key], dim=2)
            vals = torch.cat([cache_v[index], value], dim=2)
            new_k.append(keys[:, :, -(WINDOW - 1):])
            new_v.append(vals[:, :, -(WINDOW - 1):])
            keys = keys.repeat_interleave(repeat, dim=1)
            vals = vals.repeat_interleave(repeat, dim=1)
            weights = torch.softmax(query @ keys.transpose(-1, -2) * scale + bias, dim=-1)
            out = (weights @ vals).transpose(1, 2).reshape(1, n, self.n_head * self.head_dim)
            x = x + attn.wo(out) * layer.attention_layer_scale.gamma
            x = x + layer.feed_forward(layer.ffn_norm(x)) * layer.ffn_layer_scale.gamma
        x = self.norm(x)
        return x.transpose(1, 2), torch.stack(new_k), torch.stack(new_v)


class ConvDecoder(nn.Module):
    """Post-transformer output [1, 1024, m] -> audio [1, m * 2048]."""

    def __init__(self, codec):
        super().__init__()
        self.upsample = codec.quantizer.upsample
        self.decoder = codec.decoder

    def forward(self, post):
        return self.decoder(self.upsample(post))[:, 0]


class StreamingDecoder:
    """Reference (torch) streaming loop, mirrored by the ONNX runtime wrapper."""

    def __init__(self, step: PostStep, conv: ConvDecoder):
        self.step, self.conv = step, conv
        layers = len(step.layers)
        shape = (layers, 1, step.n_kv, WINDOW - 1, step.head_dim)
        self.cache_k, self.cache_v = torch.zeros(shape), torch.zeros(shape)
        self.position = 0
        self.history = torch.zeros(1, 1024, 0)

    def push(self, codes):
        post, self.cache_k, self.cache_v = self.step(codes, torch.tensor([self.position]), self.cache_k, self.cache_v)
        self.position += codes.shape[2]
        window = torch.cat([self.history, post], dim=2)
        context = window.shape[2] - post.shape[2]
        audio = self.conv(window)[0, context * HOP:]
        self.history = window[:, :, -CONV_CONTEXT:]
        return audio


def main():
    torch.set_num_threads(8)
    codec, module = load_codec()
    remove_weight_norms(codec)
    step, conv = PostStep(codec, module).eval(), ConvDecoder(codec).eval()
    codes = torch.from_numpy(np.load(sys.argv[1] if len(sys.argv) > 1 else OUT_DIR / "test_codes.npy")).long()[None]

    with torch.inference_mode():
        full = codec.decode(codes)[0, 0]  # original (unpatched) decode as reference
        make_traceable(module)
        for chunks in ([4, 12], [1, 3, 5], [24]):
            stream = StreamingDecoder(step, conv)
            pieces, offset, i = [], 0, 0
            while offset < codes.shape[2]:
                size = chunks[min(i, len(chunks) - 1)]
                pieces.append(stream.push(codes[:, :, offset : offset + size]))
                offset += size
                i += 1
            audio = torch.cat(pieces)
            print(f"chunks {chunks}: len ok={audio.numel() == full.numel()} max_abs_diff={float((audio - full).abs().max()):.2e}")

    OUT_DIR.mkdir(exist_ok=True)
    layers = len(step.layers)
    cache = torch.zeros(layers, 1, step.n_kv, WINDOW - 1, step.head_dim)
    sample = codes[:, :, :12]
    torch.onnx.export(
        step, (sample, torch.tensor([0]), cache, cache), OUT_DIR / "codec_post_step.onnx",
        input_names=["codes", "start", "cache_k", "cache_v"],
        output_names=["post", "next_cache_k", "next_cache_v"],
        dynamic_axes={"codes": {2: "n"}, "post": {2: "n"}},
        opset_version=17, dynamo=False,
    )
    torch.onnx.export(
        conv, (torch.randn(1, 1024, 20),), OUT_DIR / "codec_conv.onnx",
        input_names=["post"], output_names=["audio"],
        dynamic_axes={"post": {2: "m"}, "audio": {1: "samples"}},
        opset_version=17, dynamo=False,
    )
    meta = {"window": WINDOW, "conv_context": CONV_CONTEXT, "hop": HOP, "layers": layers,
            "n_kv": step.n_kv, "head_dim": step.head_dim, "max_positions": MAX_POSITIONS,
            "sample_rate": 44100, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (OUT_DIR / "stream_codec.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("exported to", OUT_DIR)


if __name__ == "__main__":
    main()
