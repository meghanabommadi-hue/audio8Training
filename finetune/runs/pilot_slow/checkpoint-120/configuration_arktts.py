from __future__ import annotations

from transformers import PretrainedConfig


class ArkttsConfig(PretrainedConfig):
    model_type = "arktts"

    def __init__(
        self,
        vocab_size: int = 155776,
        dim: int = 896,
        n_layer: int = 24,
        n_head: int = 14,
        n_local_heads: int = 2,
        head_dim: int = 64,
        intermediate_size: int = 4864,
        max_seq_len: int = 2048,
        rope_base: float = 1_000_000,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        attention_qkv_bias: bool = True,
        attention_qk_norm: bool = False,
        attention_o_bias: bool = False,
        tie_word_embeddings: bool = True,
        codebook_size: int = 4096,
        num_codebooks: int = 10,
        semantic_begin_id: int = 151678,
        semantic_end_id: int = 155773,
        n_fast_layer: int = 4,
        fast_dim: int = 896,
        fast_n_head: int = 14,
        fast_n_local_heads: int = 2,
        fast_head_dim: int = 64,
        fast_intermediate_size: int = 4864,
        fast_attention_qkv_bias: bool = False,
        fast_attention_qk_norm: bool = False,
        fast_attention_o_bias: bool = False,
        norm_fastlayer_input: bool = True,
        initializer_range: float = 0.02,
        use_gradient_checkpointing: bool = False,
        codec_filename: str = "codec.pth",
        codec_sample_rate: int = 44100,
        codec_frame_size: int = 2048,
        codec_post_n_layer: int = 8,
        codec_post_n_head: int = 16,
        codec_post_n_local_heads: int = 8,
        codec_post_intermediate_size: int = 1216,
        ras_window_size: int = 10,
        ras_temperature: float = 1.0,
        ras_top_p: float = 0.9,
        eos_token_id: int = 151645,
        pad_token_id: int = 151643,
        bos_token_id: int = 17,
        slow_backbone: str = "falcon_h1",
        # Falcon-H1 backbone parameters.
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        attention_in_multiplier: float = 1.0,
        attention_out_multiplier: float = 1.0,
        embedding_multiplier: float = 0.10888671875,
        expansion_factor: float = 1.5,
        hidden_act: str = "silu",
        key_multiplier: float = 1.0,
        lm_head_multiplier: float = 0.078125,
        mamba_chunk_size: int = 128,
        mamba_conv_bias: bool = True,
        mamba_d_conv: int = 4,
        mamba_d_head: int = 32,
        mamba_d_ssm: int = 768,
        mamba_d_state: int = 64,
        mamba_expand: int = 2,
        mamba_n_groups: int = 1,
        mamba_n_heads: int = 24,
        mamba_norm_before_gate: bool = False,
        mamba_proj_bias: bool = False,
        mamba_rms_norm: bool = False,
        mamba_use_mlp: bool = True,
        mlp_bias: bool = False,
        mlp_multipliers: list[float] | None = None,
        num_logits_to_keep: int = 1,
        projectors_bias: bool = False,
        ssm_in_multiplier: float = 1.0,
        ssm_multipliers: list[float] | None = None,
        ssm_out_multiplier: float = 1.0,
        time_step_floor: float = 0.0001,
        time_step_max: float = 0.1,
        time_step_min: float = 0.001,
        time_step_rank: str | int = "auto",
        use_cache: bool = True,
        **kwargs,
    ):
        kwargs.pop("model_type", None)
        kwargs.pop("audio_embed_dim", None)
        kwargs.pop("is_reward_model", None)
        super().__init__(
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.hidden_size = self.dim
        self.n_layer = int(n_layer)
        self.num_hidden_layers = self.n_layer
        self.n_head = int(n_head)
        self.num_attention_heads = self.n_head
        self.n_local_heads = int(n_local_heads)
        self.num_key_value_heads = self.n_local_heads
        self.head_dim = int(head_dim)
        self.intermediate_size = int(intermediate_size)
        self.max_seq_len = int(max_seq_len)
        self.max_position_embeddings = self.max_seq_len
        self.rope_base = float(rope_base)
        self.rope_theta = self.rope_base
        self.norm_eps = float(norm_eps)
        self.rms_norm_eps = self.norm_eps
        self.dropout = float(dropout)
        self.attention_qkv_bias = bool(attention_qkv_bias)
        self.attention_qk_norm = bool(attention_qk_norm)
        self.attention_o_bias = bool(attention_o_bias)
        self.codebook_size = int(codebook_size)
        self.num_codebooks = int(num_codebooks)
        self.semantic_begin_id = int(semantic_begin_id)
        self.semantic_end_id = int(semantic_end_id)
        self.n_fast_layer = int(n_fast_layer)
        self.fast_dim = int(fast_dim)
        self.fast_n_head = int(fast_n_head)
        self.fast_n_local_heads = int(fast_n_local_heads)
        self.fast_head_dim = int(fast_head_dim)
        self.fast_intermediate_size = int(fast_intermediate_size)
        self.fast_attention_qkv_bias = bool(fast_attention_qkv_bias)
        self.fast_attention_qk_norm = bool(fast_attention_qk_norm)
        self.fast_attention_o_bias = bool(fast_attention_o_bias)
        self.norm_fastlayer_input = bool(norm_fastlayer_input)
        self.initializer_range = float(initializer_range)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.codec_filename = str(codec_filename)
        self.codec_sample_rate = int(codec_sample_rate)
        self.codec_frame_size = int(codec_frame_size)
        self.codec_post_n_layer = int(codec_post_n_layer)
        self.codec_post_n_head = int(codec_post_n_head)
        self.codec_post_n_local_heads = int(codec_post_n_local_heads)
        self.codec_post_intermediate_size = int(codec_post_intermediate_size)
        self.ras_window_size = int(ras_window_size)
        self.ras_temperature = float(ras_temperature)
        self.ras_top_p = float(ras_top_p)
        self.slow_backbone = str(slow_backbone)
        self.attention_bias = bool(attention_bias)
        self.attention_dropout = float(attention_dropout)
        self.attention_in_multiplier = float(attention_in_multiplier)
        self.attention_out_multiplier = float(attention_out_multiplier)
        self.embedding_multiplier = float(embedding_multiplier)
        self.expansion_factor = float(expansion_factor)
        self.hidden_act = str(hidden_act)
        self.key_multiplier = float(key_multiplier)
        self.lm_head_multiplier = float(lm_head_multiplier)
        self.mamba_chunk_size = int(mamba_chunk_size)
        self.mamba_conv_bias = bool(mamba_conv_bias)
        self.mamba_d_conv = int(mamba_d_conv)
        self.mamba_d_head = int(mamba_d_head)
        self.mamba_d_ssm = int(mamba_d_ssm)
        self.mamba_d_state = int(mamba_d_state)
        self.mamba_expand = int(mamba_expand)
        self.mamba_n_groups = int(mamba_n_groups)
        self.mamba_n_heads = int(mamba_n_heads)
        self.mamba_norm_before_gate = bool(mamba_norm_before_gate)
        self.mamba_proj_bias = bool(mamba_proj_bias)
        self.mamba_rms_norm = bool(mamba_rms_norm)
        self.mamba_use_mlp = bool(mamba_use_mlp)
        self.mlp_bias = bool(mlp_bias)
        self.mlp_multipliers = list(mlp_multipliers) if mlp_multipliers is not None else [1.0, 1.0]
        self.num_logits_to_keep = int(num_logits_to_keep)
        self.projectors_bias = bool(projectors_bias)
        self.ssm_in_multiplier = float(ssm_in_multiplier)
        self.ssm_multipliers = list(ssm_multipliers) if ssm_multipliers is not None else [1.0] * 5
        self.ssm_out_multiplier = float(ssm_out_multiplier)
        self.time_step_floor = float(time_step_floor)
        self.time_step_max = float(time_step_max)
        self.time_step_min = float(time_step_min)
        self.time_step_rank = time_step_rank
        self.use_cache = bool(use_cache)


__all__ = ["ArkttsConfig"]
