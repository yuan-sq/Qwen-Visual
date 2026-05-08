import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import Optional, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers import SiglipImageProcessor, SiglipVisionModel
from transformers.models.qwen2 import Qwen2Config, Qwen2Model
from transformers.modeling_outputs import CausalLMOutputWithPast

warnings.filterwarnings('ignore')


class VLMConfig(PretrainedConfig):
    model_type = "qwen2_vl"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Qwen2.5-0.5B-Instruct architecture
        self.hidden_size = kwargs.get("hidden_size", 896)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 24)
        self.num_attention_heads = kwargs.get("num_attention_heads", 14)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 2)
        self.intermediate_size = kwargs.get("intermediate_size", 4864)
        self.vocab_size = kwargs.get("vocab_size", 151936)
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1000000.0)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.bos_token_id = kwargs.get("bos_token_id", 151643)
        self.eos_token_id = kwargs.get("eos_token_id", 151645)
        self.pad_token_id = kwargs.get("pad_token_id", self.eos_token_id)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.dropout = kwargs.get("dropout", 0.0)
        # Vision config
        self.image_special_token = kwargs.get("image_special_token", "<|image_pad|>")
        self.image_ids = kwargs.get("image_ids", [151652])
        self.image_hidden_size = kwargs.get("image_hidden_size", 768)
        self.image_token_len = kwargs.get("image_token_len", 196)


class MMVisionProjector(nn.Module):
    def __init__(self, in_dim=768, out_dim=896):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class QwenVL(PreTrainedModel, GenerationMixin):
    config_class = VLMConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: VLMConfig = None, qwen_model=None, vision_model_path=None):
        self.config = config or VLMConfig()
        super().__init__(self.config)

        # Language model backbone — use Qwen2Model (the decoder without lm_head)
        if qwen_model is not None:
            # Extract Qwen2Model from a Qwen2ForCausalLM
            self.model = qwen_model.model if hasattr(qwen_model, 'model') else qwen_model
            # Reuse the pretrained lm_head (already tied to embed_tokens)
            self.lm_head = qwen_model.lm_head
        else:
            qwen_config = Qwen2Config(
                hidden_size=self.config.hidden_size,
                num_hidden_layers=self.config.num_hidden_layers,
                num_attention_heads=self.config.num_attention_heads,
                num_key_value_heads=self.config.num_key_value_heads,
                intermediate_size=self.config.intermediate_size,
                vocab_size=self.config.vocab_size,
                hidden_act=self.config.hidden_act,
                max_position_embeddings=self.config.max_position_embeddings,
                rms_norm_eps=self.config.rms_norm_eps,
                rope_theta=self.config.rope_theta,
                tie_word_embeddings=getattr(self.config, 'tie_word_embeddings', True),
                bos_token_id=self.config.bos_token_id,
                eos_token_id=self.config.eos_token_id,
                pad_token_id=getattr(self.config, 'pad_token_id', self.config.eos_token_id),
                head_dim=getattr(self.config, 'head_dim', self.config.hidden_size // self.config.num_attention_heads),
                attention_dropout=0.0,
                initializer_range=0.02,
                use_cache=True,
                sliding_window=4096,
                use_sliding_window=False,
                max_window_layers=21,
            )
            self.model = Qwen2Model(qwen_config)
            self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
            if self.config.tie_word_embeddings:
                self.model.embed_tokens.weight = self.lm_head.weight

        # Vision encoder
        self.vision_encoder, self.processor = self.__class__.get_vision_model(vision_model_path)

        # Vision projector: LayerNorm(768) -> Linear(768,896) -> GELU -> Linear(896,896)
        self.vision_proj = MMVisionProjector(
            in_dim=self.config.image_hidden_size,
            out_dim=self.config.hidden_size,
        )

    @staticmethod
    def get_vision_model(model_path: str):
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        if not os.path.exists(model_path):
            return None, None
        try:
            model = SiglipVisionModel.from_pretrained(model_path)
        except (RuntimeError, ValueError):
            return None, None
        processor = SiglipImageProcessor.from_pretrained(model_path)
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processor

    @staticmethod
    def image2tensor(image, processor):
        if image.mode in ['RGBA', 'LA']:
            image = image.convert('RGB')
        return processor(images=image, return_tensors="pt")

    @staticmethod
    def get_image_embeddings(image_inputs, vision_model):
        if hasattr(image_inputs, 'keys'):
            image_inputs = {k: v.squeeze(1) if v.ndim > 2 and v.shape[1] == 1 else v for k, v in image_inputs.items()}
        with torch.no_grad():
            outputs = vision_model(**image_inputs)
        return outputs.last_hidden_state  # (B, 196, 768)

    def _get_embed_tokens(self):
        """Get embed_tokens, unwrapping PeftModel if needed."""
        model = self.model
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
            model = model.base_model.model
        return model.embed_tokens

    @torch.compiler.disable
    def inject_vision_features(self, hidden_states, input_ids, vision_features):
        """Replace placeholder token embeddings with projected vision features.
        Maintains gradient flow from vision_features back to the projector."""
        marker = self.config.image_ids[0]
        batch_size, seq_len, hidden_dim = hidden_states.shape
        marker_mask = (input_ids == marker)

        if not marker_mask.any():
            return hidden_states

        outputs = []
        for b in range(batch_size):
            mask = marker_mask[b]
            if not mask.any():
                outputs.append(hidden_states[b])
                continue

            # Build sequence by concatenating text segments (detached) and
            # vision segments (with gradient)
            segments = []
            vf_available = vision_features.shape[1]
            vf_idx, i = 0, 0
            while i < seq_len and vf_idx < vf_available:
                if mask[i]:
                    start = i
                    while i < seq_len and mask[i]:
                        i += 1
                    n = min(i - start, vf_available - vf_idx)
                    segments.append(vision_features[b, vf_idx:vf_idx + n])
                    vf_idx += n
                else:
                    start = i
                    while i < seq_len and not mask[i]:
                        i += 1
                    segments.append(hidden_states[b, start:i].detach())

            # Remaining text after vision features run out
            if i < seq_len:
                segments.append(hidden_states[b, i:].detach())

            out = torch.cat(segments, dim=0)
            # Pad or truncate to maintain sequence length
            if out.shape[0] < seq_len:
                pad = hidden_states.new_zeros(seq_len - out.shape[0], hidden_dim)
                out = torch.cat([out, pad], dim=0)
            elif out.shape[0] > seq_len:
                out = out[:seq_len]
            outputs.append(out)

        return torch.stack(outputs, dim=0)

    def _get_vision_features(self, pixel_values):
        """Extract and project vision features from pixel_values."""
        if hasattr(pixel_values, 'keys'):
            # Dict format from dataset collate_fn
            pv = pixel_values.get('pixel_values', pixel_values)
            if pv.ndim == 5:
                bs, num, c, h, w = pv.shape
                flat = {'pixel_values': pv.flatten(0, 1)}
                embeds = self.get_image_embeddings(flat, self.vision_encoder)
                embeds = embeds.view(bs, num, -1, self.config.image_hidden_size)
                embeds = embeds[:, 0]  # use first image per sample
            else:
                embeds = self.get_image_embeddings({'pixel_values': pv}, self.vision_encoder)
        else:
            if pixel_values.ndim == 6:
                pixel_values = pixel_values.squeeze(2)
            if pixel_values.ndim == 5:
                embeds = self.get_image_embeddings(
                    {'pixel_values': pixel_values[:, 0]}, self.vision_encoder
                )
            else:
                embeds = self.get_image_embeddings(
                    {'pixel_values': pixel_values}, self.vision_encoder
                )
        return self.vision_proj(embeds)  # (B, 196, hidden_size)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                labels: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.FloatTensor] = None,
                **kwargs):
        # Text embeddings (handles PeftModel wrapping transparently)
        hidden_states = self._get_embed_tokens()(input_ids)

        # Inject vision features on the first forward pass (when input has markers)
        if pixel_values is not None and (input_ids == self.config.image_ids[0]).any():
            vision_features = self._get_vision_features(pixel_values)
            hidden_states = self.inject_vision_features(
                hidden_states, input_ids, vision_features
            )

        # Pass through Qwen2 decoder
        outputs = self.model(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        # LM head
        slice_ = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(outputs.last_hidden_state[:, slice_, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.last_hidden_state,
        )

    def generate(self, *args, num_return_sequences=1, **kwargs):
        if num_return_sequences > 1 and 'pixel_values' in kwargs:
            pv = kwargs['pixel_values']
            if hasattr(pv, 'keys'):
                kwargs['pixel_values'] = {k: v.repeat(num_return_sequences, *([1] * (v.ndim - 1))) for k, v in pv.items()}
            else:
                kwargs['pixel_values'] = pv.repeat(num_return_sequences, *([1] * (pv.ndim - 1)))
        return super().generate(*args, num_return_sequences=num_return_sequences, **kwargs)
