import torch
from torch import nn
from transformers import AutoModel

from modeling_gemma import GemmaConfig, GemmaForCausalLM, KVCache

class CodeEncoderConfig:
    def __init__(
        self,
        model_name_or_path: str = "microsoft/codebert-base",
        hidden_size: int = 768,
        num_patches: int = 256,
        patch_length: int = 20,
        freeze_encoder: bool = True,
    ):
        self.model_name_or_path = model_name_or_path
        self.hidden_size = hidden_size
        self.num_patches = num_patches
        self.patch_length = patch_length
        self.freeze_encoder = freeze_encoder


class CodePatchConfig:
    def __init__(
        self,
        code_encoder_config: dict,
        text_config: dict,
        projection_dim: int,
        freeze_llm: bool = True,
        **kwargs,
    ):
        self.code_encoder_config = CodeEncoderConfig(**code_encoder_config)
        self.text_config = GemmaConfig(**text_config)
        self.projection_dim = projection_dim
        self.freeze_llm = freeze_llm


class CodePatchModel(nn.Module):
    def __init__(self, config: CodeEncoderConfig):
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config.model_name_or_path)

        # Add positional embeddings for the patches, similar to SigLIP's vision embeddings
        self.position_embedding = nn.Embedding(config.num_patches, config.hidden_size)
        self.register_buffer("position_ids", torch.arange(config.num_patches).expand((1, -1)), persistent=False)

        if config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, patch_length = input_ids.shape
        hidden_size = self.config.hidden_size

        # Reshape from (batch_size, num_patches, patch_length) to (batch_size * num_patches, patch_length)
        # to process all patches in a single batch.
        input_ids = input_ids.view(-1, patch_length)
        attention_mask = attention_mask.view(-1, patch_length)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the [CLS] token's embedding for each patch.
        # This serves as the summary vector for the patch.
        patch_embeddings = outputs.last_hidden_state[:, 0, :]

        # Reshape back to (batch_size, num_patches, hidden_size)
        patch_embeddings = patch_embeddings.view(batch_size, num_patches, hidden_size)

        # Add the learned positional embeddings to the patch embeddings
        positional_embeddings = self.position_embedding(self.position_ids)
        patch_embeddings = patch_embeddings + positional_embeddings

        return patch_embeddings


class CodePatchMultiModalProjector(nn.Module):
    def __init__(self, config: CodePatchConfig):
        super().__init__()
        self.linear = nn.Linear(
            config.code_encoder_config.hidden_size, config.projection_dim, bias=True
        )

    def forward(self, code_features: torch.Tensor) -> torch.Tensor:
        projected_features = self.linear(code_features)
        return projected_features


class CodePatchForConditionalGeneration(nn.Module):
    def __init__(self, config: CodePatchConfig):
        super().__init__()
        self.config = config
        self.code_encoder = CodePatchModel(config.code_encoder_config)
        self.multi_modal_projector = CodePatchMultiModalProjector(config)
        self.language_model = GemmaForCausalLM(config.text_config)

        if self.config.freeze_llm:
            for param in self.language_model.parameters():
                param.requires_grad = False

    def get_input_embeddings(self, input_ids):
        return self.language_model.get_input_embeddings()(input_ids)

    def forward(
        self,
        code_input_ids: torch.Tensor,
        code_attention_mask: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        kv_cache: KVCache = None,
        **kwargs, # Accept and ignore the 'labels' from the collate_fn
    ):
        # 1. Get embeddings for the code patches
        code_features = self.code_encoder(
            input_ids=code_input_ids, attention_mask=code_attention_mask
        )

        # 2. Project the code embeddings to match the language model's dimension
        projected_code_features = self.multi_modal_projector(code_features)

        # 3. Get embeddings for the text prompt
        prompt_embeds = self.get_input_embeddings(prompt_input_ids)

        # 4. Merge features and create the correct 2D padding mask for the full sequence
        inputs_embeds = torch.cat([projected_code_features, prompt_embeds], dim=1)
        
        code_padding_mask_2d = torch.ones(
            projected_code_features.shape[:2], dtype=torch.long, device=projected_code_features.device
        )
        padding_mask_2d = torch.cat([code_padding_mask_2d, prompt_attention_mask], dim=1)

        # 5. Create a 4D causal attention mask from the 2D padding mask.
        batch_size, seq_length = padding_mask_2d.shape
        causal_mask = torch.tril(torch.ones((seq_length, seq_length), dtype=torch.bool, device=padding_mask_2d.device))
        attention_mask_4d = causal_mask[None, None, :, :] & padding_mask_2d[:, None, None, :]
        final_attention_mask = torch.zeros_like(attention_mask_4d, dtype=inputs_embeds.dtype)
        final_attention_mask.masked_fill_(~attention_mask_4d, torch.finfo(inputs_embeds.dtype).min)

        # 6. Create position_ids that correctly account for padding
        position_ids = (padding_mask_2d.cumsum(-1) - 1).masked_fill(padding_mask_2d == 0, 1)

        # 7. Feed the combined sequence to the language model
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=final_attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )

        return outputs 