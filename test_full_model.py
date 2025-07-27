import torch
from transformers import AutoTokenizer

from modeling_codepatch import (
    CodePatchConfig,
    CodeEncoderConfig,
    CodePatchForConditionalGeneration,
)
from modeling_gemma import GemmaConfig

# Re-using the processor from the previous test for consistency
def process_code_for_encoder(code_string: str, tokenizer, config: CodeEncoderConfig, device: str):
    inputs = tokenizer(
        code_string,
        return_tensors="pt",
        truncation=True,
        max_length=config.num_patches * config.patch_length,
    )
    input_ids = inputs["input_ids"].squeeze(0)
    num_full_patches = len(input_ids) // config.patch_length
    truncated_input_ids = input_ids[:num_full_patches * config.patch_length]
    patches = truncated_input_ids.view(num_full_patches, config.patch_length)
    num_padding_patches = config.num_patches - num_full_patches
    if num_padding_patches < 0:
        patches = patches[:config.num_patches]
        num_padding_patches = 0
    padding = torch.zeros((num_padding_patches, config.patch_length), dtype=torch.long)
    final_patches = torch.cat((patches, padding), dim=0)
    attention_mask = (final_patches != tokenizer.pad_token_id).long()
    final_patches = final_patches.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    return final_patches, attention_mask


def main():
    # --- 1. Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 2. Configuration ---
    print("Creating model configurations...")
    # Config for the code encoder part
    code_encoder_config = CodeEncoderConfig()

    # A dummy config for the Gemma language model part
    # Using smaller values to keep the test model lightweight
    text_config_dict = {
        "vocab_size": 256000,
        "hidden_size": 512, # Small for testing
        "intermediate_size": 1024,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "max_position_embeddings": 1024,
    }

    # The main config that brings them together
    full_config = CodePatchConfig(
        code_encoder_config=vars(code_encoder_config),
        text_config=text_config_dict,
        projection_dim=text_config_dict["hidden_size"], # Projector must match Gemma's hidden size
    )
    print("Configurations created successfully.")

    # --- 3. Model Instantiation ---
    print("Instantiating the full CodePatchForConditionalGeneration model...")
    model = CodePatchForConditionalGeneration(full_config).to(device).eval()
    print("Model instantiated successfully.")

    # --- 4. Prepare Inputs ---
    print("Preparing code patches and prompt tensors...")
    # Prepare code input
    code_tokenizer = AutoTokenizer.from_pretrained(code_encoder_config.model_name_or_path)
    matlab_file_path = "../test.m"
    with open(matlab_file_path, "r") as f:
        matlab_code = f.read()
    code_input_ids, code_attention_mask = process_code_for_encoder(
        matlab_code, code_tokenizer, code_encoder_config, device
    )

    # Prepare text prompt input
    # For testing purposes, we can reuse the same tokenizer for the prompt.
    # In a real scenario, this would be Gemma's specific tokenizer.
    prompt_tokenizer = code_tokenizer
    prompt_text = "What is the title of this plot?"
    prompt_inputs = prompt_tokenizer(
        prompt_text, return_tensors="pt", padding="max_length", max_length=32
    )
    prompt_input_ids = prompt_inputs["input_ids"].to(device)
    prompt_attention_mask = prompt_inputs["attention_mask"].to(device)

    print(f"Shape of code_input_ids: {code_input_ids.shape}")
    print(f"Shape of prompt_input_ids: {prompt_input_ids.shape}")

    # --- 5. Run Forward Pass ---
    print("\nRunning the forward pass through the integrated model...")
    with torch.no_grad():
        outputs = model(
            code_input_ids=code_input_ids,
            code_attention_mask=code_attention_mask,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
        )

    # --- 6. Check Output ---
    print("Forward pass complete.")
    logits = outputs["logits"]
    print(f"Shape of the final output logits: {logits.shape}")
    
    expected_seq_len = code_input_ids.shape[1] + prompt_input_ids.shape[1]
    print(f"Expected sequence length: {expected_seq_len}")
    print(f"Expected vocabulary size: {text_config_dict['vocab_size']}")

    assert logits.shape[0] == 1
    assert logits.shape[1] == expected_seq_len
    assert logits.shape[2] == text_config_dict['vocab_size']

    print("\nTest passed! The model architecture is sound and data flows correctly.")


if __name__ == "__main__":
    main() 