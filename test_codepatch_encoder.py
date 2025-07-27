import torch
from transformers import AutoTokenizer

from modeling_codepatch import CodePatchModel, CodeEncoderConfig


def process_code_for_encoder(code_string: str, tokenizer, config: CodeEncoderConfig, device: str):
    """
    Processes a raw code string into tokenized and patched tensors for the CodePatchModel.
    """
    # Tokenize the entire code string
    inputs = tokenizer(
        code_string,
        return_tensors="pt",
        truncation=True,
        max_length=config.num_patches * config.patch_length, # Ensure we don't exceed max length
    )

    input_ids = inputs["input_ids"].squeeze(0) # Remove batch dimension

    # Calculate the number of full patches we can create
    num_full_patches = len(input_ids) // config.patch_length
    
    # Truncate to only include full patches
    truncated_input_ids = input_ids[:num_full_patches * config.patch_length]

    # Reshape into patches
    patches = truncated_input_ids.view(num_full_patches, config.patch_length)

    # Pad patches if necessary to reach config.num_patches
    num_padding_patches = config.num_patches - num_full_patches
    if num_padding_patches < 0:
        # If we have more patches than configured, truncate them
        patches = patches[:config.num_patches]
        num_padding_patches = 0

    padding = torch.zeros((num_padding_patches, config.patch_length), dtype=torch.long)
    
    # Combine patches and padding
    final_patches = torch.cat((patches, padding), dim=0)

    # Create a corresponding attention mask
    # 1 for real tokens, 0 for padding
    attention_mask = (final_patches != tokenizer.pad_token_id).long()
    
    # Add a batch dimension and send to device
    final_patches = final_patches.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    
    return final_patches, attention_mask, num_full_patches


def main():
    # --- 1. Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the tokenizer for our code encoder
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    # Instantiate our model's configuration
    config = CodeEncoderConfig()

    # Instantiate the CodePatchModel
    model = CodePatchModel(config).to(device).eval()
    print("CodePatchModel loaded successfully.")

    # --- 2. Load and Process Data ---
    matlab_file_path = "../test.m"
    print(f"Loading MATLAB code from: {matlab_file_path}")
    with open(matlab_file_path, "r") as f:
        matlab_code = f.read()

    print("Processing code into patches...")
    input_patches, attention_mask, num_real_patches = process_code_for_encoder(matlab_code, tokenizer, config, device)

    print(f"Shape of input patches tensor: {input_patches.shape}")
    print(f"Shape of attention mask tensor: {attention_mask.shape}")

    # --- 3. Run Inference ---
    print("\nFeeding patches to the model...")
    with torch.no_grad():
        output_vectors = model(input_ids=input_patches, attention_mask=attention_mask)

    # --- 4. Display Results ---
    print("Inference complete.")
    print(f"Shape of the output vectors: {output_vectors.shape}")
    print(f"Number of actual code patches created from the file: {num_real_patches}")

    print("\n--- Summary vectors for each non-padding patch ---")
    for i in range(num_real_patches):
        patch_vector = output_vectors[0, i]
        # Printing the first 16 dimensions for brevity
        print(f"Patch {i+1:03d} summary vector (first 16 dims): {patch_vector[:16].tolist()}")

    if num_real_patches < config.num_patches:
        print(f"\nPatches {num_real_patches + 1} through {config.num_patches} are padding and were not shown.")


if __name__ == "__main__":
    main() 