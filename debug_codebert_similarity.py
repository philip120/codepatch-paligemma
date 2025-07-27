import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from modeling_codepatch import CodePatchModel, CodeEncoderConfig
from processing_codepatch import CodePatchProcessor

def get_code_embedding(code_string: str, model: CodePatchModel, processor: CodePatchProcessor):
    """
    Processes a code string and returns its summary vector from the CodePatchModel.
    We only take the embedding of the first patch as these are short snippets.
    """
    inputs = processor(code=code_string, text="") # Text part is not needed for this
    
    code_input_ids = inputs["code_input_ids"]
    code_attention_mask = inputs["code_attention_mask"]

    with torch.no_grad():
        patch_embeddings = model(input_ids=code_input_ids, attention_mask=code_attention_mask)
    
    # Return the embedding of the first patch
    return patch_embeddings[0, 0, :]

def main():
    print("--- Debugging CodeBERT's Semantic Understanding ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. Setup Model and Processor ---
    config = CodeEncoderConfig()
    processor = CodePatchProcessor(
        "microsoft/codebert-base", "microsoft/codebert-base", config, device=device
    )
    model = CodePatchModel(config).to(device).eval()

    # --- 2. Define Code Snippets ---
    code_A = "title('Step Response');"       # Similar
    code_B = "title('Impulse Response');"    # Similar
    code_C = "grid on;"                      # Different

    print(f"\nCode A: {code_A}")
    print(f"Code B: {code_B}")
    print(f"Code C: {code_C}")

    # --- 3. Get Embeddings ---
    print("\nGenerating embeddings for each code snippet...")
    vec_A = get_code_embedding(code_A, model, processor)
    vec_B = get_code_embedding(code_B, model, processor)
    vec_C = get_code_embedding(code_C, model, processor)

    # --- 4. Calculate Cosine Similarity ---
    # We add a batch dimension for the similarity function
    sim_AB = F.cosine_similarity(vec_A.unsqueeze(0), vec_B.unsqueeze(0))
    sim_AC = F.cosine_similarity(vec_A.unsqueeze(0), vec_C.unsqueeze(0))

    # --- 5. Print Results ---
    print("\n--- Cosine Similarity Results ---")
    print(f"Similarity between A and B (title vs. title): {sim_AB.item():.4f}")
    print(f"Similarity between A and C (title vs. grid): {sim_AC.item():.4f}")

    print("\nHypothesis Check:")
    if sim_AB > 0.8 and sim_AC < 0.6:
        print("PASSED: Similar code snippets are much closer in embedding space than different ones.")
    else:
        print("FAILED: The embeddings do not seem to capture semantic similarity as expected.")

if __name__ == "__main__":
    main() 