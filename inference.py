import torch
import argparse

from modeling_codepatch import CodePatchConfig, CodeEncoderConfig, CodePatchForConditionalGeneration
from processing_codepatch import CodePatchProcessor
from transformers import GemmaForCausalLM, GemmaConfig
from ast_parser.ast_utils import get_semantic_patches


def get_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained CodePatch model.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint .pt file.")
    parser.add_argument("--code_file", type=str, required=True, help="Path to the MATLAB code file.")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling.")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Penalty for repeating tokens.")
    return parser.parse_args()

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    gemma_model_name = "google/gemma-2b"

    # --- 1. Re-create the exact model and processor configuration used for training ---
    code_encoder_config = CodeEncoderConfig(freeze_encoder=True)
    processor = CodePatchProcessor(
        "microsoft/codebert-base", gemma_model_name, code_encoder_config, device=device
    )
    # Load the official Gemma config to ensure architectures match
    text_config = GemmaConfig.from_pretrained(gemma_model_name)
    
    full_config = CodePatchConfig(
        code_encoder_config=vars(code_encoder_config), text_config=text_config.to_dict(),
        projection_dim=text_config.hidden_size, freeze_llm=True,
    )

    # --- 2. Instantiate the model and load the trained weights ---
    print("Instantiating model and loading REAL Gemma weights...")
    model = CodePatchForConditionalGeneration(full_config).to(device)
    
    # Load the pre-trained weights for the language model part FIRST
    gemma_model = GemmaForCausalLM.from_pretrained("google/gemma-2b", torch_dtype=torch.bfloat16)
    model.language_model.load_state_dict(gemma_model.state_dict())

    print(f"Loading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.multi_modal_projector.load_state_dict(checkpoint['projector_state_dict'])
    model.code_encoder.position_embedding.load_state_dict(checkpoint['pos_embedding_state_dict'])
    model.eval() # Set model to evaluation mode
    print("Checkpoint loaded successfully.")

    # --- 3. Prepare Inputs ---
    print(f"Loading code from: {args.code_file}")
    with open(args.code_file, "r") as f:
        code_string = f.read()
    
    # Give the model a "kick-start" prompt to encourage it to begin generating a description.
    prompt_text = "This MATLAB script generates a plot of"
    inputs = processor(code=code_string, text=prompt_text)
    
    # --- 4. Generate Text ---
    print("\n--- Generating Description ---")
    print(f"Input Code File: {args.code_file}")
    
    generated_token_ids = []
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            outputs = model(**inputs)
            logits = outputs["logits"]
            
            # Get the logits for the very last token
            next_token_logits = logits[:, -1, :]

            # --- FIX: Clamp the logits to a reasonable range to prevent mode collapse ---
            next_token_logits = torch.clamp(next_token_logits, min=-30, max=30)

            # Apply repetition penalty
            if generated_token_ids:
                # Loop through unique generated tokens and apply the penalty
                for token_id in set(generated_token_ids):
                    if next_token_logits[0, token_id] > 0:
                        next_token_logits[0, token_id] /= args.repetition_penalty
                    else:
                        next_token_logits[0, token_id] *= args.repetition_penalty

            # Apply temperature and top-p sampling
            next_token_logits = next_token_logits / args.temperature
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)

            next_token_id = sample_top_p(probs, args.top_p)
            
            # Stop if the model generates an EOS token
            if next_token_id.item() == processor.text_tokenizer.eos_token_id:
                break
            
            generated_token_ids.append(next_token_id.item())
            
            # Append the new token to the prompt for the next iteration
            inputs["prompt_input_ids"] = torch.cat([inputs["prompt_input_ids"], next_token_id], dim=1)
            inputs["prompt_attention_mask"] = torch.cat(
                [inputs["prompt_attention_mask"], torch.ones_like(next_token_id)], dim=1
            )

    # --- 5. Decode and Print Output ---
    generated_text = processor.text_tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    print(f"Generated Description: {prompt_text} {generated_text}")
    print("\n--- Inference Complete ---")

if __name__ == "__main__":
    main()
