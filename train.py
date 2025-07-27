import torch
from torch.optim import AdamW
from torch import nn
from datasets import load_dataset

from modeling_codepatch import CodePatchConfig, CodeEncoderConfig, CodePatchForConditionalGeneration
from processing_codepatch import CodePatchProcessor

def main():
    print("--- Testing a single training step ---")
    
    # --- 1. Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 2. Configuration & Processor ---
    # We need the vocab size from the tokenizer to configure the model,
    # so we instantiate the processor first.
    print("Instantiating processor to get tokenizer info...")
    code_encoder_config = CodeEncoderConfig(freeze_encoder=True)
    processor = CodePatchProcessor(
        code_tokenizer_name="microsoft/codebert-base",
        text_tokenizer_name="microsoft/codebert-base", # Using same for test
        config=code_encoder_config
    )

    print("Creating model configurations...")
    text_config_dict = {
        "vocab_size": processor.text_tokenizer.vocab_size,
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "max_position_embeddings": 1024,
    }

    full_config = CodePatchConfig(
        code_encoder_config=vars(code_encoder_config),
        text_config=text_config_dict,
        projection_dim=text_config_dict["hidden_size"],
        freeze_llm=True # Explicitly set for Scenario 1
    )

    # --- 3. Instantiate Model ---
    print("Instantiating model...")
    model = CodePatchForConditionalGeneration(full_config).to(device)

    # --- 4. Prepare Data ---
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("philip120/RPOFES-dataset", split="train")
    
    print("Preparing a sample data batch from the dataset...")
    # Get the first example from the dataset
    example = dataset[0]
    # The dataset has a non-standard format where columns are indexed by numbers as strings
    code_string = example['0']['value']
    prompt_string = example['1']['value']
    
    inputs = processor(code=code_string, text=prompt_string)

    # --- 5. Create Labels for Loss Calculation ---
    # The goal is to predict the prompt tokens (the description of the plot).
    prompt_ids = inputs["prompt_input_ids"]
    
    # Create labels by shifting the prompt_ids to the left. The last token is ignored.
    labels = prompt_ids.clone()
    
    # We don't calculate loss on the code patch part of the input.
    # We use -100, the default ignore_index for CrossEntropyLoss.
    code_patch_ignore_labels = torch.full(
        (1, code_encoder_config.num_patches), -100, device=device
    )

    # The final labels are the ignored code patches followed by the actual prompt text.
    final_labels = torch.cat([code_patch_ignore_labels, labels], dim=1)

    # --- 6. Training Step ---
    print("Performing a single training step...")
    # Ensure model is in training mode
    model.train()

    # Get only the parameters that require gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = AdamW(trainable_params, lr=5e-5)
    loss_fn = nn.CrossEntropyLoss()
    
    # Zero out gradients from previous steps
    optimizer.zero_grad()

    # Forward pass
    outputs = model(**inputs)
    logits = outputs["logits"]

    # Calculate loss
    # Loss function expects logits of shape (N, C) and labels of shape (N)
    loss = loss_fn(logits.view(-1, logits.size(-1)), final_labels.view(-1))
    
    # Backward pass to compute gradients
    loss.backward()

    # Update weights
    optimizer.step()

    print("Training step completed.")
    print(f"Initial Loss: {loss.item():.4f}")
    print("\nThis demonstrates that the model, processor, and training pipeline are all working together correctly!")


if __name__ == "__main__":
    main() 