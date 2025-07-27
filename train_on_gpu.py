import torch
from torch.optim import AdamW
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from datasets import load_dataset
from tqdm import tqdm
import argparse
import os

# We now need the real Gemma model from the transformers library
from transformers import GemmaForCausalLM, GemmaConfig

from modeling_codepatch import CodePatchConfig, CodeEncoderConfig, CodePatchForConditionalGeneration
from processing_codepatch import CodePatchProcessor

def get_args():
    parser = argparse.ArgumentParser(description="Train a CodePatch model.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for fine-tuning.")
    parser.add_argument("--output_dir", type=str, default="codepatch_checkpoint", help="Directory to save model checkpoints.")
    parser.add_argument("--checkpoint_to_load", type=str, default=None, help="Path to a checkpoint to continue training from.")
    return parser.parse_args()

def collate_fn(batch, processor, code_config):
    codes = [item['0']['value'] for item in batch]
    prompts = [item['1']['value'] for item in batch]

    inputs = processor(code=codes, text=prompts)
    
    labels = inputs["prompt_input_ids"].clone()
    
    # Create ignore labels for the code patch part
    code_patch_ignore_labels = torch.full(
        (len(batch), code_config.num_patches), -100, device=labels.device
    )
    
    # Concatenate ignore labels with the actual prompt labels
    final_labels = torch.cat([code_patch_ignore_labels, labels], dim=1)

    # The model expects labels in the final dict
    inputs['labels'] = final_labels
    return inputs

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    gemma_model_name = "google/gemma-2b"

    code_encoder_config = CodeEncoderConfig(freeze_encoder=True)
    processor = CodePatchProcessor(
        "microsoft/codebert-base", gemma_model_name, code_encoder_config, device=device
    )

    # Load the official Gemma config
    text_config = GemmaConfig.from_pretrained(gemma_model_name)

    full_config = CodePatchConfig(
        code_encoder_config=vars(code_encoder_config),
        text_config=text_config.to_dict(), # Pass the config as a dictionary
        projection_dim=text_config.hidden_size,
        freeze_llm=True, # <-- Freeze the LLM and only train the projector/embeddings
    )

    print("Instantiating model and loading REAL Gemma weights...")
    model = CodePatchForConditionalGeneration(full_config).to(device)

    # Load the pre-trained weights for the language model part
    gemma_model = GemmaForCausalLM.from_pretrained(gemma_model_name, torch_dtype=torch.bfloat16)
    model.language_model.load_state_dict(gemma_model.state_dict())
    
    # If a checkpoint is provided, load the trained projector and embedding weights
    if args.checkpoint_to_load:
        print(f"Loading projector/embedding weights from: {args.checkpoint_to_load}")
        checkpoint = torch.load(args.checkpoint_to_load, map_location=device)
        model.multi_modal_projector.load_state_dict(checkpoint['projector_state_dict'])
        model.code_encoder.position_embedding.load_state_dict(checkpoint['pos_embedding_state_dict'])
        print("Successfully loaded projector and embedding weights.")

    model.train()

    print("Loading dataset...")
    dataset = load_dataset("philip120/RPOFES-dataset", split="train")
    
    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, processor, code_encoder_config)
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler()

    print("--- Starting Training ---")
    for epoch in range(args.epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Move all tensors in the batch to the device
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            with autocast():
                outputs = model(**batch)
                logits = outputs["logits"]
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        # Save the trainable parts of the model
        checkpoint_path = os.path.join(args.output_dir, f"epoch_{epoch+1}_checkpoint.pt")
        torch.save({
            'projector_state_dict': model.multi_modal_projector.state_dict(),
            'pos_embedding_state_dict': model.code_encoder.position_embedding.state_dict(),
        }, checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")

    print("--- Training Complete ---")

if __name__ == "__main__":
    main() 