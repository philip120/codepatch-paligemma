import torch
from torch.optim import AdamW
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from datasets import load_dataset
from tqdm import tqdm
import argparse
import os

from modeling_codepatch import CodePatchConfig, CodeEncoderConfig, CodePatchForConditionalGeneration
from processing_codepatch import CodePatchProcessor

def get_args():
    parser = argparse.ArgumentParser(description="Train a CodePatch model.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--output_dir", type=str, default="codepatch_checkpoint", help="Directory to save model checkpoints.")
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

    code_encoder_config = CodeEncoderConfig(freeze_encoder=True)
    processor = CodePatchProcessor(
        "microsoft/codebert-base", "microsoft/codebert-base", code_encoder_config, device=device
    )

    text_config_dict = {
        "vocab_size": processor.text_tokenizer.vocab_size,
        "hidden_size": 2048, # Matching a typical small LLM
        "intermediate_size": 8192,
        "num_hidden_layers": 12,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "max_position_embeddings": 4096,
    }

    full_config = CodePatchConfig(
        code_encoder_config=vars(code_encoder_config),
        text_config=text_config_dict,
        projection_dim=text_config_dict["hidden_size"],
        freeze_llm=True,
    )

    print("Instantiating model...")
    model = CodePatchForConditionalGeneration(full_config).to(device)
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