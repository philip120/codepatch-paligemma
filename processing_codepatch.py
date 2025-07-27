from transformers import AutoTokenizer
import torch
from typing import List, Union

from modeling_codepatch import CodeEncoderConfig


class CodePatchProcessor:
    def __init__(self, code_tokenizer_name: str, text_tokenizer_name: str, config: CodeEncoderConfig, device: str = None):
        self.code_tokenizer = AutoTokenizer.from_pretrained(code_tokenizer_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_name)
        self.config = config
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    def _process_code(self, code_strings: List[str]):
        """ Processes a batch of code strings into tokenized and patched tensors. """
        max_length = self.config.num_patches * self.config.patch_length
        inputs = self.code_tokenizer(
            code_strings,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        
        input_ids = inputs["input_ids"]
        
        # Reshape the tokenized inputs into patches
        patched_input_ids = input_ids.view(
            -1, self.config.num_patches, self.config.patch_length
        )
        
        # The attention mask for the patches is simply whether the token is a pad token or not.
        # We can create it directly from the patched input_ids.
        attention_mask = (patched_input_ids != self.code_tokenizer.pad_token_id).long()

        return patched_input_ids.to(self.device), attention_mask.to(self.device)

    def _process_text(self, text_prompts: List[str]):
        """ Processes a batch of text prompts into tokenized tensors. """
        prompt_inputs = self.text_tokenizer(
            text_prompts, return_tensors="pt", padding="longest", truncation=True, max_length=512
        )
        prompt_input_ids = prompt_inputs["input_ids"].to(self.device)
        prompt_attention_mask = prompt_inputs["attention_mask"].to(self.device)
        return prompt_input_ids, prompt_attention_mask

    def __call__(self, code: Union[str, List[str]], text: Union[str, List[str]]):
        if isinstance(code, str):
            code = [code]
        if isinstance(text, str):
            text = [text]

        code_input_ids, code_attention_mask = self._process_code(code)
        prompt_input_ids, prompt_attention_mask = self._process_text(text)

        return {
            "code_input_ids": code_input_ids,
            "code_attention_mask": code_attention_mask,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
        } 