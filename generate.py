from transformers import GPT2LMHeadModel
from tokenizers import ByteLevelBPETokenizer
import torch
import re
from typing import Optional

class TextGenerator:
    def __init__(self, model_path: str, tokenizer_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = self._load_tokenizer(tokenizer_path)
        self.model = self._load_model(model_path).to(self.device)
        self.model.eval()
        
    def _load_tokenizer(self, path: str) -> ByteLevelBPETokenizer:
        tokenizer = ByteLevelBPETokenizer.from_file(
            vocab_filename=f"{path}/vocab.json",
            merges_filename=f"{path}/merges.txt"
        )
        
        # Добавляем обязательные специальные токены
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
        tokenizer.add_special_tokens(special_tokens)
        return tokenizer
    
    def _load_model(self, path: str) -> GPT2LMHeadModel:
        model = GPT2LMHeadModel.from_pretrained(path)
        return model
    
    def generate(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 0.5,
        top_k: int = 30,
        repetition_penalty: float = 1.2,
        stop_sequences: Optional[list] = None
    ) -> str:
        input_ids = self.tokenizer.encode(prompt).ids
        input_ids = torch.tensor([input_ids]).to(self.device)
        
        output = self.model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=self.tokenizer.token_to_id("[PAD]"),
            eos_token_id=self.tokenizer.token_to_id("[SEP]")
        )
        
        text = self.tokenizer.decode(output[0].tolist())
        
        # Пост-обработка
        text = self._postprocess(text, stop_sequences)
        return text
    
    def _postprocess(self, text: str, stop_sequences: Optional[list]) -> str:
        # Удаление повторяющихся фраз
        text = re.sub(r'(\b\w+\b)(?:\s+\1)+', r'\1', text)
        
        # Обрезка по стоп-последовательностям
        if stop_sequences:
            for seq in stop_sequences:
                if seq in text:
                    text = text[:text.index(seq)] + seq
        
        # Удаление лишних специальных токенов
        for token in ["[PAD]", "[CLS]", "[SEP]"]:
            text = text.replace(token, "")
            
        return text.strip()