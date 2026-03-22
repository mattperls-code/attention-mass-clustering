import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from contextlib import contextmanager

tokenizer_name = "meta-llama/Llama-2-7b"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

base_model_name = "meta-llama/Llama-2-7b-hf"
ft_model_name = "castorini/rankllama-v1-7b-lora-passage"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

@contextmanager
def using_device(model):
    model.to(device)

    yield model

    model.to("cpu")

    if device.type == "cuda": torch.cuda.empty_cache()

attn_implementation = ("flash_attention_2" if device == "cuda" else "sdpa") if Path(sys.modules['__main__'].__file__).name == "evaluate_model.py" else "eager"
print(f"Using attn_implementation={attn_implementation}")

base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=1, attn_implementation=attn_implementation, dtype=torch.float16)
base_model.eval()

ft_model = PeftModel.from_pretrained(
    AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=1, attn_implementation=attn_implementation, dtype=torch.float16),
    ft_model_name
).merge_and_unload()
ft_model.eval()

def get_attention_layers(model, query: str, document: str):
    tokens = tokenizer(query, document, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**tokens, output_attentions=True)

        return torch.stack(outputs.attentions)[:, 0]
    
def use_model(model, query: str, document: str):
    tokens = tokenizer(query, document, return_tensors="pt").to(device)
    
    with torch.no_grad():
        return model(**tokens).logits.item()