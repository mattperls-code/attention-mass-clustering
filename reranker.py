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

# attn_implementation = ("flash_attention_2" if device.type == "cuda" else "sdpa") if Path(sys.modules["__main__"].__file__).name == "evaluate_model.py" else "eager"
attn_implementation = "sdpa" if Path(sys.modules["__main__"].__file__).name == "evaluate_model.py" else "eager"
print(f"Using attn_implementation={attn_implementation}")

base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=1, attn_implementation=attn_implementation, dtype=torch.float16)
base_model.eval()

ft_model = PeftModel.from_pretrained(
    AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=1, attn_implementation=attn_implementation, dtype=torch.float16),
    ft_model_name
).merge_and_unload()
ft_model.eval()

@contextmanager
def use_lora_ablated_model(heads: list[tuple[int, int]]):
    num_q_heads = ft_model.config.num_attention_heads
    num_kv_heads = ft_model.config.num_key_value_heads
    hidden_size = ft_model.config.hidden_size

    q_head_dim = hidden_size // num_q_heads
    kv_head_dim = hidden_size // num_kv_heads
    kv_groups = num_q_heads // num_kv_heads

    cache = {}

    with torch.no_grad():
        for layer_index, head_index in heads:
            q_start, q_end = head_index * q_head_dim, (head_index + 1) * q_head_dim
            
            kv_index = head_index // kv_groups
            kv_start, kv_end = kv_index * kv_head_dim, (kv_index + 1) * kv_head_dim

            ft_attn = ft_model.model.layers[layer_index].self_attn
            base_attn = base_model.model.layers[layer_index].self_attn

            cache[(layer_index, head_index)] = {
                "q_proj": ft_attn.q_proj.weight[q_start:q_end, :].clone(),
                "k_proj": ft_attn.k_proj.weight[kv_start:kv_end, :].clone(),
                "v_proj": ft_attn.v_proj.weight[kv_start:kv_end, :].clone(),
                "o_proj": ft_attn.o_proj.weight[:, q_start:q_end].clone(),
            }

            ft_attn.q_proj.weight[q_start:q_end, :].copy_(base_attn.q_proj.weight[q_start:q_end, :])
            ft_attn.k_proj.weight[kv_start:kv_end, :].copy_(base_attn.k_proj.weight[kv_start:kv_end, :])
            ft_attn.v_proj.weight[kv_start:kv_end, :].copy_(base_attn.v_proj.weight[kv_start:kv_end, :])
            ft_attn.o_proj.weight[:, q_start:q_end].copy_(base_attn.o_proj.weight[:, q_start:q_end])

    try:
        yield ft_model

    finally:
        with torch.no_grad():
            for (layer_index, head_index), weights in cache.items():
                q_start, q_end = head_index * q_head_dim, (head_index + 1) * q_head_dim
                kv_index = head_index // kv_groups
                kv_start, kv_end = kv_index * kv_head_dim, (kv_index + 1) * kv_head_dim

                ft_attn = ft_model.model.layers[layer_index].self_attn
                ft_attn.q_proj.weight[q_start:q_end, :].copy_(weights["q_proj"])
                ft_attn.k_proj.weight[kv_start:kv_end, :].copy_(weights["k_proj"])
                ft_attn.v_proj.weight[kv_start:kv_end, :].copy_(weights["v_proj"])
                ft_attn.o_proj.weight[:, q_start:q_end].copy_(weights["o_proj"])

def get_attention_layers(model, query: str, document: str):
    tokens = tokenizer(query, document, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**tokens, output_attentions=True)

        return torch.stack(outputs.attentions)[:, 0]
    
def use_model(model, query: str, documents: list[str]):
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    tokens = tokenizer(
        [ query ] * len(documents),
        documents,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    model.config.pad_token_id = tokenizer.eos_token_id
    
    with torch.no_grad():
        return model(**tokens).logits.squeeze(-1).tolist()