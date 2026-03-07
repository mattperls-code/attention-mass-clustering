import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

tokenizer_name = "meta-llama/Llama-2-7b"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

base_model_name = "meta-llama/Llama-2-7b-hf"
model_name = "castorini/rankllama-v1-7b-lora-passage"

base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=1, attn_implementation="eager")
model = PeftModel.from_pretrained(base_model, model_name)
model = model.merge_and_unload()

def get_attention_layers(query: str, document: str):
    tokens = tokenizer(query, document, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**tokens, output_attentions=True)

        return outputs.attentions