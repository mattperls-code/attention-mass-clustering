from transformers import AutoTokenizer

tokenizer_name = "meta-llama/Llama-2-7b"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)