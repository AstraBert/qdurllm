import warnings
warnings.filterwarnings('ignore')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import setup_chat_format
from typing import Dict, List

model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
quantization_config = BitsAndBytesConfig(load_in_4bit = True, bnb_4bit_compute_dtype = torch.bfloat16, bnb_4bit_use_double_quant = True, bnb_4bit_quant_type = 'nf4')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
quantized_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype = torch.bfloat16, quantization_config = quantization_config).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.chat_template = None
(quantized_model, tokenizer) = setup_chat_format(model = quantized_model, tokenizer = tokenizer)

def pipe(prompt: List[Dict[(str, str)]], temperature: float, top_p: float, max_new_tokens: int, repetition_penalty: float) -> str:
    tokenized_chat = tokenizer.apply_chat_template(prompt, tokenize = True, add_generation_prompt = True, return_tensors = 'pt').to(device)
    outputs = quantized_model.generate(tokenized_chat, max_new_tokens = max_new_tokens, temperature = temperature, top_p = top_p, repetition_penalty = repetition_penalty).to(device)
    results = tokenizer.decode(outputs[0])
    return results

