from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # normal float 4
    bnb_4bit_use_double_quant=True,  # quantização dupla, melhora a estabilidade do modelo
    bnb_4bit_compute_dtype=torch.bfloat16,  # define o tipo de dado para cálculo, brain float point
)

model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=quantization_config
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
