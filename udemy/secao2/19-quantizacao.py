# https://huggingface.co/blog/4bit-transformers-bitsandbytes

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

print("1")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # normal float 4
    bnb_4bit_use_double_quant=True, # quantização dupla, melhora a estabilidade do modelo
    bnb_4bit_compute_dtype=torch.bfloat16 # define o tipo de dado para cálculo, brain float point
) 

print("2")
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
print("3")
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
print("4")
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = ("Quem foi a primeira pessoa no espaço?")
messages = [
    {"role": "user", "content": prompt }
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
model_inputs = encodeds.to('cpu')
generated_ids = model.generate(
    model_inputs, 
    max_new_tokens=1000, 
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id 
)
decoded = tokenizer.batch_decode(generated_ids)
res = decoded[0]
print(res)
