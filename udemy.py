from transformers import FlaxAutoModelForCausalLM
# adequados para tarefas de geração de texto
from transformers import AutoTokenizer
# fazer a transformação em número 
from transformers import pipeline
from transformers import BitsAndBytesConfig
# para melhorar a eficiência computacional
import torch
import getpass
import os

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(device)
