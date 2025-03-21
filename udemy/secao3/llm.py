import os
import getpass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

model_id = "microsoft/Phi-3-mini-4k-instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # normal float 4
    bnb_4bit_use_double_quant=True, # quantização dupla, melhora a estabilidade do modelo
    bnb_4bit_compute_dtype=torch.bfloat16 # define o tipo de dado para cálculo, brain float point
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
