# Experimentos com LangChain e HuggingFace

## Estudos do curso Domine LLMs com LangChain

Instrutor: Jones Granatyr

### HuggingFace

Crie um Token no HuggingFace.

#### Geração de Texto

#### Tipos de Modelo

- Ao navegar pelos repositórios de modelos de IA generativa (como o HuggingFace) 
 observará modelos listados como os sufixos 'instruct' ou 'chat'.
- Isso porque há pelo menos três tipos principais de LLMs:
  - **Models Base (base models)** - Modelos Base passam apenas pele pré-treinamento 
  e completam textos com as palavras mais prováveis.
  - **Modelos ajustados com instruções (Instruct-tuned)** - Modelos ajustados para 
  instruções passam por uma etapa adicional de ajuste para instruções, melhorando 
  a capacidade de seguir comandos específicos.
  - **Modelos de chat (chat models)** - foram ajustados para funcionar em chatbots, 
  portanto, podem ser mais apropriados para conversas. Alguns modelos de chat 
  podem ser chamados de instrução e vice-versa dependendo da situação, então para 
  facilitar você irá encontrar situações onde um modelo usado para chat é classificado
  como de instrução.
- A versao 'instruct' do modelo foi ajustada para seguir instruções fornecidas. Esses 
modelos 'esperam' ser solicitados a fazer algo.
- Em contraste, modelos não ajustados para instruções simplesmente geram uma saída 
que continua a partir do prompt.
- Qual modelo usar:
  - Para criar chatbots, implementar RGA ou usar agentes, use modelos 'instruct' ou 'chat'.
  - Em caso de dúvida, use um modelo 'instruct'.

O modelo Llama 3, modelo open source da empresa Meta e que no segundo semestre de 2024 demonstrou superar em diveras tarefas versões atuais e recentes de modelos proprietários como o próprio ChatGPT.

https://hugginface.co/meta-llama/Meta-Llama-3-8B

https://hugginface.co/meta-llama/Meta-Llama-3-8B-Instruct
