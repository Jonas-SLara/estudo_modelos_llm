import re
from transformers import AutoModelForCausalLM, AutoTokenizer

FILE = "task_manager.py"
FILE_MODELO = "resposta_completa_qwen.md"
MODELO = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(MODELO)
modelo = AutoModelForCausalLM.from_pretrained(
    MODELO,
    torch_dtype="auto",
    device_map="auto",
)

print("### Digite o seu prompt: ")
prompt = input()

mensagens = [
    {
        "role": "user", 
        "content": prompt
    }
]

texto = tokenizer.apply_chat_template(
    mensagens,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,  # True liga o modo de raciocínio
)
entradas = tokenizer([texto], return_tensors="pt").to(modelo.device)

saida = modelo.generate(
    **entradas,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
)

novos_tokens = saida[0][len(entradas.input_ids[0]):]

## DECODIFICAR A SAIDA E SALVAR EM ARQUIVO
saida = tokenizer.decode(novos_tokens, skip_special_tokens=True)

blocos = re.findall(r"```[\w+-]*\n(.*?)```", saida, flags=re.DOTALL)
codigo = blocos[0] if blocos else saida  # se não achar bloco, salva a resposta inteira

with open(FILE, "w", encoding="utf-8") as f:
    f.write(codigo.strip() + "\n")

with open(FILE_MODELO, "w", encoding="utf-8") as f:
    f.write(saida)

print(f"### Saída salva em {FILE} e em {FILE_MODELO}")