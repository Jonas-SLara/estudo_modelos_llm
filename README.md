```bash
# Linux/macOS
source venv/bin/activate

pip install --upgrade pip
pip install transformers torch accelerate
pip freeze > requirements.txt
```


## Resumo das bibliotecas usadas

1. torch (PyTorch)

    O que é: É o motor de computação (deep learning framework)

    Para que serve: O PyTorch é a base de tudo. Ele gerencia as redes neurais e faz o trabalho pesado de matemática e cálculo (álgebra linear). Ele permite criar tensores (semelhantes a arrays do NumPy) que rodam de forma extremamente rápida utilizando a placa de vídeo (GPU), o que é indispensável para treinar e rodar modelos de IA modernos.

2. transformers (Hugging Face Transformers)

    O que é: A biblioteca de modelos e arquiteturas de IA mantida pela empresa Hugging Face.

    Para que serve: Ela fornece implementações prontas para uso de milhares de modelos de ponta (como BERT, GPT, T5, Whisper, Llama, etc.). Em vez de você precisar programar uma rede neural do zero, a biblioteca transformers permite baixar um modelo pré-treinado com poucas linhas de código, tokenizar textos, gerar respostas, traduzir idiomas e realizar fine-tuning (ajuste fino) de forma muito simples.

3. accelerate

    O que é: Uma biblioteca complementar desenvolvida também pela Hugging Face para otimização e execução em hardware.

    Para que serve: O treinamento e a execução de modelos gigantescos exigem infraestruturas complexas (como múltiplas GPUs, uso de precisão mista em FP16 ou BF16, ou TPUs). O accelerate abstrai toda essa complexidade de infraestrutura, permitindo que o seu código PyTorch rode de forma eficiente em qualquer hardware sem que você precise reescrever códigos complexos de distribuição paralela.



## Pipeline de inferência de um LLM
oque o codigo faz 

1. Carregamento do modelo e tokenizer

Pesos da rede neural + vocabulário e regras de tokenização


2. Preparação do prompt

Mensagens → template de chat → tokens


3. Inferência (generate)

A rede neural calcula distribuições de probabilidade e seleciona tokens


4. Decodificação

Tokens gerados → texto legível

## AutoTokenizer

É responsável por transformar texto em tokens e fazer o processo inverso.

Por exemplo:

```bash
texto = "Olá, como você está?"tokens = tokenizer.encode(texto)print(tokens)
```
O resultado será uma sequência de números inteiros que representam os tokens reconhecidos pelo vocabulário do modelo.
É importante entender que token não significa necessariamente palavra.

Um token pode representar:
- Uma palavra inteira.
- Parte de uma palavra.
- Um caractere.
- Um espaço ou pontuação associado a uma palavra.
- Um token especial de controle.
O modelo não trabalha diretamente com strings. Ele trabalha com representações numéricas.

## AutoModelForCausalLM

É responsável por carregar a arquitetura e os pesos da rede neural treinada.
O CausalLM significa Causal Language Model.
É uma arquitetura de linguagem que aprende a prever o próximo token considerando os tokens anteriores.
Imagine a seguinte entrada:

O céu está
O modelo calcula quais tokens poderiam aparecer em seguida:

azul      → 0.65
nublado   → 0.20
bonito    → 0.10
escuro    → 0.05


Esses valores são apenas ilustrativos.

O modelo seleciona um próximo token de acordo com sua estratégia de geração, adiciona esse token à sequência e repete o processo.

É assim que um modelo autoregressivo constrói respostas.

## Identificação do modelo

```bash
nome = "Qwen/Qwen3-0.6B"
```

Aqui você está identificando qual modelo deseja utilizar.
Esse identificador aponta para um repositório no Hugging Face Hub.

Qwen3-0.6B

Família Qwen3 · Alibaba
Repositório oficial no Hugging Face

Vamos decompor o nome:

Parte	Significado
Qwen	Família de modelos
3	Terceira geração
0.6B	Aproximadamente 600 milhões de parâmetros

## O que são parâmetros?

Parâmetros são os valores numéricos aprendidos durante o treinamento da rede neural.
São eles que armazenam os padrões estatísticos e representações aprendidas pelo modelo.

De forma simplificada:
- Modelo pequeno: menos parâmetros, menor consumo de memória e geralmente menor capacidade.
- Modelo grande: mais parâmetros, maior consumo de memória e potencialmente maior capacidade.
Mas quantidade de parâmetros não é sinônimo automático de inteligência. Arquitetura, qualidade dos dados, treinamento e pós-treinamento também importam.

## Carregando o tokenizer

```bash
tokenizer = AutoTokenizer.from_pretrained(nome)
```

O método from_pretrained() é um dos principais recursos do ecossistema Transformers.
Ele procura os arquivos necessários para carregar o tokenizer correspondente ao modelo.

Normalmente, esses arquivos incluem:
- Vocabulário.
- Regras de tokenização.
- Configurações do tokenizer.
- Tokens especiais.
Se os arquivos não estiverem disponíveis no cache local, a biblioteca os baixa do Hugging Face Hub.

Por que o tokenizer precisa ser o mesmo do modelo?

Porque o modelo foi treinado utilizando uma representação específica de tokens.

Imagine que o tokenizer transforme:
programação

em:
[1234, 5678]

Se você utilizar outro tokenizer, poderá obter:
[987, 654, 321]

Esses IDs possuem significados diferentes para os pesos da rede neural.
Portanto, não é seguro utilizar um tokenizer aleatório ou incompatível com o modelo.

## Carregando a rede neural

```bash
modelo = AutoModelForCausalLM.from_pretrained(    nome,    torch_dtype="auto",    device_map="auto",)
```

Você está carregando efetivamente a rede neural que fará a inferência.
Vamos analisar os parâmetros.

### torch_dtype="auto"

O PyTorch utiliza diferentes tipos numéricos para representar os pesos da rede neural como:

- float32	Ponto flutuante de 32 bits
- float16	Ponto flutuante de 16 bits
- bfloat16	Ponto flutuante de 16 bits
- int8	Inteiro de 8 bits

O parâmetro torch_dtype="auto" permite que a biblioteca escolha um tipo compatível com a configuração disponibilizada pelo modelo.

Isso influencia diretamente o consumo de memória e o desempenho.

Por exemplo, imagine uma rede com 600 milhões de parâmetros:
- Em FP32: aproximadamente 2,4 GB apenas para os pesos.
- Em FP16: aproximadamente 1,2 GB apenas para os pesos.
São estimativas decimais, sem contar memória adicional de execução.

Uma observação importante: reduzir a precisão numérica não é exatamente a mesma coisa que fazer quantização. Técnicas como INT8 e 4-bit possuem mecanismos próprios.

### device_map="auto"

Esse parâmetro controla onde os componentes do modelo serão carregados.

O PyTorch trabalha com dispositivos como:
- cpu: processador.
- cuda: GPU NVIDIA.
- mps: GPU Apple Silicon.
Com device_map="auto", o Transformers utiliza o mecanismo de distribuição de dispositivos do Accelerate para tentar posicionar os componentes do modelo nos recursos disponíveis.

Por exemplo:

device_map="cpu"
Forçaria o carregamento na CPU.

device_map="auto"
permite a escolha automática, podendo distribuir componentes entre GPU e CPU quando necessário.

Uma observação: o device_map não cria uma GPU nem garante aceleração. Se você não tiver uma GPU compatível, o modelo poderá ser executado na CPU.

## Construindo o prompt

```bash
prompt = "Crie em Python um gerenciador de tarefas de linha de comando com adicionar, listar e concluir tarefas."mensagens = [{"role": "user", "content": prompt}]
```

Aqui você começa a preparar a entrada que será enviada ao modelo.
Primeiro, temos uma string Python:

Depois, você transforma essa string em uma estrutura de mensagens:

```bash
mensagens = [    
    {        
        "role": "user",        
        "content": prompt    
    }
]
```

Essa estrutura segue um padrão conhecido como **Chat Messages Format**.
É uma representação de conversa composta por mensagens com papéis diferentes.

Os principais papéis são:

| Role	| Responsabilidade
|--|--|
system | Define instruções e comportamento geral do assistente
|--|--|
user | Representa a entrada do usuário
|--|--|
assistant | Representa respostas do modelo
|--|--|
tool | Representa resultados de ferramentas externas, dependendo do formato utilizado
|--|--|


Isso é importante porque os modelos de chat não recebem necessariamente apenas um texto isolado. Eles são treinados para reconhecer estruturas de conversação.

Uma analogia com desenvolvimento web: pense que mensagens é um objeto de requisição estruturado que será posteriormente serializado em um formato específico para o modelo.


## Aplicando o Chat Template

```bash
texto = tokenizer.apply_chat_template(
    mensagens, 
    tokenize=False, 
    add_generation_prompt=True,    
    enable_thinking=False,
)
```

Essa é uma das partes mais importantes para entender como os modelos conversacionais funcionam.
O apply_chat_template() converte sua estrutura de mensagens em um formato textual que o modelo reconhece.
Por exemplo, você enviou

[    {"role": "user", "content": "Olá!"}]



O tokenizer pode transformar isso em algo semelhante a:
<|im_start|>user
Olá!<|im_end|>
<|im_start|>assistant


Os tokens especiais variam conforme a arquitetura e o template do modelo.

No caso do Qwen3, temos marcadores como:

| Token especial | Significado |
|---|---|
| \u003c|im_start|\u003e | Início de uma mensagem |
| \u003c|im_sep|\u003e | Separação entre cabeçalho e conteúdo |
| \u003c|im_end|\u003e | Fim de uma mensagem |
| \u003c|im_start|\u003eassistant | Identificação do assistente |

O modelo aprende a interpretar esses marcadores durante seu treinamento.
Agora vamos entender os argumentos.

> tokenize=False

Determina que o método deve retornar uma string formatada, em vez de já converter o conteúdo em IDs numéricos.
Se fosse: tokenize=True O próprio método retornaria os tokens preparados.
Você optou por separar as duas operações: primeiro preparar o texto, depois tokenizá-lo.

> add_generation_prompt=True

Esse parâmetro adiciona o marcador que indica ao modelo que ele deve começar a gerar uma resposta.
Imagine que a conversa termine assim:

<|im_start|>user
Explique Python.
<|im_end|>
<|im_start|>assistant

Observe que ainda não existe conteúdo na mensagem do assistente.
Isso é intencional.

O modelo entende que o próximo conteúdo deverá ser produzido pelo próprio assistente.
Sem esse marcador, dependendo do template, o modelo poderia não receber a indicação correta de que deve iniciar uma nova resposta.

> enable_thinking=False

Esse parâmetro é específico do template do Qwen3.

O Qwen3 possui suporte a dois formatos de geração:
- Com thinking: o modelo pode produzir conteúdo de raciocínio em um canal específico antes da resposta final.
- Sem thinking: o modelo gera diretamente a resposta final, sem ativar esse formato de raciocínio.
Quando ativado, o template pode incluir marcadores como:
<|im_start|>assistant
<|im_sep|>analysis

Ou, na resposta final:
<|im_start|>assistant
<|im_sep|>final


Um detalhe importante: enable_thinking=False não desativa a capacidade de raciocínio que o modelo aprendeu. Ele altera o modo de geração e os marcadores utilizados pelo template.

## Transformando o texto em tensores

entradas = tokenizer([texto], return_tensors="pt").to(modelo.device)
Agora você está transformando o prompt formatado em uma representação numérica que pode ser processada pela rede neural.

Vamos separar as operações.

> tokenizer([texto])

Converte a string em tokens.

Por exemplo:
"Olá, como você está?"

Pode virar uma sequência ilustrativa:
[1234, 567, 890, 234]


Esses números são IDs dos tokens no vocabulário.
Por que [texto] e não texto?
Porque você está fornecendo um lote (batch) de entradas.
Mesmo tendo apenas um prompt, ele está representado como uma lista com um elemento.

Isso permite trabalhar com múltiplas entradas posteriormente:

```bash
textos = [    "Explique Python.",    "Explique Java.",    "Explique Rust."]
entradas = tokenizer(textos, return_tensors="pt", padding=True)
```

Agora você teria três entradas agrupadas em um batch.

> return_tensors="pt"

Determina que os resultados sejam retornados como tensores PyTorch.
O PyTorch é o framework de computação utilizado para executar a rede neural.
Um tensor pode ser entendido como uma estrutura multidimensional de dados numéricos.
No seu exemplo, a entrada principal será uma matriz com duas dimensões:

[batch_size, quantidade_de_tokens]


Por exemplo:
[1, 25]

Significa uma entrada com 25 tokens.

> .to(modelo.device)

Move os tensores para o dispositivo onde o modelo está alocado.
Isso é necessário porque, em geral, você não pode realizar operações entre tensores na CPU e pesos na GPU sem fazer a movimentação apropriada.

Existe uma observação técnica: com modelos distribuídos entre vários dispositivos, modelo.device pode não representar todos os dispositivos utilizados. No seu cenário simples, normalmente funciona.

## Gerando a resposta

```bash
saida = modelo.generate(    
    **entradas,    
    max_new_tokens=1024,    
    do_sample=True,    
    temperature=0.7,    
    top_p=0.8,    
    top_k=20
)
```

Aqui acontece efetivamente a inferência do modelo.
O método generate() recebe os tensores de entrada e executa o processo de geração autoregressiva.

Internamente, de forma simplificada:
1. O modelo recebe os tokens do prompt.
2. Os tokens passam pelas camadas da rede neural.
3. O modelo calcula os logits para o próximo token.
4. Os logits são transformados em uma distribuição de probabilidades.
5. Uma estratégia de seleção escolhe o próximo token.
6. O token selecionado é acrescentado à sequência.
7. O processo continua até atingir uma condição de parada.Agora vamos aos parâmetros.


## Parametros

O operador ** desempacota um dicionário de argumentos.

Por exemplo:

```bash
dados = {
    "input_ids": tensor_ids,
    "attention_mask": tensor_mask
}

modelo.generate(**dados)
```

### max_new_tokens=1024

Define o número máximo de novos tokens que o modelo poderá gerar.
É importante entender que isso não representa 1024 palavras.
São 1024 tokens, incluindo pontuação, palavras e fragmentos de palavras.
Também não representa o tamanho total da entrada: o prompt não entra nessa contagem.
Quanto maior esse valor, maior poderá ser o consumo de memória e o tempo de inferência.

### do_sample=True

Define se a geração utilizará amostragem probabilística.
Quando habilitado, o modelo não necessariamente escolhe o token mais provável.

Imagine uma distribuição:
Token	Probabilidade
Python	0.60
Java	0.25
Rust	0.10
Go	0.05

Em vez de sempre selecionar Python, o modelo pode sortear entre as possibilidades conforme suas probabilidades ajustadas.

Isso torna a geração menos determinística.

Se você utilizar: do_sample=False

A geração utiliza uma estratégia determinística de seleção gulosa (greedy decoding), escolhendo o token de maior probabilidade em cada etapa, salvo outras estratégias de geração configuradas.

### temperature=0.7

A temperatura controla a distribuição de probabilidades antes da seleção do token.
De maneira simplificada, a transformação é:

Temperatura	Comportamento
0.2	Distribuição mais concentrada
0.7	Distribuição moderadamente concentrada
1.0	Distribuição original do softmax
1.5	Distribuição mais distribuída

Temperaturas menores tendem a favorecer respostas mais previsíveis. Temperaturas maiores aumentam a diversidade.

Importante: temperatura não é um controle de inteligência ou conhecimento do modelo.

## top_k=20

Limita os candidatos aos 20 tokens com maior probabilidade.
Mesmo que o vocabulário possua dezenas de milhares de tokens, somente os 20 mais prováveis poderão participar da seleção.

Qual a relação entre top_p e top_k?

São estratégias complementares:
- top_k: limita pela quantidade de candidatos.
- top_p: limita pela probabilidade acumulada.
Você está utilizando ambas para controlar a diversidade da geração.

Um detalhe: esses parâmetros não são totalmente independentes. A temperatura altera as probabilidades utilizadas na seleção e os filtros podem atuar sobre essa distribuição.

## Separando os tokens gerados

novos_tokens = saida[0][len(entradas.input_ids[0]):]
Essa linha é muito interessante porque demonstra como a geração funciona internamente.
O resultado de modelo.generate() não contém somente a resposta nova.

Ele contém:
1. Os tokens originais enviados no prompt.
2. Os tokens gerados pelo modelo.
Imagine que sua entrada tenha:

entradas.input_ids[0] = [10, 20, 30, 40]

E o modelo gere:

saida[0] = [10, 20, 30, 40, 50, 60, 70]

Os primeiros quatro tokens são a entrada original.
Os últimos três são a resposta gerada.

Quando você executa:

len(entradas.input_ids[0])

Obtém a quantidade de tokens da entrada.
E quando utiliza:

saida[0][4:]

Está realizando um slicing para obter somente os novos tokens:

[50, 60, 70]

Isso evita imprimir novamente o prompt junto com a resposta.

## Decodificando os tokens

print(tokenizer.decode(novos_tokens, skip_special_tokens=True))
Agora você faz o processo inverso da tokenização.

Anteriormente:

Texto → Tokens

Agora:
Tokens → Texto


O método decode() recebe os IDs dos tokens e reconstrói uma string legível.

skip_special_tokens=True

Remove tokens especiais durante a decodificação.

Por exemplo, marcadores internos como:
<|im_start|>
<|im_end|>
<|im_sep|>


Não aparecerão no resultado final.

Isso é útil para apresentar ao usuário apenas o conteúdo da resposta, sem os marcadores estruturais utilizados pelo modelo.



## Trilha de aprendizagem: Hugging Face e LLMs
1. Tokenização e embeddings
   Próximo passo
   Entenda como IDs de tokens são convertidos em vetores numéricos e como esses vetores representam informações para a rede neural.
   
   Conceitos: Token IDs, Vocabulary, Embedding Layer, Vector Space.
2. Arquitetura Transformer
   
   Entenda como a rede processa as informações.
   
   Conceitos: Self-Attention, Query, Key, Value, Multi-Head Attention, Positional Encoding, Feed Forward.
3. Inferência e estratégias de geração
   
   Aprofunde o que você já está experimentando.
   
   Conceitos: Logits, Softmax, Greedy Search, Sampling, Temperature, Top-K, Top-P, KV Cache.
4. Fine-tuning e adaptação
   
   Aprenda a adaptar modelos para tarefas específicas.
   
   Conceitos: Pretraining, Supervised Fine-Tuning, LoRA, QLoRA, PEFT.
5. RAG e integração com aplicações
   
   Conecte LLMs com fontes externas de conhecimento.
   
   Conceitos: Embeddings, Vector Database, Chunking, Retrieval, Reranking, RAG.
6. Servir modelos em produção
   
   Integre modelos com aplicações backend.
   
   Conceitos: FastAPI, Streaming, Batching, Quantization, vLLM, Ollama, Docker.


## Prompt

```bash
Crie em Python um gerenciador de tarefas de linha de comando com adicionar, listar e concluir tarefas.
```

## Saída

```bash
Claro! Vamos criar um **gerenciador de tarefas de linha de comando em Python** com funções para **adicionar, listar e concluir tarefas**. A interface é simples e pode ser adaptada para usar em um terminal ou script.

### Exemplo: Gerenciador de Tarefas em Python

```python
# Gerenciador de tarefas com adicionar, listar e concluir tarefas
import os
import sys

def adicionar_tarefa(tarefa):
    """Adicionar uma tarefa a uma lista de tarefas."""
    if not tarefa.strip():
        print("Erro: Tarefa deve ter conteúdo.")
        return False
    return True

def listar_tarefas():
    """Listar as tarefas no terminal."""
    print("Tarefas:")
    tarefas = [f"{tarefa.strip()} - Tarefa adicionada"]
    for tarefa in tarefas:
        print(tarefa)

def concluir_tarefa(tarefa):
    """Concluir uma tarefa."""
    print("Tarefa {} foi concluída.".format(tarefa.strip()))
    return True

def main():
    print("Bem-vindo ao Gerenciador de Tarefas!")
    print("Opções:")
    print("1. Adicionar tarefa")
    print("2. Listar tarefas")
    print("3. Concluir tarefa")
    print("4. Sair")

    while True:
        escolha = input("Escolha uma opção: ")

        if escolha == "1":
            tarefa = input("Digite a tarefa: ")
            if adicionar_tarefa(tarefa):
                print("Tarefa adicionada com sucesso.")
            else:
                print("Erro ao adicionar tarefa.")

        elif escolha == "2":
            listar_tarefas()

        elif escolha == "3":
            tarefa = input("Digite a tarefa a ser concluída: ")
            if concluir_tarefa(tarefa):
                print("Tarefa concluída.")
            else:
                print("Erro ao concluir tarefa.")

        elif escolha == "4":
            print("Saindo do gerenciador de tarefas.")
            sys.exit(0)

        else:
            print("Opção inválida. Por favor, escolha 1, 2, 3 ou 4.")

if __name__ == "__main__":
    main()
```

### Exemplo de Uso:
```bash
python task_manager.py
```

### Descrição:
- **Adicionar tarefa**: Pode ser adicionada com conteúdo.
- **Listar tarefas**: Mostra as tarefas no terminal.
- **Concluir tarefa**: Conclui uma tarefa, informando o resultado.

Se quiser, posso adaptar para usar em um script ou com biblioteca de tarefas.
```