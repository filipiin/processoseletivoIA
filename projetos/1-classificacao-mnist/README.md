# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Filipe Leite Ribeiro:**

### 1️⃣ Resumo da Arquitetura do Modelo

A rede neural construída é uma CNN (Rede Neural Convolucional) enxuta e eficiente, dividida em duas grandes partes:

Extração de Características (3 Blocos): Cada bloco conta com uma camada Conv2D (começando com 32 filtros, depois 64 e 128) para capturar desde linhas simples até padrões complexos. Logo após, aplico um BatchNormalization para estabilizar o aprendizado numérico e um MaxPooling2D (2x2) para reduzir o peso computacional.

Classificação: Após achatar os dados (Flatten), passo por uma camada Dense de 128 neurônios. Para segurar o overfitting, aplico um Dropout de 30% (desativando neurônios aleatórios). A saída é uma Dense de 10 neurônios com ativação softmax, gerando as probabilidades para os dígitos de 0 a 9.

Treinamento e Validação: Separei 10% dos dados para validação (validation_split=0.1). Usei o callback de EarlyStopping (paciência = 2) monitorando a val_loss. Assim, se o modelo parar de aprender, o treino é abortado e os melhores pesos são automaticamente restaurados (restore_best_weights=True).

### 2️⃣ Bibliotecas Utilizadas

tensorflow (v2.15.1): O motor principal do projeto, usado tanto para treinar (Keras) quanto para converter o modelo (TFLite).

numpy (v1.26.4): Essencial para a manipulação rápida dos arrays das imagens.

os (nativa do Python): Utilizada para ler o tamanho final dos arquivos no disco.Liste as principais bibliotecas utilizadas, preferencialmente com suas versões.

### 3️⃣ Técnica de Otimização do Modelo

A técnica aplicada foi a Dynamic Range Quantization (habilitada através do comando converter.optimizations = [tf.lite.Optimize.DEFAULT]).

De forma simples: ela pega os "pesos" da rede, que originalmente estão em formato float32 (ponto flutuante, pesados), e os converte para int8 (inteiros, leves). É um atalho excelente para encolher drasticamente o tamanho do arquivo para rodar em dispositivos Edge, sem precisar de um dataset de calibração extra.

### 4️⃣ Resultados Obtidos

Acurácia: O modelo atingiu uma excelente acurácia de validação de 99,42% (com perda em 0,0261). No teste cego, cravou 99,39%.

Tamanhos dos Arquivos:

model.h5 (Original): 2,91 MB

model.tflite (Otimizado): 247,73 KB

Impacto: Conseguimos uma redução de 91,5% no tamanho do modelo, o que o torna ideal para aplicações embarcadas.

### 5️⃣ Comentários Adicionais (Opcional)

A escolha dos hiperparâmetros fez bastante diferença na eficiência do código. Dobrar os filtros convolucionais a cada bloco (32 > 64 > 128) é uma estratégia bem clássica que funcionou muito bem, pois evita desperdiçar poder de processamento logo na entrada. O dropout de 30% foi o "sweet spot": segurou a rede para não decorar o MNIST, mas manteve informação suficiente para a alta acurácia.

Um aprendizado importante sobre a otimização de Edge: embora a Dynamic Range Quantization seja fantástica para salvar espaço de armazenamento, as ativações durante a inferência ainda rodam em float. Isso significa que ganhamos muito em "tamanho no disco", mas o ganho em "velocidade de processamento" não é tão extremo. Para um ambiente de produção com hardware extremamente restrito, o próximo passo lógico seria testar uma Full Integer Quantization.

### 6️⃣ Exemplo de Inferência

Rodando inferência em 5 amostras usando model.tflite:

Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4