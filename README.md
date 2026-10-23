# AutoLeitura

Leitura automática de medidores de consumo (água e energia) por visão computacional.
Projeto de Visão Computacional, Ciência da Computação, UNISAGRADO, 3º ano, Turma C.

Integrantes: Gabriel Quirino Bressanin · Guilherme Barros Oliveira · Heitor Lupino · João Pedro Ferreira · Matheus Risso

A foto do medidor passa por um pipeline de dois estágios e sai como leitura numérica com índice de confiança:

```
foto ─► OpenCV (CLAHE, redimensionamento) ─► YOLOv8 (região do contador)
     ─► recorte + retificação ─► CRNN + CTC (dígitos) ─► pós-processamento (confiança, nº de dígitos)
     ─► validação (consumo, virada do contador, alertas) ─► SQLite ─► app web (computador e celular) / API REST
```

## O app de autoleitura

Funciona no navegador do **computador e do celular**, sem instalar nada:

1. **Imóveis e medidores**: cadastre o imóvel e os medidores (energia/água), com a tarifa da sua conta.
2. **Nova leitura**: fotografe o medidor (no celular, o botão de foto abre a câmera). O modelo lê o valor,
   mostra a confiança e o recorte do contador; dá para corrigir algum número antes de salvar.
3. **Validação automática** antes de salvar:
   - leitura **menor que a anterior** → bloqueia (erro de leitura ou indício de fraude), a não ser que o usuário confirme;
   - **virada do contador** (99999 → 00000) é tratada no cálculo do consumo;
   - consumo **50% acima da média** dos últimos meses (na água: possível vazamento), consumo zero, confiança baixa,
     mês repetido.
4. **Consumo**: consumo do mês, média, conta estimada, gráfico mês a mês, histórico em CSV e o recorte de cada
   leitura salvo para conferência.
5. **Desempenho do modelo**: métricas do conjunto de teste e a taxa de leituras que o usuário não precisou corrigir.

A **API REST** (`api.py`) mostra como uma concessionária integraria a leitura: `POST /leitura` com a foto
devolve o valor e a confiança em JSON; `GET /medidores/{id}/consumo` devolve o histórico.

## Estrutura

| Arquivo | O que faz | Seção da proposta |
|---|---|---|
| `autoleitura/preprocess.py` | Leitura, CLAHE, redimensionamento, recorte e retificação (deskew) | 5 (etapas 2 e 4), 6 |
| `autoleitura/detector.py` | Wrapper do YOLOv8 que devolve a caixa do contador | 5 (etapa 3), 7 |
| `autoleitura/recognizer.py` | CRNN (CNN + BiLSTM) com decodificação CTC e confiança por dígito | 5 (etapa 5), 7 |
| `autoleitura/pipeline.py` | Pipeline completo e desenho do resultado | 5 (etapas 6 e 9) |
| `autoleitura/consumo.py` | Consumo do mês, virada do contador, conta estimada e alertas | 5 (etapa 6), 11 |
| `autoleitura/db.py` | Imóveis, medidores, leituras e avaliações em SQLite | 5 (etapa 7) |
| `autoleitura/metrics.py` | Acurácia de leitura, acurácia por dígito, Levenshtein | 11 |
| `scripts/prepare_data.py` | Converte UFPR-AMR / Roboflow para os datasets do detector e do reconhecedor | 6 |
| `scripts/make_synthetic.py` | Gera medidores sintéticos para testar o pipeline | — |
| `train_detector.py` | Treina o YOLOv8 com augmentation | 7 |
| `train_recognizer.py` | Treina a CRNN com augmentation (só no treino) | 7 |
| `evaluate.py` | Avaliação ponta a ponta no teste e gravação no banco | 11 |
| `app.py` | App web de autoleitura (computador e celular) | 5 (etapa 8) |
| `rodar_app.py` | Abre o app e mostra o endereço para o celular | 5 (etapa 8) |
| `api.py` | API REST (FastAPI) para integração | 5 (etapa 9) |
| `notebooks/AutoLeitura_Colab.ipynb` | Treino completo no Google Colab | 4, 8 |

## Como rodar

### 1. Treinar (Google Colab)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joaopferreira7/AutoLeitura/blob/main/notebooks/AutoLeitura_Colab.ipynb)

1. Clique no botão acima, ative a GPU (`Ambiente de execução → Alterar o tipo → T4 GPU`) e execute as células.
   O notebook clona este repositório; o Drive guarda os datasets e os pesos treinados.
2. Para salvar alterações no notebook, use `Arquivo → Salvar uma cópia no GitHub`.
3. Ao final, baixe `MeuDrive/AutoLeitura/weights/detector.pt` e `recognizer.pt` para a pasta `weights/` local.

### 2. Rodar o app (computador e celular)
Coloque `detector.pt` e `recognizer.pt` em `weights/` e rode:
```bash
pip install -r requirements.txt
python rodar_app.py
```
O terminal mostra dois endereços: um para o computador (`http://localhost:8501`) e outro para o celular
(`http://<ip-do-computador>:8501`), que precisa estar **na mesma rede Wi-Fi**. Na primeira vez, o Windows pergunta
se o Python pode usar a rede: permita em **Redes privadas**. No celular, use "Adicionar à tela inicial" para ter
um ícone como o de um app.

A câmera ao vivo dentro da página só funciona no celular com HTTPS; pelo endereço da rede local, use o botão de
enviar foto, que já oferece a câmera do celular.

A API: `uvicorn api:app --host 0.0.0.0 --port 8000` (documentação em `http://localhost:8000/docs`).

A inferência roda em CPU sem problemas (~100–300 ms por foto com `yolov8n`/`yolov8s`).

### Treino local (alternativa ao Colab)
Se algum integrante tiver GPU NVIDIA, dá para treinar localmente instalando o PyTorch com CUDA
(veja pytorch.org) e rodando os mesmos comandos do notebook:
```bash
python scripts/prepare_data.py --source ufpr --input data/raw/ufpr
python train_detector.py --model yolov8s.pt --epochs 100 --device 0
python train_recognizer.py --epochs 80 --workers 6
python evaluate.py --errors erros_teste.csv
```

## Dados

- **UFPR-AMR** (principal): 2.000 fotos de medidores de energia da Copel (2340×4160 a 3120×4160 px),
  com split oficial 800/400/800. Cada foto tem um `.txt` com `camera`, `reading`, `position` (x y w h do contador)
  e `digit 1..5`. O parser (`parse_ufpr_txt`) foi validado nas 2.000 anotações, incluindo 8 arquivos
  em que a chave vem com erro de digitação no próprio dataset (`creading`, `qreading`).
  O link de download é enviado por e-mail pelos autores e é pessoal (licença só para uso acadêmico):
  **não coloquem o link nem os dados no repositório**. O notebook pede o link na hora.
  Atenção: o `AMR-Annotations.zip` que vem no mesmo e-mail **não** é o dataset; são só anotações de outros datasets (gás), sem imagens.
- **Roboflow Universe** (água): use um dataset exportado em *YOLOv8* com classes `0`–`9` (um box por dígito).
  A leitura é montada ordenando os dígitos da esquerda para a direita. Junte com o UFPR-AMR usando `--append`.
- **Sintético**: `scripts/make_synthetic.py` serve só como teste de fumaça do pipeline. Não usem como resultado do trabalho.

## Métricas (seção 11)

`evaluate.py` imprime e grava no banco:
- **mAP@0.5** do detector;
- **acurácia de leitura completa** (todos os dígitos certos), a métrica principal;
- **acurácia por dígito** e por classe (0–9);
- **distância de Levenshtein** média;
- acurácia do reconhecedor isolado (recortes anotados), para separar erro de detecção de erro de leitura.

## Resultados (UFPR-AMR, conjunto de teste com 800 fotos)

Treino local numa RTX 3060: detector YOLOv8s (100 épocas, ~25 min) e CRNN (80 épocas, ~12 min).

| Métrica | Resultado |
|---|---|
| Detecção do contador, mAP@0.5 | **99,4%** |
| Leitura completa (os 5 dígitos certos), pipeline inteiro | **89,0%** |
| Acurácia por dígito | **96,2%** |
| Distância de Levenshtein média | 0,16 |
| Reconhecedor isolado (recortes anotados) | 89,6% |
| Tempo por foto | 16 ms (GPU) |

**Análise dos erros (88 de 800):**
- o detector praticamente não erra: o pipeline completo (89,0%) fica quase igual à CRNN com recortes perfeitos (89,6%);
- quase metade dos erros (43) está no **último dígito**, o rolete que gira continuamente e muitas vezes aparece
  no meio da troca entre dois números;
- 69% dos erros já saem com confiança abaixo de 60%.

**Confiança na prática:** com o limiar do app (60%), **72% das fotos são aceitas direto, com 95,3% de acerto**;
as demais pedem que o usuário confira os números antes de salvar. Com limiar de 80%, as leituras aceitas
direto chegam a 99,3% de acerto, mas só 37% das fotos passam sem conferência.
