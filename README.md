# AutoLeitura

Leitura automática de medidores de consumo (água e energia) por visão computacional.
Projeto de Visão Computacional, Ciência da Computação, UNISAGRADO, 3º ano, Turma C.

Integrantes: Gabriel Quirino Bressanin · Guilherme Barros Oliveira · Heitor Lupino · João Pedro Ferreira · Matheus Risso

A foto do medidor passa por um pipeline de dois estágios e sai como leitura numérica com índice de confiança:

```
foto ─► OpenCV (CLAHE, redimensionamento) ─► YOLOv8 (região do contador)
     ─► recorte + retificação ─► CRNN + CTC (dígitos) ─► pós-processamento (confiança, nº de dígitos)
     ─► SQLite ─► interface Streamlit
```

## Estrutura

| Arquivo | O que faz | Seção da proposta |
|---|---|---|
| `autoleitura/preprocess.py` | Leitura, CLAHE, redimensionamento, recorte e retificação (deskew) | 5 (etapas 2 e 4), 6 |
| `autoleitura/detector.py` | Wrapper do YOLOv8 que devolve a caixa do contador | 5 (etapa 3), 7 |
| `autoleitura/recognizer.py` | CRNN (CNN + BiLSTM) com decodificação CTC e confiança por dígito | 5 (etapa 5), 7 |
| `autoleitura/pipeline.py` | Pipeline completo e desenho do resultado | 5 (etapas 6 e 9) |
| `autoleitura/db.py` | Tabelas `leituras` e `avaliacoes` em SQLite | 5 (etapa 7) |
| `autoleitura/metrics.py` | Acurácia de leitura, acurácia por dígito, Levenshtein | 11 |
| `scripts/prepare_data.py` | Converte UFPR-AMR / Roboflow para os datasets do detector e do reconhecedor | 6 |
| `scripts/make_synthetic.py` | Gera medidores sintéticos para testar o pipeline | — |
| `train_detector.py` | Treina o YOLOv8 com augmentation | 7 |
| `train_recognizer.py` | Treina a CRNN com augmentation (só no treino) | 7 |
| `evaluate.py` | Avaliação ponta a ponta no teste e gravação no banco | 11 |
| `app.py` | Interface web (upload ou webcam → leitura + confiança, histórico) | 5 (etapa 8) |
| `notebooks/AutoLeitura_Colab.ipynb` | Treino completo no Google Colab | 4, 8 |

## Como rodar

### 1. Treinar (Google Colab)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joaopferreira7/AutoLeitura/blob/main/notebooks/AutoLeitura_Colab.ipynb)

1. Clique no botão acima, ative a GPU (`Ambiente de execução → Alterar o tipo → T4 GPU`) e execute as células.
   O notebook clona este repositório; o Drive guarda os datasets e os pesos treinados.
2. Para salvar alterações no notebook, use `Arquivo → Salvar uma cópia no GitHub`.
3. Ao final, baixe `MeuDrive/AutoLeitura/weights/detector.pt` e `recognizer.pt` para a pasta `weights/` local.

### 2. Rodar a interface (notebook do grupo)
```bash
pip install -r requirements.txt
streamlit run app.py
```

A inferência roda em CPU sem problemas (~100–300 ms por foto com `yolov8n`/`yolov8s`).

### Treino local (alternativa ao Colab)
Se algum integrante tiver GPU NVIDIA, dá para treinar localmente instalando o PyTorch com CUDA
(veja pytorch.org) e rodando os mesmos comandos do notebook:
```bash
python scripts/prepare_data.py --source ufpr --input data/raw/UFPR-AMR
python train_detector.py --model yolov8s.pt --epochs 100 --device 0
python train_recognizer.py --epochs 80
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
