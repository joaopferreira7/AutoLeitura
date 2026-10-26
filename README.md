# AutoLeitura

Leitura automática de medidores de consumo (água e energia) por visão computacional.
Projeto de Visão Computacional, Ciência da Computação, UNISAGRADO, 3º ano, Turma C.

Integrantes: Gabriel Quirino Bressanin · Guilherme Barros Oliveira · Heitor Lupino · João Pedro Ferreira · Matheus Risso

**App no ar:** https://autoleitura.streamlit.app (abre no computador e no celular)

A foto do medidor passa por um pipeline de dois estágios e sai como leitura numérica com índice de confiança:

```
foto ─► OpenCV (EXIF, redimensionamento) ─► YOLOv8 (região do contador) ─► recorte
     ─► YOLOv8 de dígitos (0–9)  ┐
     ─► CRNN + CTC (sequência)    ┴► combinação: leitura do YOLO, confiança alta só se os dois concordarem
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
| `autoleitura/digits.py` | YOLOv8 de 10 classes que detecta e classifica cada dígito no recorte | 5 (etapa 5), 7 |
| `autoleitura/pipeline.py` | Pipeline completo, combinação dos dois leitores e desenho do resultado | 5 (etapas 6 e 9) |
| `autoleitura/consumo.py` | Consumo do mês, virada do contador, conta estimada e alertas | 5 (etapa 6), 11 |
| `autoleitura/db.py` | Imóveis, medidores, leituras e avaliações em SQLite | 5 (etapa 7) |
| `autoleitura/metrics.py` | Acurácia de leitura, acurácia por dígito, Levenshtein | 11 |
| `scripts/prepare_data.py` | Converte UFPR-AMR / Roboflow para os datasets do detector e do reconhecedor | 6 |
| `scripts/make_synthetic.py` | Gera medidores sintéticos para testar o pipeline | — |
| `scripts/criar_demo.py` | Cria um imóvel de demonstração com histórico (apresentação) | — |
| `scripts/publicar_hf.py` | Publica o app no Hugging Face Spaces | — |
| `train_detector.py` | Treina o YOLOv8 com augmentation | 7 |
| `train_recognizer.py` | Treina a CRNN com augmentation (só no treino) | 7 |
| `train_digits.py` | Treina o YOLOv8 de dígitos | 7 |
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
Os pesos treinados ficam no [Release `modelos-v1`](https://github.com/joaopferreira7/AutoLeitura/releases/tag/modelos-v1)
e o app os baixa sozinho para `weights/` na primeira execução (~55 MB).
```bash
pip install -r requirements.txt
python rodar_app.py              # adicione --demo para criar os imóveis de demonstração
```
O terminal mostra dois endereços: um para o computador (`http://localhost:8501`) e outro para o celular
(`http://<ip-do-computador>:8501`), que precisa estar **na mesma rede Wi-Fi**. Na primeira vez, o Windows pergunta
se o Python pode usar a rede: permita em **Redes privadas**. No celular, use "Adicionar à tela inicial" para ter
um ícone como o de um app.

A câmera ao vivo dentro da página só funciona no celular com HTTPS; pelo endereço da rede local, use o botão de
enviar foto, que já oferece a câmera do celular.

**Link público temporário** (plano B, funciona até no 4G): `python rodar_app.py --publico` abre um túnel gratuito da
Cloudflare e mostra um link `https://...trycloudflare.com`. Precisa do cloudflared, instalado uma vez com
`winget install --id Cloudflare.cloudflared`. O link só funciona enquanto o programa estiver aberto.

A API: `uvicorn api:app --host 0.0.0.0 --port 8000` (documentação em `http://localhost:8000/docs`).

### 3. Publicação na nuvem (Streamlit Community Cloud)
O app em https://autoleitura.streamlit.app é publicado direto deste repositório (branch `main`, arquivo `app.py`,
Python 3.12) e se atualiza sozinho a cada push. Na nuvem:
- o `requirements.txt` instala o PyTorch só de CPU e o `packages.txt` instala as bibliotecas de sistema do OpenCV;
- os pesos são baixados do Release na primeira execução;
- o banco começa com os imóveis de demonstração e é apagado quando o app reinicia (não guarde dados reais lá).

A pasta `deploy/huggingface` tem uma versão em Docker para o Hugging Face Spaces, que hoje exige a assinatura PRO.

A inferência roda em CPU sem problemas (~100–300 ms por foto com `yolov8n`/`yolov8s`).

### Treino local (alternativa ao Colab)
Se algum integrante tiver GPU NVIDIA, dá para treinar localmente instalando o PyTorch com CUDA
(veja pytorch.org) e rodando os mesmos comandos do notebook:
```bash
python scripts/prepare_data.py --source ufpr --input data/raw/ufpr
python train_detector.py --model yolov8s.pt --epochs 100 --device 0
python train_recognizer.py --epochs 80 --workers 6
python train_digits.py --epochs 150 --device 0
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

Treino local numa RTX 3060: detector do contador YOLOv8s (100 épocas, ~25 min), CRNN (80 épocas, ~12 min)
e YOLOv8s de dígitos (150 épocas, ~15 min). A regra de combinação dos dois leitores foi escolhida no conjunto
de **validação**; o teste foi usado uma única vez, no final.

| Métrica | Só CRNN | CRNN + YOLO de dígitos (final) |
|---|---|---|
| Detecção do contador, mAP@0.5 | 99,4% | 99,4% |
| **Leitura completa** (os 5 dígitos certos), pipeline inteiro | 89,0% | **93,4%** |
| Acurácia por dígito | 96,2% | **97,7%** |
| Distância de Levenshtein média | 0,16 | **0,11** |
| Fotos aceitas direto pelo app (confiança ≥ 60%) | 72% | **87%** |
| Acerto nas fotos aceitas direto | 95,3% | **98,7%** |
| Tempo por foto (GPU) | 16 ms | 31 ms |

Leitores isolados, sobre os recortes anotados: CRNN 89,6% e YOLO de dígitos 92,4%.

**Por que a combinação funciona:** a maior parte dos erros está no último dígito, o rolete que gira continuamente
e aparece no meio da troca entre dois números. Os dois leitores erram de formas diferentes: na validação,
quando **concordam** (88,8% das fotos) a leitura está certa em **99,2%** dos casos. Por isso a confiança
sobe quando concordam e cai quando discordam, e o app pede conferência só nesses casos.

A meta da proposta (seção 11) era acurácia de leitura completa "em torno de 90% ou mais"; os trabalhos de
referência no UFPR-AMR relatam de 94% a 97%.

## Testes

[![Testes](https://github.com/joaopferreira7/AutoLeitura/actions/workflows/testes.yml/badge.svg)](https://github.com/joaopferreira7/AutoLeitura/actions/workflows/testes.yml)

```bash
pip install -r requirements-dev.txt
python -m pytest
```

67 testes em `tests/`: regras de consumo e alertas, banco (inclusive gravações simultâneas de vários usuários e
migração de bancos antigos), pré-processamento (fotos de celular giradas pelo EXIF, arquivos inválidos),
preparação dos dados em pastas temporárias, verificação do SHA-256 dos pesos, regra de combinação dos leitores,
fotos sem medidor, leituras simultâneas, API e as quatro telas do app. Os testes nunca tocam em `data/` nem no
banco real. Quem tem o UFPR-AMR em `data/raw/ufpr` roda também um teste de regressão com 30 fotos reais
(acerto mínimo de 85%). No GitHub, os testes rodam a cada push e toda segunda-feira, para avisar se uma
atualização de biblioteca quebrou algo.

## Riscos conhecidos e hipóteses

| Risco | Probabilidade | O que acontece | Mitigação atual / o que fazer |
|---|---|---|---|
| Fotos fora do padrão do UFPR-AMR (outros modelos de medidor, sol forte, vidro sujo, foto de longe) | Alta | Acurácia menor que os 93,4% medidos | A confiança cai quando os leitores discordam e o app pede conferência; medir com fotos reais de usuários |
| Medidores de água | Alta | O modelo só viu medidores de energia; contadores com m³, casas decimais e dígitos vermelhos | Aviso na tela; treinar com um dataset de água (Roboflow ou fotos próprias) |
| Erro com confiança alta (os dois leitores erram igual) | Baixa (9 de 800 no teste) | Leitura errada aceita sem conferência | Validação contra o mês anterior e a média; usuário pode corrigir no histórico |
| Atualização de bibliotecas (Streamlit, Ultralytics, PyTorch) | Média | App publicado para de abrir ou muda de comportamento | Streamlit e Ultralytics com versão fixa; testes semanais no GitHub |
| Nuvem gratuita do Streamlit | Média | App "dorme" após dias sem uso; banco apagado a cada reinício; CPU e memória limitadas | Abrir o link antes da apresentação; não guardar dados reais; plano B com `rodar_app.py --publico` |
| Release `modelos-v1` apagado, renomeado ou repositório privado | Baixa | O app não consegue baixar os modelos | Não apagar o Release; pesos conferidos por SHA-256 |
| Muitos usuários ao mesmo tempo | Baixa (trabalho acadêmico) | Leituras em fila (uma por vez) e SQLite no limite | Para produção: banco hospedado (PostgreSQL) e mais instâncias |
| Rolete do último dígito em transição | Alta (metade dos erros) | Último dígito errado em 1 unidade | Conferência quando há discordância; mais exemplos de transição no treino |
| Licença do UFPR-AMR | Certa | Uso só acadêmico e não comercial, inclusive dos modelos treinados | Não usar comercialmente; para produto, treinar com dados próprios |
| Privacidade (LGPD) | Média em produção | As fotos e recortes podem mostrar endereço e número de série | Demo não guarda dados reais; em produção, consentimento e política de retenção |
| iPhone enviando foto HEIC da galeria | Baixa | Arquivo recusado | O Safari costuma converter para JPEG; fotografar pela câmera do app |
| API sem autenticação | Só se publicada | Qualquer um poderia enviar fotos | A API não está publicada; em produção, chave de acesso e limite de requisições |
