# Imagem do AutoLeitura para o Hugging Face Spaces (SDK Docker, CPU gratuita)
FROM python:3.11-slim

# bibliotecas de sistema que o OpenCV (puxado pelo Ultralytics) precisa
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# o Spaces roda o container com o usuário de UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# o disco do Space é apagado a cada reinício; o banco começa com um imóvel de demonstração
ENV AUTOLEITURA_DB=/home/user/app/autoleitura.db
EXPOSE 7860
CMD python scripts/criar_demo.py && streamlit run app.py \
    --server.port 7860 --server.address 0.0.0.0 --server.headless true \
    --server.enableXsrfProtection false --server.enableCORS false
