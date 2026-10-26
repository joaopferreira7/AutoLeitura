# Script auxiliar dos testes: roda o app.py abrindo direto na página pedida em AUTOLEITURA_TESTE_PAGINA.
# O AppTest do Streamlit só troca de página quando as páginas são arquivos; as do app são funções.
import os
import runpy
from pathlib import Path

import streamlit as st

if not hasattr(st, "_navigation_original"):
    st._navigation_original = st.navigation


def _navegacao(paginas, **kwargs):
    alvo = os.environ.get("AUTOLEITURA_TESTE_PAGINA", "")
    escolhidas = [p for p in paginas if p.url_path == alvo] or paginas[:1]
    return st._navigation_original(escolhidas, **kwargs)


st.navigation = _navegacao
runpy.run_path(str(Path(__file__).resolve().parent.parent / "app.py"), run_name="__main__")
