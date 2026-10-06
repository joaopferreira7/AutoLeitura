"""Interface web do AutoLeitura (Streamlit): upload da foto -> leitura + confiança.

  streamlit run app.py
"""
import hashlib

import cv2
import pandas as pd
import streamlit as st

from autoleitura import config, db
from autoleitura.pipeline import AutoLeitura, draw_result
from autoleitura.preprocess import load_image

st.set_page_config(page_title="AutoLeitura", page_icon="🔢", layout="wide")


@st.cache_resource
def get_system():
    return AutoLeitura()


@st.cache_resource
def get_db():
    return db.connect()


def page_leitura():
    st.title("AutoLeitura")
    st.caption("Leitura automática de medidores de água e energia por visão computacional")

    if not (config.DET_WEIGHTS.exists() and config.REC_WEIGHTS.exists()):
        st.error(f"Pesos não encontrados. Treine os modelos e coloque `detector.pt` e `recognizer.pt` "
                 f"em `{config.WEIGHTS_DIR}`.")
        return

    col_in, col_opts = st.columns([3, 1])
    with col_opts:
        tipo = st.radio("Tipo de medidor", ["energia", "agua"], horizontal=True)
        usar_camera = st.toggle("Usar webcam")
    with col_in:
        up = st.camera_input("Foto do medidor") if usar_camera else \
            st.file_uploader("Foto do medidor", type=["jpg", "jpeg", "png", "bmp"])
    if up is None:
        return

    # O Streamlit reexecuta o script a cada interação; só lê e grava de novo se a foto mudou.
    data = up.getvalue()
    key = hashlib.md5(data).hexdigest()
    img = load_image(data)
    if st.session_state.get("ultima_foto") != key:
        with st.spinner("Lendo o medidor..."):
            res = get_system().read(img)
        rid = db.save_reading(get_db(), res, arquivo=getattr(up, "name", "webcam"), tipo_medidor=tipo)
        st.session_state.update(ultima_foto=key, res=res, rid=rid)
    res, rid = st.session_state.res, st.session_state.rid

    left, right = st.columns([3, 2])
    with left:
        st.image(cv2.cvtColor(draw_result(img, res), cv2.COLOR_BGR2RGB), width="stretch")
    with right:
        if res["reading"] is None:
            st.error(res["warning"])
            return
        st.metric("Leitura", res["reading"])
        st.metric("Confiança", f"{res['confidence'] * 100:.1f}%")
        st.progress(min(1.0, res["confidence"]))
        if res["warning"]:
            st.warning(res["warning"])
        elif res["confidence"] < 0.6:
            st.warning("Confiança baixa: recomenda-se conferir manualmente.")
        st.image(cv2.cvtColor(res["crop"], cv2.COLOR_BGR2RGB), caption="Recorte do contador", width="stretch")
        st.caption("Confiança por dígito: " + "  ".join(
            f"{d}={c * 100:.0f}%" for d, c in zip(res["reading"], res["digit_confidences"])))
        st.caption(f"Detecção: {res['det_confidence'] * 100:.1f}% · Tempo: {res['time_ms']:.0f} ms · Registro #{rid}")

        with st.form(f"gt_{rid}"):
            real = st.text_input("Valor real (opcional, para medir a acurácia em uso)", value=res["reading"])
            if st.form_submit_button("Confirmar leitura"):
                db.set_ground_truth(get_db(), rid, real.strip())
                st.success("Leitura confirmada.")


def page_historico():
    st.title("Histórico e desempenho")
    con = get_db()

    ev = db.list_evaluations(con)
    st.subheader("Avaliações no conjunto de teste")
    if ev:
        st.dataframe(pd.DataFrame(ev).drop(columns=["detalhes"]), width="stretch", hide_index=True)
    else:
        st.info("Nenhuma avaliação ainda. Rode `python evaluate.py`.")

    rows = db.list_readings(con)
    st.subheader("Leituras realizadas")
    if not rows:
        st.info("Nenhuma leitura ainda.")
        return
    df = pd.DataFrame(rows)
    conf = df.dropna(subset=["valor_real"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Leituras", len(df))
    c2.metric("Confiança média", f"{df['confianca'].mean() * 100:.1f}%")
    if len(conf):
        c3.metric("Acurácia (leituras confirmadas)", f"{(conf['leitura'] == conf['valor_real']).mean() * 100:.1f}%")
    st.dataframe(df, width="stretch", hide_index=True)


pg = st.navigation([st.Page(page_leitura, title="Ler medidor", icon="📷"),
                    st.Page(page_historico, title="Histórico", icon="📊")])
pg.run()
