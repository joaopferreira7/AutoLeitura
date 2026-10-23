"""AutoLeitura: app de autoleitura de medidores de água e energia (computador e celular).

  python rodar_app.py          # abre no computador e mostra o endereço para o celular (mesma rede Wi-Fi)
  streamlit run app.py         # só no computador
"""
import hashlib
from datetime import date

import cv2
import pandas as pd
import streamlit as st

from autoleitura import config, consumo, db
from autoleitura.preprocess import load_image

st.set_page_config(page_title="AutoLeitura", page_icon="🔢", layout="centered")

TIPOS = {"energia": "⚡ Energia", "agua": "💧 Água"}
# valores só de exemplo: o usuário deve conferir na própria conta
TARIFA_EXEMPLO = {"energia": (0.90, 0.0), "agua": (6.00, 0.0)}


@st.cache_resource
def get_db():
    return db.connect()


@st.cache_resource(show_spinner="Carregando os modelos...")
def get_system():
    from autoleitura.pipeline import AutoLeitura
    return AutoLeitura()


def modelos_prontos():
    return config.DET_WEIGHTS.exists() and config.REC_WEIGHTS.exists()


def brl(v):
    return "—" if v is None else f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def nome_medidor(m):
    return f"{m['imovel']} · {TIPOS[m['tipo']]}" + (f" ({m['descricao']})" if m.get("descricao") else "")


def meses(n=12):
    hoje = date.today()
    out = []
    for i in range(n):
        y, mth = divmod(hoje.year * 12 + hoje.month - 1 - i, 12)
        out.append(f"{y:04d}-{mth + 1:02d}")
    return out


def fmt_mes(ref):
    return f"{ref[5:]}/{ref[:4]}" if ref else "—"


def mostrar_alertas(alertas):
    for a in alertas:
        {"erro": st.error, "aviso": st.warning, "info": st.info}[a.nivel](a.mensagem)


def escolher_medidor(label="Medidor", key=None):
    medidores = db.list_medidores(get_db())
    if not medidores:
        st.info("Cadastre um imóvel e um medidor para começar.")
        st.page_link(pagina_imoveis, label="Cadastrar imóvel e medidor", icon="🏠")
        st.stop()
    ids = [m["id"] for m in medidores]
    por_id = {m["id"]: m for m in medidores}
    padrao = st.session_state.get("medidor_atual")
    mid = st.selectbox(label, ids, index=ids.index(padrao) if padrao in ids else 0,
                       format_func=lambda i: nome_medidor(por_id[i]), key=key)
    st.session_state.medidor_atual = mid
    return por_id[mid]


# ====================================================================== páginas
def page_nova_leitura():
    st.title("📷 Nova leitura")
    medidor = escolher_medidor()
    un = consumo.UNIDADE[medidor["tipo"]]
    con = get_db()

    if not modelos_prontos():
        st.error(f"Os modelos ainda não foram treinados. Coloque `detector.pt` e `recognizer.pt` em `{config.WEIGHTS_DIR}`.")
        return
    if medidor["tipo"] == "agua":
        st.caption("ℹ️ O modelo foi treinado com medidores de energia (UFPR-AMR); em medidores de água, confira a leitura.")

    referencia = st.selectbox("Mês de referência", meses(), format_func=fmt_mes)

    st.session_state.setdefault("upload_n", 0)
    if st.toggle("Usar a câmera dentro da página", help="No celular, o botão abaixo já permite tirar a foto pela câmera."):
        foto = st.camera_input("Foto do medidor", key=f"cam_{st.session_state.upload_n}")
    else:
        foto = st.file_uploader("Foto do medidor", type=["jpg", "jpeg", "png", "webp", "bmp"],
                                key=f"up_{st.session_state.upload_n}",
                                help="No celular, toque em Browse files e escolha a câmera para fotografar na hora.")
    if foto is None:
        st.caption("Dica: fotografe de frente, com o contador ocupando boa parte da foto e sem reflexo.")
        return

    # O Streamlit reexecuta o script a cada interação; só roda o modelo de novo se a foto mudou.
    dados = foto.getvalue()
    chave = hashlib.md5(dados).hexdigest()
    if st.session_state.get("foto_chave") != chave:
        try:
            img = load_image(dados)
        except ValueError as e:
            st.error(str(e))
            return
        with st.spinner("Lendo o medidor..."):
            res = get_system().read(img)
        from autoleitura.pipeline import draw_result
        st.session_state.update(foto_chave=chave, res=res,
                                anotada=cv2.cvtColor(draw_result(img, res), cv2.COLOR_BGR2RGB))
    res = st.session_state.res

    if res["reading"] is None:
        st.image(st.session_state.anotada, width="stretch")
        st.error("Não encontrei o contador na foto. Tente outra foto, mais de perto e de frente para o medidor.")
        return

    c1, c2 = st.columns(2)
    c1.metric("Leitura do modelo", res["reading"])
    c2.metric("Confiança", f"{res['confidence'] * 100:.0f}%")
    st.image(cv2.cvtColor(res["crop"], cv2.COLOR_BGR2RGB), caption="Contador encontrado", width="stretch")
    if res["warning"]:
        st.warning(res["warning"])
    with st.expander("Ver foto completa e detalhes"):
        st.image(st.session_state.anotada, width="stretch")
        st.caption("Leitores: " + " · ".join(f"{k}: {v or '—'}" for k, v in res.get("leitores", {}).items()))
        st.caption("Confiança por dígito: " + "  ".join(
            f"{d}={c * 100:.0f}%" for d, c in zip(res["reading"], res["digit_confidences"])))
        st.caption(f"Detecção do contador: {res['det_confidence'] * 100:.0f}% · Tempo: {res['time_ms']:.0f} ms")

    valor = st.text_input("Leitura (corrija se algum número estiver errado)", value=res["reading"],
                          max_chars=config.NUM_DIGITS, key=f"valor_{chave}").strip()

    hist = consumo.historico(medidor, db.list_readings(con, medidor["id"]))
    anteriores = [h for h in hist if h["referencia"] < referencia]
    anterior = anteriores[-1]["valor"] if anteriores else medidor.get("leitura_inicial")
    alertas = consumo.validar(
        valor, medidor["tipo"],
        confianca=res["confidence"] if valor == res["reading"] else None,
        anterior=anterior,
        consumos_anteriores=[h["consumo"] for h in anteriores],
        referencia_repetida=any(h["referencia"] == referencia for h in hist),
    )
    mostrar_alertas(alertas)

    c = consumo.consumo(anterior, valor) if valor.isdigit() else None
    c1, c2 = st.columns(2)
    c1.metric(f"Consumo de {fmt_mes(referencia)}", "—" if c is None else f"{c} {un}",
              help=f"Leitura anterior: {anterior}" if anterior else None)
    c2.metric("Conta estimada", brl(consumo.conta_estimada(c, medidor["tarifa"], medidor["taxa_fixa"])))

    bloqueado = any(a.nivel == "erro" for a in alertas)
    confirmo = st.checkbox("Conferi com o medidor e a leitura está correta") if bloqueado else True
    if st.button("Salvar leitura", type="primary", width="stretch", disabled=not (valor.isdigit() and confirmo)):
        recorte = cv2.imencode(".jpg", res["crop"], [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        db.save_reading(con, res, medidor_id=medidor["id"], referencia=referencia, valor_real=valor,
                        arquivo=getattr(foto, "name", "camera"), tipo_medidor=medidor["tipo"], recorte_jpg=recorte)
        st.session_state.upload_n += 1          # limpa o campo de foto para a próxima leitura
        st.session_state.pop("foto_chave", None)
        st.toast(f"Leitura {valor} de {fmt_mes(referencia)} salva!", icon="✅")
        st.rerun()


def page_consumo():
    st.title("📊 Consumo")
    medidor = escolher_medidor()
    un = consumo.UNIDADE[medidor["tipo"]]
    con = get_db()
    hist = consumo.historico(medidor, db.list_readings(con, medidor["id"]))
    if not hist:
        st.info("Nenhuma leitura deste medidor ainda.")
        st.page_link(pagina_leitura, label="Fazer a primeira leitura", icon="📷")
        return

    ultimo = hist[-1]
    consumos = [h["consumo"] for h in hist if h["consumo"] is not None]
    c1, c2 = st.columns(2)
    c1.metric("Última leitura", ultimo["valor"], help=f"Referência {fmt_mes(ultimo['referencia'])}")
    c2.metric(f"Consumo de {fmt_mes(ultimo['referencia'])}",
              "—" if ultimo["consumo"] is None else f"{ultimo['consumo']} {un}",
              delta=None if len(consumos) < 2 or ultimo["consumo"] is None else f"{ultimo['consumo'] - consumos[-2]:+d} {un}",
              delta_color="inverse")
    c1, c2 = st.columns(2)
    m = consumo.media(consumos)
    c1.metric("Média mensal", "—" if m is None else f"{m:.0f} {un}")
    c2.metric("Conta estimada do mês", brl(ultimo["conta"]))

    # alertas do último mês (consumo fora do padrão, consumo zero, leitura menor que a anterior)
    anterior = hist[-2]["valor"] if len(hist) >= 2 else medidor.get("leitura_inicial")
    if anterior is not None:
        mostrar_alertas([a for a in consumo.validar(ultimo["valor"], medidor["tipo"], anterior=anterior,
                                                    consumos_anteriores=[h["consumo"] for h in hist[:-1]])
                         if a.nivel != "info"])

    if consumos:
        df_graf = pd.DataFrame([{"Mês": fmt_mes(h["referencia"]), f"Consumo ({un})": h["consumo"]}
                                for h in hist if h["consumo"] is not None]).set_index("Mês")
        st.bar_chart(df_graf, y_label=un)

    tabela = pd.DataFrame([{
        "Mês": fmt_mes(h["referencia"]),
        "Leitura": h["valor"],
        f"Consumo ({un})": h["consumo"],
        "Conta estimada": brl(h["conta"]),
        "Lida pelo modelo": h["leitura"],
        "Corrigida": "sim" if h["valor_real"] and h["valor_real"] != h["leitura"] else "",
        "Confiança": None if h["confianca"] is None else round(h["confianca"] * 100),
        "id": h["id"],
    } for h in reversed(hist)])
    st.dataframe(tabela.drop(columns=["id"]), width="stretch", hide_index=True,
                 column_config={"Confiança": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100)})

    st.download_button("Baixar histórico (CSV)", tabela.drop(columns=["id"]).to_csv(index=False, sep=";").encode("utf-8-sig"),
                       file_name=f"autoleitura_{medidor['tipo']}_{medidor['id']}.csv", mime="text/csv", width="stretch")

    with st.expander("Conferir ou excluir uma leitura"):
        lid = st.selectbox("Leitura", tabela["id"].tolist(),
                           format_func=lambda i: next(f"{r['Mês']} · {r['Leitura']}" for _, r in tabela.iterrows() if r["id"] == i))
        rec = db.get_recorte(con, lid)
        if rec:
            st.image(rec, caption="Recorte salvo no momento da leitura", width="stretch")
        with st.popover("Excluir esta leitura"):
            if st.button("Sim, excluir", key=f"del_l_{lid}", type="primary"):
                db.delete_reading(con, lid)
                st.rerun()


def page_imoveis():
    st.title("🏠 Imóveis e medidores")
    con = get_db()

    with st.expander("➕ Novo imóvel", expanded=not db.list_imoveis(con)):
        with st.form("novo_imovel", clear_on_submit=True):
            nome = st.text_input("Nome", placeholder="Ex.: Minha casa, Apartamento 12")
            endereco = st.text_input("Endereço (opcional)")
            if st.form_submit_button("Cadastrar imóvel", type="primary") and nome.strip():
                db.add_imovel(con, nome.strip(), endereco.strip() or None)
                st.rerun()

    imoveis = db.list_imoveis(con)
    if not imoveis:
        return

    with st.expander("➕ Novo medidor", expanded=not db.list_medidores(con)):
        imovel_id = st.selectbox("Imóvel", [i["id"] for i in imoveis],
                                 format_func=lambda i: next(x["nome"] for x in imoveis if x["id"] == i))
        tipo = st.radio("Tipo", list(TIPOS), format_func=TIPOS.get, horizontal=True)
        tarifa_ex, taxa_ex = TARIFA_EXEMPLO[tipo]
        with st.form("novo_medidor", clear_on_submit=True):
            descricao = st.text_input("Descrição (opcional)", placeholder="Ex.: relógio da frente")
            un = consumo.UNIDADE[tipo]
            tarifa = st.number_input(f"Tarifa (R$ por {un})", min_value=0.0, value=tarifa_ex, step=0.01, format="%.4f",
                                     help="Valor de exemplo: confira na sua conta (valor total ÷ consumo).")
            taxa = st.number_input("Taxa fixa mensal (R$)", min_value=0.0, value=taxa_ex, step=1.0,
                                   help="Iluminação pública, taxa mínima, esgoto etc.")
            inicial = st.text_input("Leitura atual do medidor (opcional)", max_chars=config.NUM_DIGITS,
                                    help="Se preencher, o consumo já é calculado a partir da primeira foto.")
            if st.form_submit_button("Cadastrar medidor", type="primary"):
                if inicial and not inicial.isdigit():
                    st.error("A leitura atual deve conter só números.")
                else:
                    db.add_medidor(con, imovel_id, tipo, descricao.strip() or None, tarifa, taxa, inicial or None)
                    st.rerun()

    for im in imoveis:
        st.subheader(im["nome"])
        if im["endereco"]:
            st.caption(im["endereco"])
        meds = db.list_medidores(con, im["id"])
        if not meds:
            st.caption("Nenhum medidor cadastrado.")
        for m in meds:
            un = consumo.UNIDADE[m["tipo"]]
            with st.expander(f"{TIPOS[m['tipo']]}" + (f" · {m['descricao']}" if m["descricao"] else "")
                             + f" · {brl(m['tarifa'])}/{un}"):
                with st.form(f"med_{m['id']}"):
                    tarifa = st.number_input(f"Tarifa (R$ por {un})", min_value=0.0, value=float(m["tarifa"] or 0),
                                             step=0.01, format="%.4f")
                    taxa = st.number_input("Taxa fixa mensal (R$)", min_value=0.0, value=float(m["taxa_fixa"] or 0), step=1.0)
                    if st.form_submit_button("Salvar"):
                        db.update_medidor(con, m["id"], tarifa=tarifa, taxa_fixa=taxa)
                        st.rerun()
                with st.popover("Excluir medidor"):
                    st.write("Isso apaga o medidor **e todas as leituras dele**.")
                    if st.button("Sim, excluir", key=f"del_m_{m['id']}", type="primary"):
                        db.delete_medidor(con, m["id"])
                        st.rerun()
        with st.popover(f"Excluir imóvel “{im['nome']}”"):
            st.write("Isso apaga o imóvel, **todos os medidores e todas as leituras**.")
            if st.button("Sim, excluir", key=f"del_i_{im['id']}", type="primary"):
                db.delete_imovel(con, im["id"])
                st.rerun()


def page_modelo():
    st.title("🧠 Desempenho do modelo")
    con = get_db()
    ev = db.list_evaluations(con)
    st.subheader("Avaliação no conjunto de teste (UFPR-AMR)")
    if ev:
        df = pd.DataFrame(ev)
        for col in ("reading_acc", "digit_acc", "det_map50"):
            df[col] = (df[col] * 100).round(2)
        st.dataframe(df[["criado_em", "conjunto", "n", "reading_acc", "digit_acc", "mean_lev", "det_map50"]].rename(columns={
            "criado_em": "Data", "conjunto": "Conjunto", "n": "Imagens", "reading_acc": "Acurácia de leitura (%)",
            "digit_acc": "Acurácia por dígito (%)", "mean_lev": "Levenshtein médio", "det_map50": "mAP@0.5 (%)"}),
            width="stretch", hide_index=True)
    else:
        st.info("Nenhuma avaliação ainda. Rode `python evaluate.py`.")

    st.subheader("Em uso real")
    # só leituras feitas pelo modelo (as de demonstração não têm confiança)
    rows = [r for r in db.list_readings(con) if r["valor_real"] and r["confianca"] is not None]
    if not rows:
        st.caption("As leituras salvas no app aparecem aqui, comparando o valor do modelo com o confirmado.")
        return
    acertos = sum(r["leitura"] == r["valor_real"] for r in rows)
    c1, c2 = st.columns(2)
    c1.metric("Leituras confirmadas", len(rows))
    c2.metric("Lidas corretamente pelo modelo", f"{acertos / len(rows) * 100:.0f}%",
              help="Leituras em que o usuário não precisou corrigir nenhum número.")


pagina_leitura = st.Page(page_nova_leitura, title="Nova leitura", icon="📷", default=True)
pagina_consumo = st.Page(page_consumo, title="Consumo", icon="📊")
pagina_imoveis = st.Page(page_imoveis, title="Imóveis e medidores", icon="🏠")
pagina_modelo = st.Page(page_modelo, title="Desempenho do modelo", icon="🧠")
st.navigation([pagina_leitura, pagina_consumo, pagina_imoveis, pagina_modelo]).run()
