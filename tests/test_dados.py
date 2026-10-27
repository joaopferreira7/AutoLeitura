"""Pré-processamento das fotos, leitura das anotações, preparação dos datasets e download dos pesos."""
import io

import cv2
import numpy as np
import pytest
from PIL import Image

from autoleitura import config, pesos
from autoleitura.preprocess import crop_box, load_image, prepare_crop, shrink_crop, to_recognizer_input


def jpeg(largura, altura, orientacao=None):
    img = Image.new("RGB", (largura, altura), (200, 50, 50))
    buf = io.BytesIO()
    exif = Image.Exif()
    if orientacao:
        exif[274] = orientacao
    img.save(buf, "JPEG", exif=exif.tobytes())
    return buf.getvalue()


def test_foto_de_celular_deitada_e_endireitada():
    # orientação EXIF 6 = o celular gravou deitado e marcou "girar 90°" nos metadados
    img = load_image(jpeg(400, 300, orientacao=6))
    assert img.shape[:2] == (400, 300)


def test_foto_sem_exif_fica_como_esta():
    assert load_image(jpeg(400, 300)).shape[:2] == (300, 400)


def test_png_com_transparencia_vira_bgr():
    buf = io.BytesIO()
    Image.new("RGBA", (50, 40), (0, 0, 0, 0)).save(buf, "PNG")
    img = load_image(buf.getvalue())
    assert img.shape == (40, 50, 3) and img.dtype == np.uint8


@pytest.mark.parametrize("dados", [b"", b"nao e uma imagem", b"\xff\xd8\xff\xe0quebrado"])
def test_arquivo_invalido_vira_erro_claro(dados):
    with pytest.raises(ValueError, match="Não foi possível ler"):
        load_image(dados)


def test_recorte_respeita_as_bordas():
    img = np.zeros((100, 200, 3), np.uint8)
    assert crop_box(img, (-50, -50, 400, 400)).shape == (100, 200, 3)
    assert crop_box(img, (10, 10, 60, 30)).shape[:2] == (23, 58)      # margem de 8%: y 8..31, x 6..64


def test_reducao_do_recorte_e_entrada_da_crnn():
    grande = np.random.default_rng(0).integers(0, 255, (200, 1000, 3), dtype=np.uint8)
    assert shrink_crop(grande).shape[1] == 2 * config.REC_WIDTH
    pequeno = grande[:20, :100]
    assert shrink_crop(pequeno) is pequeno
    x = to_recognizer_input(prepare_crop(grande))
    assert x.shape == (1, config.REC_HEIGHT, config.REC_WIDTH) and 0 <= x.min() and x.max() <= 1


def test_anotacao_do_ufpr_inclusive_com_erro_de_digitacao(tmp_path):
    from prepare_data import parse_ufpr_txt
    txt = tmp_path / "meter0001.txt"
    txt.write_text("camera: LG G3 D855\nCreading: 03218\nposition: 100 200 300 80\n"
                   + "".join(f"\tdigit {i + 1}: {100 + 60 * i} 205 50 70\n" for i in range(5)))
    box, leitura, digitos = parse_ufpr_txt(txt)
    assert box == (100, 200, 400, 280) and leitura == "03218" and len(digitos) == 5
    assert digitos[0] == (100, 205, 150, 275)


def test_anotacao_incompleta_e_ignorada(tmp_path):
    from prepare_data import parse_ufpr_txt
    txt = tmp_path / "x.txt"
    txt.write_text("camera: X\nposition: 1 2 3 4\n")          # sem leitura
    assert parse_ufpr_txt(txt) is None


def test_preparacao_de_ponta_a_ponta_em_pastas_temporarias(tmp_path, monkeypatch):
    """Gera medidores sintéticos no formato do UFPR-AMR e prepara os 3 datasets, sem tocar em data/."""
    import csv

    import make_synthetic
    import prepare_data

    bruto = tmp_path / "raw"
    for split in ("training", "validation", "testing"):
        (bruto / split).mkdir(parents=True)
    rng = np.random.default_rng(1)
    for i, split in enumerate(["training"] * 4 + ["validation"] * 2 + ["testing"] * 2):
        img, (x1, y1, x2, y2), leitura, digitos = make_synthetic.make_one(rng)
        cv2.imwrite(str(bruto / split / f"meter{i:04d}.jpg"), img)
        linhas = [f"position: {int(x1)} {int(y1)} {int(x2 - x1)} {int(y2 - y1)}", f"reading: {leitura}"]
        linhas += [f"digit {k + 1}: {int(a)} {int(b)} {int(c)} {int(d)}" for k, (a, b, c, d) in enumerate(digitos)]
        (bruto / split / f"meter{i:04d}.txt").write_text("\n".join(linhas))

    for nome in ("DET_DIR", "REC_DIR", "DIG_DIR"):
        monkeypatch.setattr(config, nome, tmp_path / nome.lower())
    amostras = prepare_data.load_ufpr(bruto)
    prepare_data.assign_splits(amostras)
    prepare_data.write_datasets(amostras, prefix="t", append=False)

    rotulos = list(csv.DictReader(open(config.REC_DIR / "labels.csv", encoding="utf-8")))
    assert [r["split"] for r in rotulos].count("train") == 4 and len(rotulos) == 8
    assert len(list((config.DET_DIR / "labels" / "test").glob("*.txt"))) == 2
    # cada dígito anotado vira uma caixa com a sua classe, na ordem da leitura
    for r in rotulos:
        lbl = (config.DIG_DIR / "labels" / r["split"] / r["arquivo"].replace(".png", ".txt")).read_text().split("\n")
        assert "".join(linha.split()[0] for linha in lbl if linha) == r["leitura"]
    # rótulos YOLO normalizados entre 0 e 1
    for arq in (config.DET_DIR / "labels").rglob("*.txt"):
        valores = [float(v) for v in arq.read_text().split()[1:]]
        assert all(0 <= v <= 1 for v in valores)


def test_download_corrompido_e_recusado(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WEIGHTS_DIR", tmp_path)
    destino = tmp_path / "detector.pt"
    monkeypatch.setattr(pesos, "PESOS", [destino])
    monkeypatch.setattr(pesos.urllib.request, "urlretrieve", lambda url, arq: open(arq, "wb").write(b"lixo"))
    with pytest.raises(RuntimeError, match="SHA-256"):
        pesos.baixar(progresso=lambda _: None)
    assert not destino.exists() and not destino.with_suffix(".part").exists()


def test_peso_de_versao_anterior_e_trocado_so_quando_pedido(tmp_path, monkeypatch):
    destino = tmp_path / "detector.pt"
    destino.write_bytes(b"versao antiga")
    monkeypatch.setattr(config, "WEIGHTS_DIR", tmp_path)
    monkeypatch.setattr(pesos, "PESOS", [destino])
    monkeypatch.setattr(pesos, "SHA256", {"detector.pt": pesos.hashlib.sha256(b"versao nova").hexdigest()})
    monkeypatch.setattr(pesos.urllib.request, "urlretrieve", lambda url, arq: open(arq, "wb").write(b"versao nova"))
    pesos.baixar(progresso=lambda _: None)                     # localmente: não mexe
    assert destino.read_bytes() == b"versao antiga"
    assert pesos.desatualizados() == [destino]
    pesos.baixar(progresso=lambda _: None, atualizar=True)     # na nuvem: troca
    assert destino.read_bytes() == b"versao nova" and pesos.desatualizados() == []


def test_pesos_locais_conferem_com_o_release():
    faltando = pesos.faltando()
    if faltando:
        pytest.skip("pesos ausentes")
    for p in pesos.PESOS:
        assert pesos.sha256(p) == pesos.SHA256[p.name], f"{p.name} difere do Release em pesos.RELEASE_URL"


def test_endireitar_janela_girada():
    """Uma faixa clara girada 30° vira um recorte horizontal com o lado maior como largura."""
    from autoleitura.preprocess import endireitar
    img = np.zeros((400, 400, 3), np.uint8)
    caixa = cv2.boxPoints(((200, 200), (200, 50), 30)).astype(np.int32)
    cv2.fillPoly(img, [caixa], (255, 255, 255))
    rec = endireitar(img, caixa, folga=0)
    assert rec.shape[1] > rec.shape[0] * 3                 # bem mais largo que alto
    assert rec.mean() > 230                                # quase todo dentro da faixa clara
