"""Abre o app publicado num navegador e o acorda se estiver dormindo.

O Streamlit Community Cloud põe o app para dormir depois de algumas horas sem visitas; acordar
leva cerca de um minuto. O workflow .github/workflows/manter_acordado.yml roda este script a
cada 6 horas para o app estar pronto na hora da apresentação.

  pip install playwright && playwright install chromium
  python scripts/acordar_app.py https://autoleitura.streamlit.app
"""
import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "https://autoleitura.streamlit.app"
BOTAO = "button:has-text('get this app back up')"


def main():
    with sync_playwright() as p:
        nav = p.chromium.launch()
        pagina = nav.new_page()
        pagina.goto(URL, wait_until="domcontentloaded", timeout=90_000)
        pagina.wait_for_timeout(10_000)
        botao = pagina.locator(BOTAO)
        if botao.count():
            print("app dormindo: acordando...")
            botao.first.click()
        else:
            print("app acordado")
        # o app roda num iframe; espera o título da página inicial aparecer
        app = pagina.frame_locator("iframe[title='streamlitApp']")
        app.get_by_text("Nova leitura").first.wait_for(timeout=240_000)
        print("app pronto")
        nav.close()


if __name__ == "__main__":
    main()
