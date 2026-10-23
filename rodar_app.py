"""Abre o app no computador e mostra o endereço para acessar pelo celular.

  python rodar_app.py              # celular na mesma rede Wi-Fi
  python rodar_app.py --publico    # também gera um link https:// público (funciona até no 4G)
  python rodar_app.py --demo       # banco com os imóveis de demonstração (para a apresentação)

Na mesma rede: na primeira vez, o Windows pergunta se o Python pode se comunicar na rede;
permita em "Redes privadas".

Link público: usa o túnel rápido gratuito da Cloudflare (trycloudflare.com, sem cadastro).
Precisa do cloudflared instalado uma vez:  winget install --id Cloudflare.cloudflared
O link muda a cada execução e só funciona enquanto este programa estiver aberto. Qualquer
pessoa com o link acessa o app, então use só durante a apresentação.
"""
import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

PORTA = 8501
# 127.0.0.1 e não "localhost": no Windows, localhost pode resolver para IPv6 (::1),
# onde o Streamlit não escuta, e o túnel responderia 502
TUNEL = ["cloudflared", "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{PORTA}"]


def ip_local():
    # Abrir um socket UDP não envia nada pela rede; só descobre qual interface seria usada.
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def abrir_tunel():
    """Abre o túnel e imprime o link https:// assim que a Cloudflare informar."""
    if shutil.which("cloudflared") is None:
        print("  [link público] instale o cloudflared uma vez e rode de novo:\n"
              "                 winget install --id Cloudflare.cloudflared")
        return None
    tunel = subprocess.Popen(TUNEL, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace")

    def ler():
        for linha in tunel.stdout:
            m = re.search(r"https://[\w-]+\.trycloudflare\.com", linha)
            if m:
                print(f"\n  Link público:   {m.group(0)}   (funciona em qualquer rede, até no 4G)\n", flush=True)

    threading.Thread(target=ler, daemon=True).start()
    return tunel


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--publico", action="store_true", help="gera também um link https:// público")
    ap.add_argument("--demo", action="store_true", help="cria os imóveis de demonstração se o banco estiver vazio")
    args = ap.parse_args()

    env = dict(os.environ)
    if args.demo:
        env["AUTOLEITURA_DEMO"] = "1"

    ip = ip_local()
    print("\n  AutoLeitura")
    print(f"  No computador:  http://localhost:{PORTA}")
    if ip:
        print(f"  No celular:     http://{ip}:{PORTA}   (mesma rede Wi-Fi)")
    print("  Para encerrar:  Ctrl+C\n")
    # headless evita a pergunta de e-mail que o Streamlit faz na primeira execução;
    # por isso o navegador é aberto aqui, assim que o servidor responder
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(Path(__file__).with_name("app.py")),
         "--server.address", "0.0.0.0", "--server.port", str(PORTA), "--server.headless", "true"], env=env)
    tunel = None
    for _ in range(60):
        try:
            socket.create_connection(("127.0.0.1", PORTA), timeout=1).close()
            webbrowser.open(f"http://localhost:{PORTA}")
            if args.publico:
                tunel = abrir_tunel()
            break
        except OSError:
            if proc.poll() is not None:
                break
            time.sleep(1)
    try:
        raise SystemExit(proc.wait())
    except KeyboardInterrupt:
        proc.terminate()
    finally:
        if tunel is not None:
            tunel.terminate()
