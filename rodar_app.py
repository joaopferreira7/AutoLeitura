"""Abre o app no computador e mostra o endereço para acessar pelo celular.

  python rodar_app.py

O celular precisa estar na mesma rede Wi-Fi do computador. Na primeira vez, o Windows
pergunta se o Python pode se comunicar na rede: permita em "Redes privadas".
"""
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

PORTA = 8501


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


if __name__ == "__main__":
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
         "--server.address", "0.0.0.0", "--server.port", str(PORTA), "--server.headless", "true"])
    for _ in range(60):
        try:
            socket.create_connection(("127.0.0.1", PORTA), timeout=1).close()
            webbrowser.open(f"http://localhost:{PORTA}")
            break
        except OSError:
            if proc.poll() is not None:
                break
            time.sleep(1)
    try:
        raise SystemExit(proc.wait())
    except KeyboardInterrupt:
        proc.terminate()
