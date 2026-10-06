"""Métricas de avaliação definidas na seção 11 da proposta."""


def levenshtein(a, b):
    """Distância de edição entre duas strings (inserção, remoção, substituição)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def evaluate_readings(preds, gts):
    """Compara leituras previstas com as reais.

    Retorna:
      reading_acc  – fração de leituras com todos os dígitos corretos (métrica principal)
      digit_acc    – acurácia por dígito (posição a posição; só conta quando os tamanhos batem,
                     senão a leitura inteira conta como erro em todas as posições)
      mean_lev     – distância de Levenshtein média
      per_digit    – acurácia por classe 0–9 (quantas vezes o dígito real d foi lido corretamente)
    """
    assert len(preds) == len(gts)
    n = len(gts)
    full = sum(p == g for p, g in zip(preds, gts))
    lev = sum(levenshtein(p, g) for p, g in zip(preds, gts))

    correct_digits = total_digits = 0
    per_class = {str(d): [0, 0] for d in range(10)}
    for p, g in zip(preds, gts):
        for i, gd in enumerate(g):
            ok = len(p) == len(g) and p[i] == gd
            correct_digits += ok
            total_digits += 1
            per_class[gd][0] += ok
            per_class[gd][1] += 1

    return {
        "n": n,
        "reading_acc": full / n if n else 0.0,
        "digit_acc": correct_digits / total_digits if total_digits else 0.0,
        "mean_lev": lev / n if n else 0.0,
        "per_digit": {d: (c / t if t else None) for d, (c, t) in per_class.items()},
    }
