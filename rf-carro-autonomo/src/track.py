"""
Parser de pistas em formato emoji.

Cada célula é um emoji. As linhas são separadas por '\n'.
Dentro de cada linha, células são separadas por espaços.
"""

import numpy as np
from pathlib import Path
from typing import Tuple

# Mapeamento de emojis para códigos numéricos do grid
ASFALTO = 0     # ⚪️ asfalto
PAREDE = 1      # 🧱 parede
LARGADA = 2     # 🟢 largada (também é asfalto)
CHEGADA = 3     # 🏁 chegada (também é asfalto)

EMOJI_PARA_CODIGO = {
    "⚪": ASFALTO,
    "🧱": PAREDE,
    "🟢": LARGADA,
    "🏁": CHEGADA,
}


def _normalizar(token: str) -> str:
    """Remove o seletor de variação \uFE0F (VS-16) que aparece em alguns emojis."""
    return token.replace("\ufe0f", "")


def carregar_pista(caminho: str | Path) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """
    Lê um arquivo de pista e retorna:
        grid: np.ndarray de shape (H, W) com códigos ASFALTO/PAREDE/LARGADA/CHEGADA
        largada: (linha, coluna) da posição de largada
        chegada: (linha, coluna) da linha de chegada

    Convenção de coordenadas: (linha=y, coluna=x). Linha 0 = topo.
    """
    caminho = Path(caminho)
    texto = caminho.read_text(encoding="utf-8")
    linhas = [ln.rstrip() for ln in texto.splitlines() if ln.strip()]

    rows = []
    for ln in linhas:
        # Split por espaços (suporta múltiplos espaços)
        tokens = [_normalizar(t) for t in ln.split() if t.strip()]
        rows.append(tokens)

    largura = max(len(r) for r in rows)
    altura = len(rows)

    # Preenche com PAREDE caso linhas tenham comprimentos diferentes
    grid = np.full((altura, largura), PAREDE, dtype=np.int8)
    largada = None
    chegada = None

    for y, linha in enumerate(rows):
        for x, tok in enumerate(linha):
            if tok in EMOJI_PARA_CODIGO:
                codigo = EMOJI_PARA_CODIGO[tok]
                grid[y, x] = codigo
                if codigo == LARGADA:
                    largada = (y, x)
                elif codigo == CHEGADA:
                    chegada = (y, x)
            else:
                raise ValueError(
                    f"Emoji desconhecido em {caminho.name}, linha {y+1}, coluna {x+1}: {tok!r}"
                )

    if largada is None:
        raise ValueError(f"{caminho.name}: nenhum 🟢 (largada) encontrado")
    if chegada is None:
        raise ValueError(f"{caminho.name}: nenhum 🏁 (chegada) encontrado")

    return grid, largada, chegada


def eh_pilotavel(grid: np.ndarray, y: int, x: int) -> bool:
    """Retorna True se a célula (y, x) é asfalto (não é parede)."""
    h, w = grid.shape
    if y < 0 or y >= h or x < 0 or x >= w:
        return False
    return grid[y, x] != PAREDE


if __name__ == "__main__":
    # Teste rápido
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "pistas/pista_01.txt"
    g, larg, cheg = carregar_pista(p)
    print(f"Pista: {p}")
    print(f"Tamanho: {g.shape}")
    print(f"Largada: {larg}")
    print(f"Chegada: {cheg}")
    print(f"Células pilotáveis: {(g != PAREDE).sum()}")
    print()
    # Visualização ASCII
    chars = {ASFALTO: ".", PAREDE: "#", LARGADA: "S", CHEGADA: "F"}
    for linha in g:
        print("".join(chars[c] for c in linha))
