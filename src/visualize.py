"""
Visualização do carrinho de corrida.

Funções principais:
    renderizar_episodio(env, agente_fn, ...) — animação no terminal

A animação é renderizada diretamente no terminal usando códigos ANSI para
limpar a tela entre frames. Funciona em qualquer terminal moderno (macOS,
Linux, Windows com Terminal). Não requer pillow.

Uso típico:
    from visualize import renderizar_episodio
    renderizar_episodio(env, lambda obs: agente.escolher_acao(obs))
"""

from __future__ import annotations
import math
import time
from typing import Callable, Optional

import numpy as np

from track import PAREDE, ASFALTO, LARGADA, CHEGADA
from env import AmbienteCarro, ANGULOS_RAIOS


# === Caracteres usados na visualização do terminal ===
EMOJI_PAREDE = "🧱"
EMOJI_ASFALTO = "⚪️"
EMOJI_LARGADA = "🟢"
EMOJI_CHEGADA = "🏁"
EMOJI_RASTRO = "🟦"   # célula já percorrida pelo carro

# Carro: símbolo escolhido conforme o ângulo (8 direções)
SIMBOLOS_CARRO = ["➡️", "↘️", "⬇️", "↙️", "⬅️", "↖️", "⬆️", "↗️"]


# === Auxiliar: limpa a tela do terminal e move cursor para o topo ===
def _limpar_terminal():
    print("\033[2J\033[H", end="", flush=True)


def _simbolo_carro(theta: float) -> str:
    """Escolhe um emoji direcional para o carro com base em theta."""
    # Normaliza theta para [0, 2pi) e mapeia em 8 direções
    theta_norm = (theta + math.pi / 8) % (2 * math.pi)
    setor = int(theta_norm / (math.pi / 4)) % 8
    return SIMBOLOS_CARRO[setor]


# === Animação no terminal ===
def _renderizar_frame(env: AmbienteCarro, rastro: set, info_extra: str = "") -> str:
    """
    Constrói uma string com a representação textual do estado atual da pista.
    Retorna a string pronta para imprimir.
    """
    grid = env.grid
    h, w = grid.shape
    c = env.carro
    cy, cx = int(c.y), int(c.x)

    linhas = []
    for y in range(h):
        linha_chars = []
        for x in range(w):
            if (y, x) == (cy, cx):
                # Posição do carro: emoji direcional
                linha_chars.append(_simbolo_carro(c.theta))
            elif (y, x) in rastro:
                # Rastro do trajeto já percorrido
                linha_chars.append(EMOJI_RASTRO)
            else:
                celula = grid[y, x]
                if celula == PAREDE:
                    linha_chars.append(EMOJI_PAREDE)
                elif celula == LARGADA:
                    linha_chars.append(EMOJI_LARGADA)
                elif celula == CHEGADA:
                    linha_chars.append(EMOJI_CHEGADA)
                else:
                    linha_chars.append(EMOJI_ASFALTO)
        linhas.append("".join(linha_chars))

    grade_str = "\n".join(linhas)

    rodape = (
        f"\nPasso: {env.passos}  |  "
        f"v: {c.v:.2f}  |  "
        f"θ: {math.degrees(c.theta):.0f}°  |  "
        f"Progresso: {env.melhor_progresso_atingido}/{env.progresso_max}"
    )
    if info_extra:
        rodape += f"\n{info_extra}"

    return grade_str + rodape


def renderizar_episodio(
    env: AmbienteCarro,
    agente_fn: Callable[[np.ndarray], int],
    fps: int = 8,
    max_steps: Optional[int] = None,
    limpar_tela: bool = True,
    pausa_final: float = 1.5,
):
    """
    Roda um episódio com agente_fn(obs) -> action e renderiza no terminal.

    Args:
        env: ambiente AmbienteCarro (será chamado reset)
        agente_fn: função que recebe observação e retorna ação
        fps: frames por segundo (controla velocidade da animação)
        max_steps: limita o nº de passos (default: env.max_steps)
        limpar_tela: se True, limpa a tela entre frames (animação fluida).
                     Se False, imprime cada frame em sequência (útil para logs).
        pausa_final: tempo (s) que o último frame fica visível.

    Retorna: (recompensa_total, n_passos, info_final)
    """
    obs = env.reset()
    max_steps = max_steps or env.max_steps
    intervalo = 1.0 / fps

    rastro = set()
    reward_total = 0.0
    info_final = {}

    # Frame inicial (estado de largada)
    if limpar_tela:
        _limpar_terminal()
    print(_renderizar_frame(env, rastro, info_extra="Início do episódio"))
    time.sleep(intervalo)

    for t in range(max_steps):
        # Marca a célula atual no rastro ANTES de mover (para registrar onde passou)
        cy, cx = int(env.carro.y), int(env.carro.x)
        rastro.add((cy, cx))

        action = agente_fn(obs)
        obs, r, term, trunc, info = env.step(action)
        reward_total += r

        # Status do passo
        nomes_acoes = {0: "nada", 1: "acelerar", 2: "frear", 3: "esquerda", 4: "direita"}
        info_extra = f"Ação: {nomes_acoes[action]}  |  Reward: {r:+.2f}  |  Total: {reward_total:+.2f}"

        if limpar_tela:
            _limpar_terminal()
        print(_renderizar_frame(env, rastro, info_extra=info_extra))

        if term or trunc:
            info_final = info
            if info.get("chegada"):
                print("\n🏁 Chegou na linha de chegada!")
            elif info.get("colisao"):
                print("\n💥 Colisão!")
            elif trunc:
                print("\n⏱️  Limite de passos atingido.")
            break

        time.sleep(intervalo)

    time.sleep(pausa_final)
    print(f"\nResumo: {env.passos} passos, recompensa total = {reward_total:.2f}")
    return reward_total, env.passos, info_final


if __name__ == "__main__":
    import sys
    import pickle
    from pathlib import Path

    pista = sys.argv[1] if len(sys.argv) > 1 else "pistas/pista_01.txt"
    env = AmbienteCarro(pista)

    # Tenta carregar modelo treinado em treinamento/qlearning.pkl
    # Contrato do pickle (ver enunciado/anexo_b_pickle.md):
    #   - "q_table": dict[tuple[int,...], np.ndarray]  — chave discretizada → Q-valores
    #   - "discretization_K": int (default 5)
    raiz = Path(__file__).resolve().parent.parent
    caminho_modelo = raiz / "treinamento" / "qlearning.pkl"

    if caminho_modelo.exists():
        print(f"Carregando modelo treinado de {caminho_modelo} ...")
        with open(caminho_modelo, "rb") as f:
            modelo = pickle.load(f)
        q_table = modelo["q_table"]
        K = modelo.get("discretization_K", 5)

        def discretizar(obs):
            return tuple(min(int(v * K), K - 1) for v in obs)

        def politica_treinada(obs):
            chave = discretizar(obs)
            if isinstance(q_table, dict):
                if chave not in q_table:
                    return 0  # estado nunca visto no treino → "não fazer nada"
                valores = q_table[chave]
            else:
                valores = q_table[chave]  # np.ndarray multidimensional
            return int(np.argmax(valores))

        agente_fn = politica_treinada
    else:
        print(f"⚠️  {caminho_modelo} não encontrado — usando agente trivial (acelera 3x).")
        contador = [0]
        def agente_trivial(obs):
            contador[0] += 1
            return 1 if contador[0] <= 3 else 0
        agente_fn = agente_trivial

    reward_total, n_passos, info = renderizar_episodio(env, agente_fn, fps=4)
