"""
Ambiente do carrinho de corrida.

Estado físico interno:
    pos = (x, y) em coordenadas contínuas (x = coluna, y = linha)
    theta em radianos (0 = leste, pi/2 = sul)
    v em células-por-passo

Estado observável (o que o agente vê):
    [d_0, d_+30, d_-30, d_+60, d_-60, v_norm]   # vetor de 6 floats

Ações: 0 = nada, 1 = acelerar, 2 = frear, 3 = virar esq, 4 = virar dir

NOTA: termos como `step`, `reset`, `obs`, `action`, `reward`, `info` são
mantidos em inglês por serem o vocabulário canônico de Aprendizado por
Reforço (Sutton & Barto, Gymnasium). Todo o resto está em português.
"""

from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from track import carregar_pista, eh_pilotavel, PAREDE, LARGADA, CHEGADA, ASFALTO


# === Constantes do carro ===
# Estes valores podem ser ajustados como parte do EP. Documente mudanças no relatório.
V_MAX = 2.0                       # velocidade máxima (células/passo)
V_DELTA = 0.5                     # incremento de velocidade por aceleração/freio
THETA_DELTA = math.radians(30)    # ângulo por virada (15° é mais realista mas mais difícil de aprender)
N_RAIOS = 5
ANGULOS_RAIOS = [0, math.radians(30), math.radians(-30), math.radians(60), math.radians(-60)]
DIST_MAX_RAIO = 10.0              # alcance máximo do sensor (em células)
PASSO_RAIO = 0.1

# Recompensas
R_TEMPO = -0.1
R_COLISAO = -100.0
R_CHEGADA = 500.0


@dataclass
class EstadoCarro:
    x: float
    y: float
    theta: float
    v: float


def calcular_campo_progresso(grid: np.ndarray, largada: Tuple[int, int]) -> np.ndarray:
    """
    BFS a partir da largada, atribuindo a cada célula pilotável sua distância
    (em passos de grid) até a largada. Células de parede recebem -1.

    Esse campo é usado para reward shaping: o "progresso" do carro é a célula
    de maior distância já alcançada.
    """
    h, w = grid.shape
    campo = np.full((h, w), -1, dtype=np.int32)
    fila = deque()
    sy, sx = largada
    campo[sy, sx] = 0
    fila.append((sy, sx))
    while fila:
        y, x = fila.popleft()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and campo[ny, nx] == -1 and grid[ny, nx] != PAREDE:
                campo[ny, nx] = campo[y, x] + 1
                fila.append((ny, nx))
    return campo


def lancar_raio(
    grid: np.ndarray,
    x: float,
    y: float,
    angulo: float,
    dist_max: float = DIST_MAX_RAIO,
    passo: float = PASSO_RAIO,
) -> float:
    """
    Lança um raio a partir de (x, y) na direção 'angulo' (radianos) e retorna
    a distância até a primeira parede (ou dist_max se não bater em nada).

    Convenção: x cresce para direita (coluna), y cresce para baixo (linha).
    angulo = 0 aponta para +x (direita); angulo = pi/2 aponta para +y (baixo).
    """
    dx = math.cos(angulo)
    dy = math.sin(angulo)
    h, w = grid.shape
    d = 0.0
    while d < dist_max:
        d += passo
        cx = x + dx * d
        cy = y + dy * d
        ix, iy = int(cx), int(cy)
        if iy < 0 or iy >= h or ix < 0 or ix >= w or grid[iy, ix] == PAREDE:
            return d
    return dist_max


class AmbienteCarro:
    """
    Ambiente do carrinho. API inspirada no Gymnasium mas implementada do zero.

    Fluxo de uso típico (interação agente ↔ ambiente):
        1. env = AmbienteCarro("pistas/pista_XX.txt")   # cria o ambiente
        2. obs = env.reset()                            # inicia um episódio
        3. loop:
               action = politica(obs)                   # agente decide a ação
               obs, reward, terminated, truncated, info = env.step(action)
               if terminated or truncated: break        # episódio acabou

    Convenções de coordenadas:
        - x = coluna do grid (cresce para a direita)
        - y = linha do grid (cresce para BAIXO — convenção de imagem, não de matemática)
        - theta = 0 aponta para +x (leste);  theta = pi/2 aponta para +y (sul)
    """

    def __init__(self, caminho_pista: str, max_steps: int = 500, seed: Optional[int] = None):
        # Carrega a pista do arquivo emoji e identifica largada/chegada.
        self.grid, self.celula_largada, self.celula_chegada = carregar_pista(caminho_pista)

        # Pré-computa o campo de progresso (BFS a partir da largada).
        # Esse mapa atribui a cada célula sua distância em passos até a largada,
        # e é a base do reward shaping: o agente é recompensado por ALCANÇAR
        # células de distância maior do que já viu antes.
        self.campo_progresso = calcular_campo_progresso(self.grid, self.celula_largada)
        self.progresso_max = self.campo_progresso.max()  # progresso máximo possível na pista

        # Limite de passos por episódio: evita que um agente ruim fique girando
        # em círculos para sempre. Quando atingido, o episódio é "truncado".
        self.max_steps = max_steps

        # Gerador de aleatoriedade local (não usa o estado global do numpy).
        # Permite reproduzir resultados ao passar o mesmo `seed`.
        self.rng = np.random.default_rng(seed)

        # Estado físico do carro — só fica definido após chamar reset().
        self.carro: Optional[EstadoCarro] = None

        # Contadores resetados a cada episódio.
        self.passos = 0
        self.melhor_progresso_atingido = 0

    def reset(self) -> np.ndarray:
        """Inicia um novo episódio. Deve ser chamado ANTES do primeiro step()."""
        sy, sx = self.celula_largada
        # Posiciona o carro no centro da célula de largada (+0.5 para evitar
        # ambiguidade na fronteira entre células). Começa parado (v=0) e
        # apontando para o leste (theta=0).
        self.carro = EstadoCarro(x=sx + 0.5, y=sy + 0.5, theta=0.0, v=0.0)
        self.passos = 0
        self.melhor_progresso_atingido = 0
        return self._observar()

    def _observar(self) -> np.ndarray:
        """
        Constrói o vetor de observação que o agente recebe.

        O agente NÃO vê sua posição (x,y) nem sua orientação theta diretamente —
        só os sensores LIDAR (5 raios) e a velocidade. Isso força a política a
        depender de features locais (o que está à minha frente?), tornando o
        aprendizado mais transferível entre pistas.
        """
        c = self.carro
        # Lança 5 raios nas direções configuradas em ANGULOS_RAIOS, sempre
        # relativos ao ângulo atual do carro (theta + offset).
        raios = [lancar_raio(self.grid, c.x, c.y, c.theta + a) for a in ANGULOS_RAIOS]
        # Normaliza para [0, 1] dividindo pelo alcance máximo do sensor.
        # Estados normalizados ajudam tanto algoritmos tabulares (discretização
        # mais uniforme) quanto métodos baseados em função de aproximação.
        raios_norm = [r / DIST_MAX_RAIO for r in raios]
        v_norm = c.v / V_MAX
        return np.array(raios_norm + [v_norm], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Executa UMA ação no ambiente e retorna a tupla padrão de RL:
            obs        — nova observação (vetor de 6 floats)
            reward     — recompensa recebida neste passo
            terminated — True se o episódio acabou por chegada ou colisão
            truncated  — True se atingiu o limite de passos (max_steps)
            info       — dicionário com metadados ({"chegada": True}, {"colisao": True} ou {})
        """
        assert self.carro is not None, "Chame reset() antes de step()"
        assert 0 <= action <= 4
        c = self.carro

        # --- 1. Atualiza variáveis físicas do carro conforme a ação ---
        # Note que a velocidade é "trancada" em [0, V_MAX] — o carro não anda
        # para trás. Virar não muda velocidade; acelerar/frear não muda theta.
        if action == 1:    # acelerar
            c.v = min(c.v + V_DELTA, V_MAX)
        elif action == 2:  # frear
            c.v = max(c.v - V_DELTA, 0.0)
        elif action == 3:  # virar à esquerda
            c.theta -= THETA_DELTA
        elif action == 4:  # virar à direita
            c.theta += THETA_DELTA
        # action == 0: nada — mantém v e theta, mas o carro continua se movendo
        # pela inércia (próximo bloco usa v atual para deslocar a posição).

        # --- 2. Calcula a posição alvo a partir da física simples ---
        # Cinemática 2D: dx = v·cos(θ), dy = v·sin(θ).
        # Importante: NÃO atualiza c.x/c.y ainda — primeiro precisamos checar
        # se a nova posição é válida (não colide com parede).
        novo_x = c.x + c.v * math.cos(c.theta)
        novo_y = c.y + c.v * math.sin(c.theta)

        self.passos += 1
        terminated = False
        truncated = False
        info = {}

        # --- 3. Verifica colisão na posição alvo ---
        # Trunca floats para inteiros para descobrir em qual célula caiu.
        # Colisão acontece se sair do grid OU cair em uma célula de PAREDE.
        # Nota: não há "deslizamento" — bateu, acabou (terminated=True).
        ix, iy = int(novo_x), int(novo_y)
        h, w = self.grid.shape
        fora_dos_limites = (iy < 0 or iy >= h or ix < 0 or ix >= w)
        if fora_dos_limites or self.grid[iy, ix] == PAREDE:
            reward = R_COLISAO
            terminated = True
            info["colisao"] = True
            # Retorno antecipado: não atualiza c.x/c.y (o carro "morreu" antes de chegar lá).
            return self._observar(), reward, terminated, truncated, info

        # --- 4. Movimento aceito: confirma a nova posição ---
        c.x = novo_x
        c.y = novo_y

        # --- 5. Reward shaping baseado em progresso ---
        # `progresso_celula` é a distância (em passos de BFS) da célula atual
        # até a largada. Quanto maior, mais perto do "fim natural" da pista.
        progresso_celula = self.campo_progresso[iy, ix]
        # Recompensa SOMENTE pelo progresso NOVO — diferença entre o progresso
        # da célula atual e o melhor já atingido no episódio. Assim, andar para
        # trás ou ficar girando em círculos não gera recompensa (max(0, ...)).
        # Esse design evita que o agente exploit "ir-e-voltar" por reward shaping.
        delta_progresso = max(0, progresso_celula - self.melhor_progresso_atingido)
        self.melhor_progresso_atingido = max(self.melhor_progresso_atingido, progresso_celula)

        # Reward total do passo = custo de tempo + bônus de progresso.
        # R_TEMPO < 0 incentiva o agente a terminar o episódio rápido.
        reward = R_TEMPO + float(delta_progresso)

        # --- 6. Verifica se chegou à linha de chegada ---
        if self.grid[iy, ix] == CHEGADA:
            reward += R_CHEGADA  # bônus grande (+500) por completar a pista
            terminated = True
            info["chegada"] = True

        # --- 7. Verifica limite de passos (truncamento, não terminação) ---
        # Diferença sutil mas importante em RL: `truncated` significa que o
        # episódio acabou por limite externo, não porque o ambiente "concluiu".
        # Algoritmos como Q-Learning tratam os dois casos diferente no bootstrap.
        if self.passos >= self.max_steps:
            truncated = True

        return self._observar(), reward, terminated, truncated, info

    @property
    def obs_dim(self) -> int:
        """Dimensão do vetor de observação (5 raios LIDAR + 1 velocidade = 6)."""
        return N_RAIOS + 1

    @property
    def n_actions(self) -> int:
        """Número de ações disponíveis."""
        return 5


if __name__ == "__main__":
    # Teste rápido: carro vai em frente até bater
    import sys
    pista = sys.argv[1] if len(sys.argv) > 1 else "pistas/pista_01.txt"
    env = AmbienteCarro(pista)
    obs = env.reset()
    print(f"obs inicial: {obs}")
    print(f"Carro em ({env.carro.x:.2f}, {env.carro.y:.2f}), theta={math.degrees(env.carro.theta):.0f}, v={env.carro.v}")
    print(f"Progresso máximo da pista: {env.progresso_max}")
    print()
    print("Acelerando 3x e indo em frente:")
    reward_total = 0.0
    for t in range(15):
        if t < 3:
            action = 1  # acelera
        else:
            action = 0  # nada
        obs, r, term, trunc, info = env.step(action)
        reward_total += r
        print(
            f"t={t}: a={action}, pos=({env.carro.x:.2f},{env.carro.y:.2f}), "
            f"v={env.carro.v:.1f}, r={r:.2f}, term={term}, info={info}"
        )
        if term or trunc:
            break
    print(f"\nRecompensa total: {reward_total:.2f}")
