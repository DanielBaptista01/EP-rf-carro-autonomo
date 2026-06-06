"""
Esqueleto da sua solução para o EP do carrinho (versão tabular).

Você deve implementar:
    - AgenteQLearning  (tabular)

E preencher main() para orquestrar:
    1. Treinamento round-robin nas pistas 01-16 → salva treinamento/qlearning.pkl.
    2. Avaliação gulosa (ε = 0) nas pistas de holdout 17 e 18 → gera
       q_learning_pista_17.txt e q_learning_pista_18.txt (formato do README §4.3).
"""

import sys
import random
import argparse
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from env import AmbienteCarro  # noqa: E402

# === Configuração ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Diretório onde o modelo treinado será salvo via pickle
DIR_TREINAMENTO = Path("treinamento")
DIR_TREINAMENTO.mkdir(exist_ok=True)

# Conjuntos de pistas
PISTAS_TREINO = [f"pistas/pista_{i:02d}.txt" for i in range(1, 17)]   # 01..16
PISTAS_HOLDOUT = [f"pistas/pista_{i:02d}.txt" for i in range(17, 19)] # 17, 18


# ============================================================================
# Q-LEARNING TABULAR
# ============================================================================

class AgenteQLearning:
    """
    Agente Q-Learning Tabular Model-Free.
    Gerencia a Tabela Q e a matemática de atualização de Bellman.
    """

    def __init__(self, obs_dim, n_actions, K=5, alpha=0.1, gamma=0.95, eps_inicial=1.0, eps_final=0.01):
        self.n_actions = n_actions
        self.K = K
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_inicial
        self.eps_final = eps_final
        
        # Otimização de memória: Dicionário que inicializa novos estados com vetor de zeros automaticamente
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))

    @classmethod
    def from_modelo(cls, modelo_dict):
        """Construtor alternativo para instanciar agente apenas para avaliação."""
        agente = cls(obs_dim=6, n_actions=5, K=modelo_dict["discretization_K"])
        agente.Q.update(modelo_dict["q_table"])
        agente.eps = 0.0 # Política 100% gulosa para avaliação
        return agente

    def discretizar(self, obs):
        """
        Converte vetor float [0, 1] em chave discreta (tupla de ints).
        O binning é feito garantindo que valores limites de saturação não estourem o array K.
        """
        return tuple(min(int(v * self.K), self.K - 1) for v in obs)

    def escolher_acao(self, obs):
        """Política ε-greedy equilibrando exploração do ambiente e explotação da Tabela Q."""
        estado = self.discretizar(obs)
        if random.random() < self.eps:
            return random.randint(0, self.n_actions - 1)
        else:
            return int(np.argmax(self.Q[estado]))

    def atualizar(self, s, a, r, s_prox, terminou):
        """
        Aplica a regra de update da Diferença Temporal (TD).
        Se for o estado terminal (bateu ou chegou), o valor futuro Q(s_prox) = 0
        pois não há mais transições válidas.
        """
        estado = self.discretizar(s)
        prox_estado = self.discretizar(s_prox)
        
        max_q_prox = 0.0 if terminou else np.max(self.Q[prox_estado])
        erro_td = r + self.gamma * max_q_prox - self.Q[estado][a]
        
        self.Q[estado][a] += self.alpha * erro_td


# ============================================================================
# LOOP DE TREINAMENTO (round-robin nas 16 pistas de treino)
# ============================================================================

def treinar_round_robin(pistas_treino, agente, n_episodios_por_pista, max_passos, decaimento_eps_episodios, verbose=True):
    """
    Loop de treinamento em round-robin mitigando Catastrophic Forgetting.
    """
    historico_recompensas = []
    historico_sucessos = []
    rewards_por_pista = {p: [] for p in pistas_treino}
    
    n_total = n_episodios_por_pista * len(pistas_treino)
    envs = {p: AmbienteCarro(p, max_steps=max_passos, seed=SEED) for p in pistas_treino}

    # Taxa fixa pré-calculada para garantir convergência de Epsilon exata aos 80%
    taxa_decaimento = -np.log(agente.eps_final) / decaimento_eps_episodios

    for ep in range(n_total):
        # 1. Schedule do epsilon (Exponencial)
        if ep <= decaimento_eps_episodios:
            agente.eps = np.exp(-taxa_decaimento * ep)
        else:
            agente.eps = agente.eps_final

        # 2. Sortear pista (Round-Robin Aleatório)
        pista = random.choice(pistas_treino)
        env = envs[pista]

        # 3. Execução do Episódio
        obs = env.reset()
        done = False
        recompensa_acumulada = 0.0
        
        while not done:
            action = agente.escolher_acao(obs)
            obs_prox, reward, term, trunc, info = env.step(action)
            done = term or trunc
            
            agente.atualizar(obs, action, reward, obs_prox, done)
            
            obs = obs_prox
            recompensa_acumulada += reward

        # 4. Registro de Métricas
        sucesso = info.get("chegada", False)
        historico_recompensas.append(recompensa_acumulada)
        historico_sucessos.append(sucesso)
        rewards_por_pista[pista].append(recompensa_acumulada)
        
        if verbose and ep % 10000 == 0:
            taxa_sucesso = sum(historico_sucessos[-1000:]) / 1000 if len(historico_sucessos) > 0 else 0
            print(f"Ep {ep}/{n_total} | Eps: {agente.eps:.3f} | Tx Sucesso: {taxa_sucesso:.1%} | Q-States: {len(agente.Q)}")

    return historico_recompensas, historico_sucessos, rewards_por_pista


# ============================================================================
# AVALIAÇÃO (com ε = 0)
# ============================================================================

def avaliar(env, agente, n_episodios=10):
    """
    Roda política estritamente gulosa para extrair telemetria do Holdout.
    """
    agente.eps = 0.0 # Garantia de explotação pura
    
    melhor_tempo = float('inf')
    melhor_recompensa = float('-inf')
    vel_max_geral = 0.0
    vel_soma_geral = 0.0
    passos_soma_geral = 0
    sucesso_geral = False

    for _ in range(n_episodios):
        obs = env.reset()
        done = False
        passos = 0
        recompensa_ep = 0.0
        vel_max_ep = 0.0
        vel_soma_ep = 0.0
        
        while not done:
            acao = agente.escolher_acao(obs)
            obs_prox, reward, term, trunc, info = env.step(acao)
            done = term or trunc
            
            # obs[5] é v_norm (normalizada de 0 a 1). Velocidade real é obtida multiplicando pelo V_MAX (2.0) do ambiente
            velocidade_real = obs_prox[5] * 2.0 
            vel_max_ep = max(vel_max_ep, velocidade_real)
            vel_soma_ep += velocidade_real
            
            passos += 1
            recompensa_ep += reward
            obs = obs_prox
            
        if info.get("chegada", False):
            sucesso_geral = True
            melhor_tempo = min(melhor_tempo, passos)
            melhor_recompensa = max(melhor_recompensa, recompensa_ep)
            vel_max_geral = max(vel_max_geral, vel_max_ep)
            vel_soma_geral += vel_soma_ep
            passos_soma_geral += passos

    vel_media = vel_soma_geral / passos_soma_geral if passos_soma_geral > 0 else 0.0

    return {
        "n_passos": melhor_tempo if sucesso_geral else passos,
        "recompensa_total": melhor_recompensa if sucesso_geral else recompensa_ep,
        "sucesso": sucesso_geral,
        "velocidade_media": vel_media,
        "velocidade_maxima": vel_max_geral if sucesso_geral else vel_max_ep
    }


# ============================================================================
# SALVAR / CARREGAR MODELO
# ============================================================================

def treinar_ou_carregar(nome, fn_treinar, recarregar=False):
    arquivo = DIR_TREINAMENTO / f"{nome}.pkl"
    if arquivo.exists() and not recarregar:
        print(f"Carregando {arquivo} ...")
        with open(arquivo, "rb") as f:
            return pickle.load(f)
    else:
        print(f"Treinando {nome} ...")
        resultado = fn_treinar()
        with open(arquivo, "wb") as f:
            # Como DefaultDict lambda não é serializável nativamente pelo pickle, convertemos para dict puro antes de salvar
            resultado["q_table"] = dict(resultado["q_table"])
            pickle.dump(resultado, f)
        print(f"Salvo em {arquivo}")
        return resultado


# ============================================================================
# GERAÇÃO DOS ARQUIVOS DE SAÍDA
# ============================================================================

def escrever_saida(caminho, nome_algoritmo, pista, resultado_avaliacao, n_episodios_treinados, n_estados):
    """
    Grava os artefatos de texto padronizados exatamente como a matriz de avaliação requer.
    """
    with open(caminho, "w") as f:
        f.write(f"=== Pista: {pista} ===\n")
        f.write(f"Algoritmo: {nome_algoritmo} (round-robin em pistas 01-16)\n")
        f.write(f"Episódios totais de treinamento: {n_episodios_treinados}\n")
        f.write(f"Estados populados: {n_estados}\n")
        f.write(f"Tempo de chegada (passos): {resultado_avaliacao['n_passos']}\n")
        f.write(f"Velocidade média: {resultado_avaliacao['velocidade_media']:.2f}\n")
        f.write(f"Velocidade máxima atingida: {resultado_avaliacao['velocidade_maxima']:.2f}\n")
        f.write(f"Recompensa total: {resultado_avaliacao['recompensa_total']:.2f}\n")
        sucesso_str = "SIM" if resultado_avaliacao['sucesso'] else "NAO"
        f.write(f"Sucesso: {sucesso_str}\n")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodios-por-pista", type=int, default=30_000,
                        help="Episódios de treino por pista no round-robin (default: 30000)")
    parser.add_argument("--max-passos", type=int, default=500)
    parser.add_argument("--K", type=int, default=5,
                        help="Baldes da discretização (default: 5; ver README §3.2)")
    parser.add_argument("--recarregar", action="store_true",
                        help="Força re-treino mesmo se o pickle existir")
    parser.add_argument("--avaliar", type=str, default=None,
                        help="Apenas avalia o modelo salvo na pista especificada (pula treino)")
    args = parser.parse_args()

    # ─── Treinamento round-robin (ou carregamento) ────────────────────────
    def fn_treinar():
        agente = AgenteQLearning(obs_dim=6, n_actions=5, K=args.K)
        n_total = args.episodios_por_pista * len(PISTAS_TREINO)
        
        rewards, sucessos, rewards_por_pista = treinar_round_robin(
            PISTAS_TREINO, agente, args.episodios_por_pista, args.max_passos,
            decaimento_eps_episodios=int(0.8 * n_total),
        )
        
        return {
            "q_table": agente.Q, 
            "discretization_K": args.K,
            "n_episodes_trained": n_total,
            "rewards_history": rewards,
            "rewards_por_pista": rewards_por_pista,
            "config": {"alpha": agente.alpha, "gamma": agente.gamma},
            "seed": SEED,
            "tracks_used": PISTAS_TREINO,
        }

    modelo = treinar_ou_carregar("qlearning", fn_treinar, recarregar=args.recarregar)

    # ─── Avaliação ─────────────────────────────────────────────────────────
    agente_avaliacao = AgenteQLearning.from_modelo(modelo)
    n_estados_populados = len(modelo["q_table"])

    pistas_avaliar = [args.avaliar] if args.avaliar else PISTAS_HOLDOUT
    for pista in pistas_avaliar:
        env = AmbienteCarro(pista, max_steps=args.max_passos, seed=SEED)
        resultado = avaliar(env, agente_avaliacao)
        
        nome_pista = Path(pista).stem  # ex: "pista_17"
        escrever_saida(f"q_learning_{nome_pista}.txt", "Q-Learning",
                       pista, resultado, modelo["n_episodes_trained"], n_estados_populados)
        
        print(f"Avaliação {nome_pista}: Sucesso={resultado['sucesso']} | Reward={resultado['recompensa_total']:.1f} | Passos={resultado['n_passos']}")

    print("\nPronto.")


if __name__ == "__main__":
    main()