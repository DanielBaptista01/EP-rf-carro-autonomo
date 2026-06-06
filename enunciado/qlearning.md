# Q-Learning — Matemática, Algoritmo e Implementação

Este documento explica o algoritmo de **Q-Learning** que você precisa implementar no EP do carrinho. Cobre a intuição, a matemática, o pseudocódigo e dicas práticas de implementação em Python.

Pré-requisitos: você já leu o `README.md` do projeto (especialmente §1 sobre o ambiente, §3 sobre a representação de estado e discretização, e §1.5 sobre recompensas).

---

## 1. Intuição

O carrinho vive em um **Processo de Decisão de Markov (MDP)**: a cada instante ele observa um estado $s$, escolhe uma ação $a$, recebe uma recompensa $r$ e transita para um novo estado $s'$. Ele não conhece a pista nem a função de recompensa de antemão — precisa **aprender por interação**.

A pergunta central do RL é: **qual ação tomar em cada estado para maximizar a recompensa acumulada ao longo do tempo?**

O Q-Learning responde estimando, para cada par $(s, a)$, um número $Q(s, a)$ que representa **o quão boa é a ação $a$ no estado $s$, se a partir daí o agente jogar de forma ótima**. Essa estimativa é refinada a cada interação, e a política aprendida é simplesmente "no estado $s$, escolha a ação $a$ que maximiza $Q(s, a)$".

---

## 2. Matemática

### 2.1 Retorno e função de valor

Defina o **retorno descontado** a partir do passo $t$ como:

$$
G_t \;=\; r_{t+1} + \gamma\, r_{t+2} + \gamma^2 r_{t+3} + \cdots \;=\; \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
$$

onde $\gamma \in [0, 1)$ é o **fator de desconto**. $\gamma$ próximo de 1 valoriza muito o futuro distante; $\gamma$ próximo de 0 faz o agente míope (só olha recompensa imediata). No EP usamos $\gamma = 0{,}99$ — futuro distante importa, mas há um leve incentivo a chegar logo.

A **função de valor-ação ótima** é:

$$
Q^{*}(s, a) \;=\; \mathbb{E}\!\left[\,G_t \;\middle|\; s_t = s,\; a_t = a,\; \text{seguindo a política ótima depois}\right]
$$

Ou seja: a recompensa esperada se eu tomar a ação $a$ no estado $s$ e depois agir otimamente.

### 2.2 Equação de Bellman ótima

A função $Q^{*}$ satisfaz a **equação de Bellman ótima**:

$$
Q^{*}(s, a) \;=\; \mathbb{E}\!\left[\,r + \gamma \max_{a'} Q^{*}(s', a')\,\right]
$$

Em palavras: o valor ótimo de $(s, a)$ é a recompensa imediata $r$ mais o desconto vezes o **melhor valor que posso obter no próximo estado** $s'$. Esse `max` é a chave: ele encarna a hipótese "depois eu jogo otimamente".

### 2.3 Atualização TD do Q-Learning

Como o agente não conhece nem a função de transição nem a recompensa esperada, ele **estima** $Q^{*}$ por amostragem. Cada vez que observa uma transição $(s, a, r, s')$, atualiza a estimativa atual $Q(s, a)$ na direção do **alvo TD** ("Temporal-Difference"):

$$
Q(s, a) \;\leftarrow\; Q(s, a) \;+\; \alpha\,\Big[\underbrace{r + \gamma \max_{a'} Q(s', a')}_{\text{alvo TD}} \;-\; Q(s, a)\Big]
$$

- $\alpha \in (0, 1]$ é a **taxa de aprendizado**. Controla o tamanho do passo. $\alpha = 0{,}1$ (valor sugerido) significa "mova 10% na direção do alvo".
- $r + \gamma \max_{a'} Q(s', a')$ é o que **deveria** ser $Q(s, a)$ segundo a equação de Bellman, usando a estimativa atual.
- $r + \gamma \max_{a'} Q(s', a') - Q(s, a)$ é o **erro TD**: quanto a estimativa atual está errada.

> **Por que `max` e por que isso é "off-policy"?** O Q-Learning sempre usa $\max_{a'}$ no alvo, independente da ação que o agente vai realmente tomar no próximo passo. Isso significa que ele aprende sobre a política **gulosa** (`argmax Q`) enquanto pode estar **explorando** com uma política diferente (ex.: $\varepsilon$-greedy). Por isso é classificado como **off-policy**: a política que está sendo melhorada não precisa ser a mesma usada para coletar dados.

### 2.4 Caso terminal

Se $s'$ é terminal (chegada ou colisão), não há "próximo estado" — o $\max$ não faz sentido. A regra fica:

$$
Q(s, a) \;\leftarrow\; Q(s, a) \;+\; \alpha\,\big[r - Q(s, a)\big]
$$

Equivalentemente, defina $\max_{a'} Q(s', a') = 0$ quando $s'$ é terminal. No código, isso é controlado pela flag `terminated` que o `env.step()` retorna.

### 2.5 Exploração: $\varepsilon$-greedy

Se o agente sempre escolhesse a ação `argmax Q(s, a)` desde o início (quando $Q$ ainda está zerada/aleatória), ele ficaria preso em comportamentos ruins. Precisa **explorar**. A política $\varepsilon$-greedy é:

$$
a \;=\; \begin{cases}
\text{ação aleatória uniforme} & \text{com probabilidade } \varepsilon \\
\arg\max_{a'} Q(s, a') & \text{com probabilidade } 1 - \varepsilon
\end{cases}
$$

No EP, $\varepsilon$ **decai linearmente** de $1{,}0$ (totalmente aleatório no início) a $0{,}05$ (quase só guloso no final) ao longo dos primeiros 80% dos episódios. Os 20% finais ficam em $\varepsilon = 0{,}05$ — exploração residual constante para garantir robustez.

### 2.6 Garantias teóricas

O Q-Learning **converge para $Q^{*}$ com probabilidade 1** se:
1. Todas as transições $(s, a)$ forem visitadas infinitas vezes.
2. A taxa de aprendizado $\alpha_t$ decair adequadamente (na prática, $\alpha$ constante e moderado funciona bem).

Em ambientes contínuos como este, "todos os estados visitados" depende da discretização e do schedule de $\varepsilon$. Por isso $K = 5$ (estado-espaço manejável) e $\varepsilon$ alto no início (cobertura) são tão importantes.

---

## 3. Algoritmo

Pseudocódigo do loop de treinamento:

```
Entrada: K (discretização), α, γ, ε_inicial, ε_final, N (episódios), T (passos máx.)
Saída:   tabela Q

Inicializar Q[s, a] = 0  para todo (s, a)

para ep = 1, 2, ..., N:
    ε ← schedule_linear(ep, ε_inicial, ε_final, 0.8 * N)
    obs ← env.reset()
    s ← discretizar(obs)
    para passo = 1, 2, ..., T:
        # Política ε-greedy
        com probabilidade ε:
            a ← ação aleatória uniforme
        senão:
            a ← argmax_{a'} Q[s, a']

        obs', r, terminou, truncou, _ ← env.step(a)
        s' ← discretizar(obs')

        # Atualização TD
        se terminou:
            alvo ← r
        senão:
            alvo ← r + γ * max_{a'} Q[s', a']
        Q[s, a] ← Q[s, a] + α * (alvo - Q[s, a])

        s ← s'
        se terminou ou truncou:
            break
```

Depois do treinamento, a **política aprendida** é simplesmente a gulosa: $\pi(s) = \arg\max_{a} Q(s, a)$. Para avaliar, rode episódios com $\varepsilon = 0$.

---

## 4. Implementação em Python

### 4.1 Estrutura de dados para $Q$

Há duas opções idiomáticas:

**Opção A — dicionário** (recomendado por flexibilidade):

```python
from collections import defaultdict
import numpy as np

n_actions = 5
Q = defaultdict(lambda: np.zeros(n_actions))

# acesso
chave = discretizar(obs)       # tupla de 6 ints
valores = Q[chave]             # np.array de tamanho 5
melhor_acao = int(np.argmax(valores))
```

Vantagem: só aloca memória para estados realmente visitados. Com $K=5$ e 6 dimensões, o teórico é $5^6 = 15{.}625$ estados, mas na prática o agente visita muito menos — o dicionário economiza memória.

**Opção B — array NumPy multidimensional**:

```python
import numpy as np

K, n_actions = 5, 5
Q = np.zeros((K, K, K, K, K, K, n_actions))   # shape (5,5,5,5,5,5,5)
valores = Q[chave]             # chave é tupla de 6 ints
```

Vantagem: indexação mais rápida. Desvantagem: aloca 78.125 floats mesmo sem visitar todos.

Para este EP, qualquer uma serve. A Opção A é mais natural se você quer reportar "estados populados" no relatório.

### 4.2 Discretização

```python
def discretizar(obs, K=5):
    """obs ∈ [0, 1]^6 → tupla de 6 ints em {0, ..., K-1}."""
    return tuple(min(int(v * K), K - 1) for v in obs)
```

O `min(..., K - 1)` protege o caso $v = 1.0$ exatamente (sem isso, daria índice $K$, fora do range).

### 4.3 Schedule linear de $\varepsilon$

```python
def schedule_epsilon(ep, eps_inicial=1.0, eps_final=0.05, eps_decai_em=24_000):
    """Decai linearmente de eps_inicial a eps_final em eps_decai_em episódios."""
    if ep >= eps_decai_em:
        return eps_final
    frac = ep / eps_decai_em
    return eps_inicial + frac * (eps_final - eps_inicial)
```

Com 30.000 episódios e decaimento em 80%, `eps_decai_em = 24_000`.

### 4.4 Política $\varepsilon$-greedy

```python
import random

def escolher_acao(Q, s, eps, n_actions):
    if random.random() < eps:
        return random.randrange(n_actions)
    return int(np.argmax(Q[s]))
```

### 4.5 Atualização TD

```python
def atualizar(Q, s, a, r, s_prox, terminou, alpha=0.1, gamma=0.99):
    if terminou:
        alvo = r
    else:
        alvo = r + gamma * np.max(Q[s_prox])
    Q[s][a] += alpha * (alvo - Q[s][a])
```

### 4.6 Loop completo

```python
def treinar(env, K=5, alpha=0.1, gamma=0.99,
            n_episodios=30_000, eps_inicial=1.0, eps_final=0.05):
    Q = defaultdict(lambda: np.zeros(env.n_actions))
    eps_decai_em = int(0.8 * n_episodios)
    historico_rewards = []

    for ep in range(n_episodios):
        eps = schedule_epsilon(ep, eps_inicial, eps_final, eps_decai_em)
        obs = env.reset()
        s = discretizar(obs, K)
        reward_total = 0.0

        for _ in range(env.max_steps):
            a = escolher_acao(Q, s, eps, env.n_actions)
            obs_prox, r, terminou, truncou, _ = env.step(a)
            s_prox = discretizar(obs_prox, K)
            atualizar(Q, s, a, r, s_prox, terminou, alpha, gamma)
            s = s_prox
            reward_total += r
            if terminou or truncou:
                break

        historico_rewards.append(reward_total)

    return Q, historico_rewards
```

### 4.7 Avaliação (política gulosa)

```python
def avaliar(env, Q, K=5, n_episodios=10):
    """Roda n_episodios com ε = 0 e retorna estatísticas médias."""
    resultados = []
    for _ in range(n_episodios):
        obs = env.reset()
        s = discretizar(obs, K)
        reward_total = 0.0
        velocidades = []
        n_passos = 0
        sucesso = False
        for _ in range(env.max_steps):
            a = int(np.argmax(Q[s]))   # ε = 0
            obs, r, terminou, truncou, info = env.step(a)
            s = discretizar(obs, K)
            reward_total += r
            velocidades.append(obs[5] * env.V_MAX)  # desnormaliza
            n_passos += 1
            if terminou or truncou:
                sucesso = info.get("chegada", False)
                break
        resultados.append({
            "n_passos": n_passos,
            "reward": reward_total,
            "velocidade_media": float(np.mean(velocidades)),
            "sucesso": sucesso,
        })
    return resultados
```

---

## 5. Hiperparâmetros sugeridos

Use estes valores como ponto de partida:

| Hiperparâmetro | Valor | Justificativa |
|---|---|---|
| **Episódios de treinamento** | 30.000 | Suficiente para convergência em `pista_03.txt` com $K=5$. |
| **Limite de passos por episódio** | 500 | Evita episódios infinitos quando o agente fica girando em círculos. |
| **Discretização $K$** | 5 | Ver §3.2 do `README.md` para a justificativa (casa com os 5 níveis de velocidade). |
| **Taxa de aprendizado $\alpha$** | 0,1 | Padrão de Sutton & Barto. Suficientemente pequeno para não oscilar, grande o bastante para aprender rápido. |
| **Desconto $\gamma$** | 0,99 | Valoriza chegar à meta (recompensa de +500) mesmo que esteja muitos passos no futuro. |
| **Exploração $\varepsilon$** | $1{,}0 \to 0{,}05$ linear em 80% dos episódios | Garante exploração ampla no início e refinamento da política no final. |

### Como variar (se quiser experimentar)

- **$\alpha$ maior (0,3–0,5):** aprende mais rápido, mas pode oscilar perto do ótimo.
- **$\gamma$ menor (0,9):** agente fica mais míope, prioriza progresso imediato. Pode falhar em pistas longas.
- **$\varepsilon$ decai mais devagar (em 90% ou 95%):** mais exploração total, mais robusto, mas mais lento para convergir.
- **Episódios:** se o histórico de recompensas ainda está subindo no final, treine mais. Se já estabilizou, pode parar antes.

---

## 6. Dicas práticas e debugging

1. **Comece com a `pista_01.txt`** (a mais fácil). Se seu Q-Learning não aprende a fazer uma reta com curva única, há bug — não passe para pistas mais difíceis ainda.
2. **Plote o histórico de recompensas** (média móvel de 100). A curva deve subir: começa em ~$-100$ a $-50$ (muitas colisões) e estabiliza em algo positivo (passa a chegar à linha de chegada com bônus +500).
3. **Visualize a política treinada** com `renderizar_episodio` em `src/visualize.py`. Ver o carro em ação revela bugs que números não revelam.
4. **Se a curva fica plana em $-100$:** o agente sempre colide. Aumente $\varepsilon$ inicial, aumente o número de episódios, ou comece em pista mais simples.
5. **Se a curva sobe mas a política avaliada bate na parede:** provavelmente bug na avaliação — confira que está usando $\varepsilon = 0$ (gulosa) e a mesma função de discretização do treinamento.
6. **Salve o modelo** via `pickle` ao final do treinamento (ver [`anexo_b_pickle.md`](anexo_b_pickle.md)). Treinar 30.000 episódios pode levar vários minutos.
7. **Use seed fixa** (`SEED = 42`) para reprodutibilidade.

---

## 7. Referências

- **Sutton & Barto**, *Reinforcement Learning: An Introduction*, 2ª ed., 2018. Capítulo 6 (Temporal-Difference Learning), seção 6.5 (Q-Learning). Disponível gratuitamente em http://incompleteideas.net/book/the-book.html.
- **Watkins, C. J. C. H.** (1989). *Learning from Delayed Rewards*. Tese de doutorado, Cambridge University. Trabalho original que introduziu o Q-Learning.
