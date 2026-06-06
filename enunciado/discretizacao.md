# Discretização do Estado

O Q-Learning tabular precisa de uma **tabela $Q[s, a]$** indexada por estados discretos. Mas a observação que `env.step()` devolve é um vetor de 6 floats em $[0, 1]$ — há infinitos estados possíveis. Para tornar a tabela manejável, precisamos **discretizar** o vetor: mapear cada float em um inteiro pequeno.

Este documento explica:

1. A técnica de discretização usada (binning uniforme).
2. Como o trade-off de $K$ funciona.
3. Por que $K = 5$ neste EP.
4. Um exemplo passo a passo de como o vetor de 6 floats vira a chave da tabela $Q$.

---

## 1. A técnica: binning uniforme

A ideia é simples — divida o intervalo $[0, 1]$ em $K$ pedaços iguais (baldes) e numere-os de $0$ a $K-1$. Para um valor $v \in [0, 1]$, descubra em qual balde ele cai.

A fórmula:

```python
def discretizar(obs, K=5):
    return tuple(min(int(v * K), K - 1) for v in obs)
```

Passo a passo, para cada componente $v$:

1. **Multiplica por $K$:** leva $[0, 1] \to [0, K]$.
2. **`int(...)`:** trunca para o inteiro abaixo. Resultado em $\{0, 1, \dots, K\}$ — o valor $K$ só aparece se $v$ era exatamente $1{,}0$.
3. **`min(\cdot,\ K - 1)`:** clamp na borda direita. Garante que $v = 1{,}0$ caia no balde $K - 1$ em vez de criar um $K$-ésimo balde com um único ponto.

### A régua dos baldes (com $K = 5$)

```
   balde 0       balde 1       balde 2       balde 3       balde 4
┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
│ [0.0, 0.2)  │ [0.2, 0.4)  │ [0.4, 0.6)  │ [0.6, 0.8)  │ [0.8, 1.0]  │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
0.0           0.2           0.4           0.6           0.8           1.0
```

O balde 4 inclui o extremo $1{,}0$ por causa do `min`.

### Mini-quiz

| $v$ | Conta | Balde |
| --- | --- | --- |
| 0,00 | `min(int(0.00 * 5), 4) = min(0, 4)` | **0** |
| 0,15 | `min(int(0.75), 4) = min(0, 4)` | **0** |
| 0,20 | `min(int(1.00), 4) = min(1, 4)` | **1** |
| 0,59 | `min(int(2.95), 4) = min(2, 4)` | **2** |
| 0,60 | `min(int(3.00), 4) = min(3, 4)` | **3** |
| 0,99 | `min(int(4.95), 4) = min(4, 4)` | **4** |
| 1,00 | `min(int(5.00), 4) = min(5, 4)` | **4** (clamp) |

---

## 2. O trade-off de $K$

Há tensão entre dois extremos:

- **$K$ pequeno demais:** baldes largos. Estados muito diferentes caem no mesmo balde — o agente "vê tudo igual" e não consegue aprender política refinada. Diz-se que perdemos **resolução**.
- **$K$ grande demais:** baldes finos. Quase todo estado vira único, e o agente precisa visitar cada um várias vezes para aprender — com poucos episódios, a maioria dos estados fica subtreinada. Esse é o problema da **explosão de estados**.

### Tamanho da tabela em função de $K$

O vetor tem **6 dimensões** (5 raios + velocidade). O número máximo de estados discretos é $K^6$; com 5 ações, são $K^6 \times 5$ entradas em $Q$.

| $K$ | Estados $K^6$ | Entradas em $Q$ | Diagnóstico |
| --- | --- | --- | --- |
| 3 | 729 | 3.645 | Grosseiro demais. "Colado na parede" e "perto da parede" caem no mesmo balde — o agente bate com frequência. |
| **5** | **15.625** | **78.125** | **Sweet spot deste EP.** Veja §3 abaixo. |
| 8 | ≈ 262 mil | ≈ 1,3 milhão | Muitos estados nunca visitados em 30 mil episódios. Aprendizado fica lento sem ganho prático. |
| 10 | 1.000.000 | 5.000.000 | Inviável com 30 mil episódios. |

> Nem todo estado teoricamente possível é alcançável. O carro nunca encontra uma combinação arbitrária de sensores e velocidade — só as que aparecem em trajetórias válidas. Na prática, com $K = 5$ na pista 03, esperam-se centenas a alguns milhares de estados visitados — uma fração dos 15.625 possíveis.

---

## 3. Por que $K = 5$ neste EP

Três razões convergem para $K = 5$:

### 3.1 Casa exatamente com a granularidade da velocidade

O carro tem $V_\max = 2{,}0$ e incrementos de $0{,}5$. Portanto $v$ assume apenas **5 valores discretos**:

$$
v \in \{0;\ 0{,}5;\ 1{,}0;\ 1{,}5;\ 2{,}0\}
$$

Normalizada:

$$
v_\text{norm} \in \{0{,}00;\ 0{,}25;\ 0{,}50;\ 0{,}75;\ 1{,}00\}
$$

Com $K = 5$, cada valor cai em um balde **distinto**:

| $v_\text{norm}$ | Balde |
| --- | --- |
| 0,00 | 0 |
| 0,25 | 1 |
| 0,50 | 2 |
| 0,75 | 3 |
| 1,00 | 4 |

Nenhuma agregação, nenhuma fragmentação. Com $K \in \{3, 4\}$, valores físicos distintos colidiriam no mesmo balde (perda de informação). Com $K \in \{8, 10\}$, os baldes seriam mais finos do que a física consegue distinguir.

### 3.2 Resolução adequada para o LIDAR

Cada balde cobre **20% do alcance máximo** do sensor — ou seja, **2 células** de 10. Isso permite ao agente distinguir 5 níveis qualitativos:

- **Balde 0** (0-2 céls): "colado na parede".
- **Balde 1** (2-4 céls): "perto".
- **Balde 2** (4-6 céls): "meio caminho".
- **Balde 3** (6-8 céls): "longe".
- **Balde 4** (8+ céls ou saturado): "sem parede à vista".

Suficiente para o agente decidir "freio agora vs. acelero mais", e grosseiro o bastante para generalizar entre posições próximas.

### 3.3 Tamanho de tabela manejável

Com $5^6 = 15{.}625$ estados possíveis e 30.000 episódios de treinamento, cada estado **realmente alcançado** é visitado dezenas de vezes — suficiente para o $Q$ convergir. Com $K = 10$ seriam 64× mais estados, e o mesmo orçamento de episódios deixaria a maioria deles subtreinada.

---

## 4. Da observação à chave: exemplo concreto

Considere um cenário em **pista 03**: o carro saiu da largada, andou pelo corredor superior, está a meia velocidade ($v = 1{,}0$) e se aproxima do início da curva para o sul.

**Posição:** $(x, y) = (6{,}5,\ 2{,}5)$, $\theta = 0$ (leste), $v = 1{,}0$.

```
col:    0  1  2  3  4  5  6  7  8  9  10
      🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱        ← row 0
      🧱 ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ 🧱        ← row 1
      🧱 🟢 ⚪ ⚪ ⚪ ⚪ ➡️ ⚪ ⚪ ⚪ 🧱        ← row 2 (carro aqui)
      🧱 ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ 🧱  row 3
      🧱 ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ 🧱  row 4
      🧱 🧱 🧱 🧱 🧱 🧱 🧱 ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪      row 5
                                ↑ corredor continua para sul
```

Os 5 raios partem do carro nas direções $\theta + \alpha$ para $\alpha \in \{0°,\ +30°,\ -30°,\ +60°,\ -60°\}$. Como $\theta = 0$ (leste) e $y$ cresce para baixo, ângulos positivos apontam para o sul:

```
                  d_-60       d_-30
                    ↖           ↗
                      ↖       ↗
                        ↖   ↗
                         ➡️ ──────── d_0 ────→
                        ↙   ↘
                      ↙       ↘
                    ↙           ↘
                  d_+60       d_+30
```

### Passo 1 — `env._observar()` lança 5 raios e mede o que encontra

| Sensor | Aponta para | Distância (céls) | Normalizado ($\div 10$) |
| --- | --- | --- | --- |
| $d_0$ | leste (frontal) — parede em col 10 | 3,5 | 0,35 |
| $d_{+30}$ | sudeste — corredor abre, sem parede em alcance | 10,0 (satura) | 1,00 |
| $d_{-30}$ | nordeste — parede norte (row 0) próxima | 3,0 | 0,30 |
| $d_{+60}$ | sul-sudeste — encontra a parede de row 5 | 4,1 | 0,41 |
| $d_{-60}$ | norte-nordeste — parede de row 0 | 1,8 | 0,18 |

Velocidade: $v_\text{norm} = 1{,}0 / 2{,}0 = 0{,}5$.

### Passo 2 — Vetor de observação (saída de `env.step()`)

```
obs = [0.35, 1.00, 0.30, 0.41, 0.18, 0.50]
       d_0   d+30  d-30  d+60  d-60  v_norm
```

### Passo 3 — Discretização (`discretizar(obs)`)

| Componente | $v$ | $v \times 5$ | `int(\cdot)` | `min(\cdot,\ 4)` |
| --- | --- | --- | --- | --- |
| $d_0$ | 0,35 | 1,75 | 1 | **1** |
| $d_{+30}$ | 1,00 | 5,00 | 5 | **4** (clamp) |
| $d_{-30}$ | 0,30 | 1,50 | 1 | **1** |
| $d_{+60}$ | 0,41 | 2,05 | 2 | **2** |
| $d_{-60}$ | 0,18 | 0,90 | 0 | **0** |
| $v_\text{norm}$ | 0,50 | 2,50 | 2 | **2** |

### Passo 4 — Chave final

```
chave = (1, 4, 1, 2, 0, 2)
```

Essa tupla é o índice em `Q`: `Q[chave]` é um `np.ndarray` de 5 elementos (um por ação), e $\arg\max_a Q[\text{chave}][a]$ é a melhor ação a tomar nesse estado segundo o que o agente aprendeu até agora.

---

## 5. Implementação na sua solução

Coloque um método `discretizar` no seu agente e use-o sempre — tanto ao escolher ação quanto ao fazer update:

```python
class AgenteQLearning:
    def __init__(self, ...):
        self.Q = {}              # dict: chave_discreta → np.ndarray(5)
        self.K = 5

    def discretizar(self, obs):
        return tuple(min(int(v * self.K), self.K - 1) for v in obs)

    def escolher_acao(self, obs):
        chave = self.discretizar(obs)
        if chave not in self.Q:
            self.Q[chave] = np.zeros(self.n_actions)
        # ε-greedy sobre Q[chave]
        ...

    def atualizar(self, s, a, r, s_prox, terminou):
        chave_s = self.discretizar(s)
        chave_sp = self.discretizar(s_prox)
        ...
```

**Duas armadilhas comuns:**

- **Esquecer de discretizar antes de indexar `Q`.** O vetor float não é uma chave válida para `dict[tuple, np.array]`. Sempre converta primeiro.
- **Inicializar $Q[s]$ sob demanda.** Como a maioria dos estados nunca aparece, alocar todas as $5^6$ entradas com zeros é desperdício. Use `if chave not in self.Q: self.Q[chave] = np.zeros(5)` ou um `defaultdict(lambda: np.zeros(5))`.
