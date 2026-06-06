# Anexo C: A Velocidade do Carro

Velocidade ($v$) parece um conceito óbvio, mas no contexto deste EP tem sutilezas que vale entender bem — porque é justamente o componente do estado que torna o problema interessante (e difícil).

## C.1 Definição operacional

A velocidade $v$ é um **escalar não-negativo** que mede quanto o carro avança por passo de simulação, na direção em que ele está apontando. A cada chamada de `env.step(action)`, o carro se move:

$$
x_{novo} = x + v \cdot \cos\theta
$$

$$
y_{novo} = y + v \cdot \sin\theta
$$

A direção do movimento vem do ângulo $\theta$. A velocidade é só a **magnitude** do deslocamento.

> 💡 Em física tradicional, velocidade é um vetor (magnitude + direção). Aqui, separamos as duas componentes: $v$ guarda a magnitude, $\theta$ guarda a direção. Isso simplifica a física e o controle.

## C.2 Unidade e limites

A unidade é **células do grid por passo de simulação**.

- $v = 1{,}0$ → o carro atravessa **uma célula inteira a cada passo**.
- $v = 0{,}5$ → meia célula por passo.
- $v = 0$ → parado.
- $v = V_{\max} = 2{,}0$ → duas células por passo (limite máximo).

Não há unidade de tempo “real” no problema — um “passo” é uma unidade abstrata.

## C.3 Como o agente controla a velocidade

A velocidade muda **apenas por ação do agente**. Não há fricção, não há inércia continuada — se o agente não fizer nada, $v$ permanece igual.

| Ação | Efeito em `v` |
| --- | --- |
| 0 (nada) | `v` inalterada |
| 1 (acelerar) | `v ← min(v + 0,5, V_max)` |
| 2 (frear) | `v ← max(v - 0,5, 0)` |
| 3 (virar esquerda) | `v` inalterada (só `θ` muda) |
| 4 (virar direita) | `v` inalterada (só `θ` muda) |

Isso é uma **física idealizada** — em um carro real, frenagem leva tempo, há fricção do ar, há inércia. No EP, ignoramos tudo isso para simplificar o aprendizado.

## C.4 Sem marcha-ré

Note: $v \in [0,\ V_{\max}]$. **Nunca negativa**. O carro só vai para frente; para mudar de direção, precisa virar. Isso modela um carro de F1, não um carro de rua.

## C.5 Velocidade no estado observável

A velocidade entra na **observação do agente** como o sexto componente do vetor de estado:

```
estado = [d_0, d_+30, d_-30, d_+60, d_-60, v_norm]
```

onde $v_{norm} = v / V_{\max}$ — normalizada para $[0, 1]$, igual aos sensores LIDAR. Isso é importante: tabelas com discretização funcionam melhor quando todas as features estão na mesma escala.

> 💡 **Por que o agente precisa saber a própria velocidade?** Porque a melhor ação depende dela. Andando devagar perto de uma curva, é seguro continuar acelerando. Andando rápido, é prudente frear antes. Sem ver a velocidade, o agente não conseguiria distinguir essas situações.

## C.6 Por que velocidade é o componente sutil do problema

Os 5 sensores LIDAR são “fáceis de entender” — distâncias até paredes. A velocidade é mais traiçoeira por três motivos:

### 1. Tem efeito acumulativo

Acelerar uma vez muda $v$ em apenas $+0{,}5$ — efeito imediato pequeno. Mas o efeito **se mantém** em todos os passos seguintes: o carro vai continuar andando mais rápido até alguém frear. Isso é diferente de virar (efeito imediato no ângulo) e dos sensores (refletem o estado atual).

### 2. Cria dilemas de longo prazo

Acelerar dá recompensa imediata maior (mais $\Delta$progresso por passo). Mas se você acelerar antes de uma curva, vai bater na parede e perder $-100$. O agente precisa aprender que **às vezes é certo desacelerar mesmo perdendo progresso imediato**, antecipando a curva. Esse é o **problema clássico de crédito temporal** (*temporal credit assignment*) que torna RL difícil em geral, e que aparece de forma vívida aqui.

### 3. Interage com o ângulo de virada

A virada é em ângulo absoluto ($\theta \pm 30°$), independente da velocidade. Mas o **raio da curva resultante** depende de $v$:

- Velocidade baixa + virada de 30° = curva apertada, raio pequeno.
- Velocidade alta + virada de 30° = curva larga, raio grande.

O agente precisa **coordenar** velocidade e virada para fazer curvas que caibam no corredor da pista. Se acelerar muito antes de uma curva, mesmo virando o carro vai sair pela tangente e bater.

## C.7 Implicações práticas para o seu agente

1. **Não basta aprender a virar — precisa aprender a desacelerar antes de curvas.** Essa é a habilidade mais difícil que o agente vai dominar, e geralmente é a última a emergir no treinamento.
2. **Curvas de aprendizado “boas mas não ótimas”** geralmente refletem que o agente aprendeu a chegar ao fim mas não otimizou velocidade. Anda devagar o tempo todo (seguro), nunca atinge $V_{\max}$. Política funcional, mas conservadora.
3. **Para depurar visualmente:** rode `renderizar_episodio` no `src/visualize.py` para ver o carro correndo a pista no seu terminal.
    - Se o carro **anda na velocidade máxima sempre e bate**: o problema é aprender a frear.
    - Se o carro **anda devagar sempre e nunca bate mas demora muito**: o problema é aprender a acelerar nas retas.
    - Se o carro **acelera nas retas e freia antes das curvas**: parabéns, está bem treinado.
4. **No relatório:** vale reportar a **velocidade média** atingida pela política treinada. É um indicador de quão “agressiva” é a política aprendida — uma política conservadora privilegia segurança sobre velocidade; uma agressiva persegue $V_{\max}$ a maior parte do tempo.
