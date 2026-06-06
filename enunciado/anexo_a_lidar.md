# Anexo A: O que são sensores tipo LIDAR

A §1.4 do `README.md` se refere a “sensores tipo LIDAR” como representação do estado do carro. Este anexo explica o conceito para quem nunca encontrou o termo.

## A.1 LIDAR no mundo real

**LIDAR** é a sigla para **Li**ght **D**etection **a**nd **R**anging — detecção e medição por luz. É um sensor que mede **distâncias** disparando feixes de laser e medindo o tempo que a luz leva para bater num objeto e voltar. A ideia é a mesma do **sonar** (usa som) ou do **radar** (usa ondas de rádio), só que com luz: dispara → reflete → recebe → calcula distância. Como a velocidade da luz é constante e conhecida, **tempo de voo × velocidade da luz / 2** dá a distância até o obstáculo.

**Onde aparece:**

- **Carros autônomos** (Waymo, Cruise): aquele “domo” giratório no teto do carro é um LIDAR.
- **Robôs aspiradores** (Roomba, Roborock): mapeiam sua casa.
- **Drones autônomos**: para evitar obstáculos.
- **iPhones recentes**: têm um mini-LIDAR para realidade aumentada e foco em fotos.
- **Topografia e arqueologia**: aviões com LIDAR mapeiam relevo abaixo de florestas densas.

A saída típica de um LIDAR é um **vetor de distâncias**, uma para cada direção em que ele aponta:

```
Direção        Distância
0°  (frente)   8,2 m
+30°           5,1 m
-30°           12,5 m
+60°           2,4 m
-60°           ∞ (não bateu em nada dentro do alcance)
```

## A.2 LIDAR simulado no EP

No ambiente do EP, o “LIDAR” é **simulado** — não existe luz nem laser de verdade. O que `src/env.py` faz é **ray casting**:

1. A partir da posição do carro $(x, y)$ e seu ângulo $\theta$, emitimos **5 raios** virtuais nas direções $\theta + 0°$, $\theta \pm 30°$, $\theta \pm 60°$.
2. Para cada raio, andamos passo a passo em pequenos incrementos (`step = 0,1` célula), checando se a célula atual é parede.
3. Quando bate numa parede ou ultrapassa o alcance máximo (10 células), registramos a distância percorrida.
4. O **estado do agente** vira um vetor de 6 floats:

```
[d_frente, d_+30°, d_-30°, d_+60°, d_-60°, velocidade_normalizada]
```

Esse é o **único input que o agente vê**. Ele não vê o mapa, não sabe onde está, não sabe onde é a chegada — só sabe “o que tem perto na minha frente e nos meus lados”.

> 💡 É exatamente isso que um carro real “vê” pelo LIDAR físico. A diferença é que aqui simulamos via *ray casting* num grid 2D em vez de usar luz física.

## A.3 Limitações do LIDAR (real e simulado)

Vale conhecer os limites:

- **Vidro e materiais transparentes:** LIDAR real tem dificuldade — o laser atravessa o vidro.
- **Chuva forte, neve, neblina:** as gotículas refletem o laser e geram leituras falsas.
- **Apenas um plano:** um LIDAR 2D só vê uma “fatia” horizontal. Em carros reais isso exige LIDAR 3D ou múltiplos sensores em alturas diferentes.
- **Custo:** um LIDAR automotivo decente custa milhares de dólares. É parte do motivo de a Tesla ter apostado em câmeras + visão computacional em vez de LIDAR.

## A.4 Recursos para aprofundar

- *Wikipedia: LIDAR* — boa visão geral histórica e técnica.
- *Velodyne, Ouster, Luminar* — fabricantes; sites têm whitepapers acessíveis.
- *Self-Driving Cars Specialization* (Coursera, Univ. of Toronto) — curso aborda integração LIDAR + percepção em detalhe.
