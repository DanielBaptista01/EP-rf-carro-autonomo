# EP - Carro Autônomo com Aprendizado por Reforço

Este projeto consiste na implementação de um agente inteligente capaz de pilotar um veículo autônomo em pistas bidimensionais. O modelo utiliza Aprendizado por Reforço Tabular (algoritmo Q-Learning Model-Free) para derivar uma política de controle reativa através de sensores locais (LIDAR), operando sem conhecimento prévio da geometria do mapa.

## Integrantes (Dupla)

* Daniel Santos Baptista
* Isaias Maia de Oliveira

## Estrutura do Repositório

* `solucao.py`: O motor principal do projeto que gerencia o agente Q-Learning, a política $\varepsilon$-greedy e a matemática do loop de treinamento.
* `treinamento/qlearning.pkl`: Arquivo serializado contendo o modelo treinado consolidado (Tabela Q) em 480.000 episódios.
* `docs/relatorio.md`: Relatório técnico profundo contendo a justificativa dos hiperparâmetros e a análise crítica sobre o fenômeno de *State Aliasing*.
* `src/`: Diretório base contendo a física do ambiente simulado (`env.py`) e o motor de renderização visual (`visualize.py`).
* `pistas/`: Conjunto de matrizes `.txt` representando as 18 pistas de dificuldade crescente (01 a 16 para treino, 17 e 18 para holdout).
* `q_learning_pista_17.txt` / `18.txt`: Artefatos de saída contendo a telemetria final da avaliação gulosa nas pistas cegas.

## Como Executar e Testar

1. Certifique-se de ter o **Python** (3.10+) instalado em sua máquina.
2. Instale as dependências matemáticas necessárias executando:
   ```bash
   python -m pip install -r requirements.txt
