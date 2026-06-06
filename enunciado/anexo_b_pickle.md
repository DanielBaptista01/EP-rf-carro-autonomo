# Anexo B: Salvando modelos treinados com `pickle`

Como o EP exige salvar os modelos treinados no diretório `/treinamento`, este anexo explica como fazer isso usando `pickle` — uma biblioteca da **biblioteca padrão do Python** (não precisa instalar nada).

## B.1 O que é pickle

`pickle` é o módulo do Python que **serializa** objetos: transforma uma estrutura de dados Python (dicionário, lista, classe, array NumPy) em uma sequência de bytes que pode ser salva em arquivo e, depois, **desserializada** de volta no estado original.

Em uma frase: pickle congela um objeto Python em disco, e descongela quando você quiser.

> 💡 **Analogia:** é como tirar uma “foto” do seu modelo treinado e poder revivê-la depois sem precisar re-treinar tudo.

## B.2 Uso básico

**Salvar um objeto:**

```python
import pickle

dados = {"q_table": minha_q_table, "config": {"alpha": 0.1, "gamma": 0.99}}

with open("treinamento/qlearning.pkl", "wb") as f:   # "wb" = write binary
    pickle.dump(dados, f)
```

**Carregar um objeto:**

```python
import pickle

with open("treinamento/qlearning.pkl", "rb") as f:   # "rb" = read binary
    dados = pickle.load(f)

minha_q_table = dados["q_table"]
config = dados["config"]
```

Pronto. Não há mais nada de fundamental.

## B.3 O que salvar para o Q-Learning

Salve um dicionário com tudo que você precisa para reproduzir o agente. Como o EP usa **um único modelo treinado em round-robin** nas 16 pistas de treino, **o pickle também é único** (`treinamento/qlearning.pkl`):

```python
estado_para_salvar = {
    "q_table": agent.q_table,           # dict {chave_discreta: array} ou np.ndarray
    "discretization_K": 5,              # configuração da discretização
    "n_episodes_trained": 480_000,      # total de episódios (round-robin)
    "rewards_history": rewards,         # lista de recompensas por episódio (todas as pistas)
    "rewards_por_pista": {              # opcional, mas útil para o relatório
        "pistas/pista_01.txt": [...],
        # ...
        "pistas/pista_16.txt": [...],
    },
    "config": {"alpha": 0.1, "gamma": 0.99, "eps_start": 1.0, "eps_end": 0.05},
    "seed": 42,
    "tracks_used": [f"pistas/pista_{i:02d}.txt" for i in range(1, 17)],  # 01..16
}
with open("treinamento/qlearning.pkl", "wb") as f:
    pickle.dump(estado_para_salvar, f)
```

> **Importante:** as pistas 17 e 18 são **holdout** — não devem aparecer em `tracks_used`. O professor verifica isso ao avaliar a generalização.

## B.4 Lógica recomendada para `solucao.py`

Implemente a seguinte lógica para evitar re-treinar a cada execução:

```python
import os
import pickle
from pathlib import Path

TREINAMENTO_DIR = Path("treinamento")
TREINAMENTO_DIR.mkdir(exist_ok=True)

def treinar_ou_carregar(nome, treinar_fn, recarregar=False):
    """
    Se 'treinamento/{nome}.pkl' existe e recarregar=False, carrega.
    Caso contrário, chama treinar_fn() e salva o resultado.
    """
    arquivo = TREINAMENTO_DIR / f"{nome}.pkl"
    if arquivo.exists() and not recarregar:
        print(f"Carregando {arquivo} ...")
        with open(arquivo, "rb") as f:
            return pickle.load(f)
    else:
        print(f"Treinando {nome} ...")
        resultado = treinar_fn()
        with open(arquivo, "wb") as f:
            pickle.dump(resultado, f)
        print(f"Salvo em {arquivo}")
        return resultado

# Uso:
modelo = treinar_ou_carregar("qlearning", lambda: treinar_round_robin(pistas_treino, K=5))
```

Para forçar re-treinamento (útil ao depurar), passe `recarregar=True` ou simplesmente delete o arquivo `.pkl`.

## B.5 Cuidados com pickle

- **Nunca abra um pickle de fonte desconhecida.** O processo de unpickling pode executar código arbitrário — é um vetor clássico de ataque. Para os modelos que você mesmo gerou, sem problema.
- **Compatibilidade entre versões do Python:** pickles em Python 3.10+ são geralmente intercompatíveis, mas pickles entre Python 2 e 3 quebram. Para este EP isso não é problema (use Python 3.10+).
- **Tamanho dos arquivos:** tabelas $Q$ tabulares costumam ficar entre dezenas e centenas de KB — sem problema para commitar no GitHub.
- **Reprodutibilidade:** salve **junto com o modelo** os hiperparâmetros e a seed usada.

## B.6 Documentação oficial

- [docs.python.org/3/library/pickle.html](https://docs.python.org/3/library/pickle.html) — referência completa.
