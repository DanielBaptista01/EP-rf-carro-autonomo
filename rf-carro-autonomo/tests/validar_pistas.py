"""Valida todas as pistas: parser OK, BFS atinge a chegada, mostra estatísticas."""

import sys
sys.path.insert(0, "src")

from pathlib import Path
from track import carregar_pista, PAREDE
from env import calcular_campo_progresso

DIR_PISTAS = Path("pistas")
arquivos = sorted(DIR_PISTAS.glob("pista_*.txt"))

print(f"{'Pista':<20} {'Tamanho':<12} {'Pilotáveis':<12} {'BFS atinge fim?':<20} {'Distância (passos)'}")
print("-" * 90)

tudo_ok = True
for arq in arquivos:
    try:
        grid, largada, chegada = carregar_pista(arq)
        progresso = calcular_campo_progresso(grid, largada)
        cy, cx = chegada
        dist = progresso[cy, cx]
        pilotaveis = int((grid != PAREDE).sum())
        if dist >= 0:
            status = "OK"
        else:
            status = "FALHA"
            tudo_ok = False
        print(f"{arq.name:<20} {grid.shape!s:<12} {pilotaveis:<12} {status:<20} {dist}")
    except Exception as e:
        tudo_ok = False
        print(f"{arq.name:<20} ERRO: {e}")

print("-" * 90)
print("TODAS OK" if tudo_ok else "ALGUMAS FALHARAM")
