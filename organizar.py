import re
import shutil
from collections import Counter
from pathlib import Path

FRAMES_DIR = "frames"          # pasta com os frames extraídos
OUTPUT_DIR = "por_gesto"       # pasta onde as subpastas por gesto serão criadas
MOVER = True                  # False = copia (seguro), True = move os arquivos

# vídeos com mais de um gesto: não dá para saber o gesto pelo nome
MISTURADOS = {"sortidos", "return_down"}

# exemplos: 5_left_0012.jpg -> left | 13_neutro_0003.jpg -> neutro
#           1_yaw_right_0001.jpg -> yaw_right | 1_neutro__0000.jpg -> neutro
PADRAO = re.compile(r"^\d+_(?P<gesto>.+?)_*_\d{4}$")

pasta = Path(FRAMES_DIR)
if not pasta.is_dir():
    raise SystemExit(f"Pasta '{FRAMES_DIR}' não encontrada em {Path.cwd()}.")

saida = Path(OUTPUT_DIR)
contagem = Counter()
sem_padrao = []

for arq in sorted(pasta.glob("*.jpg")):
    m = PADRAO.match(arq.stem)
    if not m:
        sem_padrao.append(arq.name)
        continue

    gesto = m.group("gesto").strip("_").lower()
    if gesto in MISTURADOS:
        gesto = "_misturados"

    destino = saida / gesto
    destino.mkdir(parents=True, exist_ok=True)

    if MOVER:
        shutil.move(str(arq), str(destino / arq.name))
    else:
        shutil.copy2(arq, destino / arq.name)
    contagem[gesto] += 1

print(f"{'Movidos' if MOVER else 'Copiados'} para '{OUTPUT_DIR}':\n")
for gesto, n in sorted(contagem.items()):
    print(f"  {gesto:<16} {n:>5}")
print(f"\nTotal: {sum(contagem.values())} imagens em {len(contagem)} pastas.")

if sem_padrao:
    print(f"\nAtenção: {len(sem_padrao)} arquivos com nome fora do padrão (não foram organizados):")
    for nome in sem_padrao[:10]:
        print(f"  {nome}")
    if len(sem_padrao) > 10:
        print("  ...")
