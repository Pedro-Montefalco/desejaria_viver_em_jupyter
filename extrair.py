import cv2
import os
from pathlib import Path
 
VIDEOS_DIR = "vídeos"         # pasta com os vídeos (com acento, como a sua)
OUTPUT_DIR = "frames"         # pasta de saída (criada se não existir)
INTERVAL = 1.0                # segundos entre frames salvos
LARGURA = 640                 # largura final (None para manter original)
QUALIDADE = 90                # qualidade do JPEG
FLIP = False                  # só True se a câmera grava espelhada
 
pasta_videos = Path(VIDEOS_DIR)
if not pasta_videos.is_dir():
    raise SystemExit(
        f"Pasta '{VIDEOS_DIR}' não encontrada em {Path.cwd()}. "
        "Rode o script de dentro de ~/dataset_luva ou ajuste VIDEOS_DIR."
    )
 
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
videos = sorted(
    p for p in pasta_videos.iterdir()
    if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv")
)
 
if not videos:
    raise SystemExit(f"Nenhum vídeo encontrado em '{VIDEOS_DIR}'.")
 
print(f"{len(videos)} vídeos encontrados em '{VIDEOS_DIR}'.", flush=True)
 
total = 0
 
for n, caminho in enumerate(videos, start=1):
    video = cv2.VideoCapture(str(caminho))
    if not video.isOpened():
        print(f"[{n}/{len(videos)}] Erro ao abrir {caminho.name}, pulando.", flush=True)
        continue
 
    fps = video.get(cv2.CAP_PROP_FPS) or 30
    passo = max(1, round(fps * INTERVAL))   # a cada quantos frames salvar
 
    print(f"\n[{n}/{len(videos)}] {caminho.name} (fps={fps:.1f}, salvando 1 a cada {passo} frames)", flush=True)
 
    i = 0
    salvos = 0
    while True:
        ok, frame = video.read()
        if not ok:
            break
 
        if i % passo == 0:
            if FLIP:
                frame = cv2.flip(frame, 1)
            if LARGURA:
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (LARGURA, round(h * LARGURA / w)))
 
            nome = f"{caminho.stem}_{salvos:04d}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, nome), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, QUALIDADE])
            salvos += 1
            total += 1
 
            segundo = i / fps
            print(f"  salvo {nome}  (em {segundo:6.1f} s)", flush=True)
        i += 1
 
    video.release()
    print(f"  -> {caminho.name}: {salvos} imagens salvas.", flush=True)
 
print(f"\nConcluído: {total} imagens em '{OUTPUT_DIR}'.", flush=True)
