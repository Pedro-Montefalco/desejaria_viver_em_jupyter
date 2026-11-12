#!/usr/bin/env python3
"""Recorta as mãos das fotos usando o detector YOLO já treinado.

Para cada foto da pasta de entrada, detecta as mãos, ordena da esquerda para a direita
e salva um recorte quadrado de cada uma (com margem), já redimensionado para o tamanho
do classificador. Nas fotos com duas mãos os arquivos saem como <nome>_esq.jpg e
<nome>_dir.jpg. A estrutura de subpastas é mantida.

Uso:
  python recortar_maos.py /home/dataset_luva/por_gesto_3
  python recortar_maos.py /home/dataset_luva/por_gesto_3 --saida /home/dataset_luva_crop/por_gesto_3

Dependências: pip install ultralytics opencv-python numpy
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DIR_BASE = Path(__file__).resolve().parent.parent


def ler_imagem(caminho):
    # imdecode evita problemas do cv2.imread com caracteres especiais no caminho
    dados = np.fromfile(str(caminho), dtype=np.uint8)
    return cv2.imdecode(dados, cv2.IMREAD_COLOR)


def salvar_imagem(caminho, imagem):
    ok, buf = cv2.imencode(caminho.suffix or ".jpg", imagem, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError(f"Falha ao codificar {caminho}")
    buf.tofile(str(caminho))


def recorte_quadrado(imagem, caixa, padding, tamanho):
    """Recorta um quadrado centrado na caixa, com margem. Se passar da borda, completa com cinza."""
    x1, y1, x2, y2 = caixa
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    lado = max(int(round(max(x2 - x1, y2 - y1) * (1 + 2 * padding))), 2)
    x0, y0 = int(round(cx - lado / 2)), int(round(cy - lado / 2))

    tela = np.full((lado, lado, 3), 127, dtype=np.uint8)
    h, w = imagem.shape[:2]
    sx1, sy1 = max(x0, 0), max(y0, 0)
    sx2, sy2 = min(x0 + lado, w), min(y0 + lado, h)
    if sx2 > sx1 and sy2 > sy1:
        tela[sy1 - y0:sy2 - y0, sx1 - x0:sx2 - x0] = imagem[sy1:sy2, sx1:sx2]
    return cv2.resize(tela, (tamanho, tamanho), interpolation=cv2.INTER_AREA)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("entrada", type=Path, help="pasta com as fotos (busca em subpastas também)")
    p.add_argument("--saida", type=Path, default=None, help="padrão: <entrada>_crop, ao lado da pasta de entrada")
    p.add_argument(
        "--pesos", type=Path,
        default=DIR_BASE / "runs_detector/yolo26n_320/weights/best.pt",
        help="pesos do detector de mãos treinado",
    )
    p.add_argument("--maos", type=int, default=2, help="máximo de mãos por foto (as de maior confiança)")
    p.add_argument("--padding", type=float, default=0.15, help="margem em fração do lado da mão, de cada lado")
    p.add_argument("--tamanho", type=int, default=224, help="lado do recorte final em pixels")
    p.add_argument("--conf", type=float, default=0.25, help="confiança mínima da detecção")
    p.add_argument("--imgsz", type=int, default=320, help="resolução usada no treino do detector")
    p.add_argument("--device", default=None, help="ex.: cuda ou cpu (padrão: automático)")
    args = p.parse_args()

    entrada = args.entrada.resolve()
    saida = (args.saida or entrada.with_name(entrada.name + "_crop")).resolve()
    if not entrada.is_dir():
        raise SystemExit(f"Pasta de entrada não encontrada: {entrada}")
    if not args.pesos.is_file():
        raise SystemExit(f"Pesos do detector não encontrados: {args.pesos} (use --pesos)")

    fotos = sorted(f for f in entrada.rglob("*") if f.suffix.lower() in EXTENSOES and saida not in f.parents)
    if not fotos:
        raise SystemExit(f"Nenhuma imagem encontrada em {entrada}")

    modelo = YOLO(str(args.pesos))
    saida.mkdir(parents=True, exist_ok=True)
    problemas = []
    total_recortes = 0

    for i, foto in enumerate(fotos, 1):
        imagem = ler_imagem(foto)
        if imagem is None:
            problemas.append(f"{foto.relative_to(entrada)}\tilegível")
            continue

        res = modelo.predict(
            imagem, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False
        )[0]
        caixas = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        if len(caixas) == 0:
            problemas.append(f"{foto.relative_to(entrada)}\tnenhuma mão detectada")
            continue

        # fica com as de maior confiança e ordena da esquerda para a direita
        manter = np.argsort(-confs)[: args.maos]
        caixas = sorted(caixas[manter], key=lambda c: (c[0] + c[2]) / 2)
        if len(caixas) == 1 and args.maos > 1:
            problemas.append(f"{foto.relative_to(entrada)}\tsó 1 mão detectada")

        nomes = ["esq", "dir"] if len(caixas) == 2 else ["mao"] if len(caixas) == 1 else [f"mao{k}" for k in range(len(caixas))]
        destino = saida / foto.relative_to(entrada).parent
        destino.mkdir(parents=True, exist_ok=True)
        for nome, caixa in zip(nomes, caixas):
            recorte = recorte_quadrado(imagem, caixa, args.padding, args.tamanho)
            salvar_imagem(destino / f"{foto.stem}_{nome}.jpg", recorte)
            total_recortes += 1

        if i % 100 == 0:
            print(f"{i}/{len(fotos)} fotos processadas", flush=True)

    if problemas:
        (saida / "revisar.txt").write_text("\n".join(problemas) + "\n", encoding="utf-8")
    print(f"{len(fotos)} fotos -> {total_recortes} recortes em {saida}")
    print(f"{len(problemas)} fotos para revisar" + (f" (lista em {saida / 'revisar.txt'})" if problemas else ""))


if __name__ == "__main__":
    main()
