"""Ejemplo sencillo de Hough para una circunferencia de radio conocido.
Ejecutar: python hough_circunferencias.py
"""
from pathlib import Path
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def crear_imagen():
    imagen = np.zeros((160, 200), dtype=bool)
    angulos = np.deg2rad(np.arange(360))
    # El centro real solo se usa para generar el ejemplo y comprobar el error.
    x = np.rint(100 + 30 * np.cos(angulos)).astype(int)
    y = np.rint(80 + 30 * np.sin(angulos)).astype(int)
    imagen[y, x] = True
    rng = np.random.default_rng(7)
    imagen[rng.integers(160, size=100), rng.integers(200, size=100)] = True
    return imagen


def detectar_centro(imagen, radio):
    alto, ancho = imagen.shape
    acumulador = np.zeros((alto, ancho), dtype=np.uint32)
    angulos = np.deg2rad(np.arange(360))
    y_borde, x_borde = np.nonzero(imagen)
    for x, y in zip(x_borde, y_borde):
        a = np.rint(x - radio * np.cos(angulos)).astype(int)
        b = np.rint(y - radio * np.sin(angulos)).astype(int)
        validos = (a >= 0) & (a < ancho) & (b >= 0) & (b < alto)
        # Cada píxel aporta como máximo un voto a cada centro candidato.
        centros = np.unique(np.column_stack((a[validos], b[validos])), axis=0)
        np.add.at(acumulador, (centros[:, 1], centros[:, 0]), 1)
    b, a = np.unravel_index(np.argmax(acumulador), acumulador.shape)
    votos = int(acumulador[b, a])
    # Un máximo con pocos votos no se acepta como detección.
    centro = (int(a), int(b)) if votos >= 60 else None
    return centro, votos, acumulador


def guardar_imagen(imagen, acumulador, centro, radio, archivo):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(imagen, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Entrada: contorno y ruido")
    axes[1].imshow(acumulador, cmap="magma")
    axes[1].set_title("Votos para posibles centros")
    axes[2].imshow(imagen, cmap="gray", vmin=0, vmax=1)
    if centro is not None:
        axes[2].add_patch(plt.Circle(centro, radio, fill=False, color="lime", linewidth=1.5))
        axes[2].plot(*centro, marker="+", color="red", markersize=12)
        axes[2].set_title(f"Centro detectado: {centro}")
    else:
        axes[2].set_title("Sin detección aceptada")
    for ax in axes:
        ax.set_xlabel("x (píxeles)")
        ax.set_ylabel("y (píxeles)")
    fig.savefig(archivo, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radio", type=int, default=30)
    args = parser.parse_args()
    if args.radio <= 0:
        parser.error("El radio debe ser positivo.")
    imagen = crear_imagen()
    centro, votos, acumulador = detectar_centro(imagen, args.radio)
    salida = Path("resultados_circunferencias")
    salida.mkdir(exist_ok=True)
    guardar_imagen(imagen, acumulador, centro, args.radio,
                   salida / f"circunferencia_radio{args.radio}.png")
    datos = {"radio_busqueda_px": args.radio, "centro_detectado": centro,
             "votos_maximos": votos, "umbral": 60, "centro_real": [100, 80],
             "radio_real_px": 30, "semilla_ruido": 7}
    print(f"Radio de búsqueda: {args.radio} píxeles")
    if centro is None:
        print(f"Sin detección: máximo de {votos} votos, inferior al umbral de 60.")
    else:
        error = float(np.hypot(centro[0] - 100, centro[1] - 80))
        datos["error_centro_px"] = error
        print(f"Centro detectado: x={centro[0]}, y={centro[1]}")
        print(f"Votos: {votos}")
        print(f"Error del centro: {error:.1f} píxeles")
    (salida / f"circunferencia_radio{args.radio}.json").write_text(
        json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Resultados guardados en: {salida.resolve()}")


if __name__ == "__main__":
    main()
