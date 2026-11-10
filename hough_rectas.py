"""Prototipo educativo de Hough: cuatro bordes de una pieza simulada.

Instalación: python -m pip install -r requirements.txt
Ejecución:   python hough_rectas.py
No requiere fotografías, OpenCV ni una interfaz gráfica.
"""
from pathlib import Path
import argparse
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def crear_bordes(caso):
    """Mapa binario de contornos; no simula la adquisición de una foto real."""
    alto, ancho = 240, 320
    bordes = np.zeros((alto, ancho), dtype=bool)
    angulo = np.deg2rad(12)
    rotacion = np.array([[np.cos(angulo), -np.sin(angulo)],
                         [np.sin(angulo), np.cos(angulo)]])
    vertices = np.array([[-90, -50], [90, -50], [90, 50], [-90, 50]])
    vertices = vertices @ rotacion.T + [160, 120]
    for inicio, fin in zip(vertices, np.roll(vertices, -1, axis=0)):
        cantidad = int(np.ceil(np.max(np.abs(fin - inicio)))) + 1
        puntos = np.rint(np.linspace(inicio, fin, cantidad)).astype(int)
        bordes[puntos[:, 1], puntos[:, 0]] = True
    rng = np.random.default_rng(21)
    if caso == "interrumpido":
        y, x = np.nonzero(bordes)
        quitar = rng.random(len(x)) < 0.20
        bordes[y[quitar], x[quitar]] = False
    if caso in ("vacio", "solo_ruido"):
        bordes[:] = False
    if caso in ("ruido", "interrumpido", "solo_ruido"):
        # Son 400 posiciones elegidas al azar; pueden repetirse o tocar un borde.
        bordes[rng.integers(alto, size=400), rng.integers(ancho, size=400)] = True
    return bordes


def transformar_hough(bordes):
    """Vota en un acumulador 2D: rho cada 1 px y theta cada 1 grado."""
    alto, ancho = bordes.shape
    diagonal = int(np.ceil(np.hypot(alto - 1, ancho - 1)))
    rhos = np.arange(-diagonal, diagonal + 1)
    thetas = np.deg2rad(np.arange(180))
    acumulador = np.zeros((len(rhos), len(thetas)), dtype=np.uint32)
    y, x = np.nonzero(bordes)
    for columna, theta in enumerate(thetas):
        valores = np.rint(x * np.cos(theta) + y * np.sin(theta)).astype(int)
        # add.at contabiliza correctamente los votos repetidos en una misma celda.
        np.add.at(acumulador[:, columna], valores + diagonal, 1)
    return acumulador, rhos, thetas


def seleccionar_rectas(acumulador, rhos, thetas, umbral=35, max_rectas=8):
    """Extrae máximos y suprime vecinos para reducir detecciones duplicadas."""
    trabajo = acumulador.copy()
    rectas = []
    for _ in range(max_rectas):
        fila, columna = np.unravel_index(np.argmax(trabajo), trabajo.shape)
        votos = int(trabajo[fila, columna])
        if votos < umbral:
            break
        rho = float(rhos[fila])
        rectas.append({"rho_px": rho,
                       "theta_grados": float(np.rad2deg(thetas[columna])),
                       "votos": votos})
        # Al cruzar 0/180 grados, la representación equivalente cambia el signo de rho.
        for desplazamiento in range(-4, 5):
            angulo = columna + desplazamiento
            rho_vecino = -rho if angulo < 0 or angulo >= 180 else rho
            centro = int(np.argmin(np.abs(rhos - rho_vecino)))
            trabajo[max(0, centro - 8):centro + 9, angulo % 180] = 0
    return rectas


def dibujar_resultado(bordes, acumulador, rhos, rectas, destino, caso):
    alto, ancho = bordes.shape
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    axes[0].imshow(bordes, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("1. Mapa de bordes de entrada")
    axes[1].imshow(acumulador, cmap="magma", origin="lower", aspect="auto",
                   extent=[-0.5, 179.5, rhos[0] - .5, rhos[-1] + .5])
    axes[1].set_title("2. Acumulador de Hough")
    axes[1].set_xlabel("Theta (grados de la normal)")
    axes[1].set_ylabel("Rho (píxeles)")
    axes[2].imshow(bordes, cmap="gray", vmin=0, vmax=1)
    colores = plt.get_cmap("tab10")
    for i, recta in enumerate(rectas):
        rho, theta = recta["rho_px"], np.deg2rad(recta["theta_grados"])
        normal = np.array([np.cos(theta), np.sin(theta)])
        direccion = np.array([-np.sin(theta), np.cos(theta)])
        extremos = rho * normal + np.array([[-1000], [1000]]) * direccion
        axes[2].plot(extremos[:, 0], extremos[:, 1], color=colores(i % 10),
                     linewidth=1.5, label=f"R{i + 1}: {recta['votos']} votos")
    if rectas:
        axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_title(f"3. Rectas detectadas: {len(rectas)}")
    for ax in (axes[0], axes[2]):
        ax.set_xlim(-.5, ancho - .5)
        ax.set_ylim(alto - .5, -.5)
        ax.set_xlabel("x (píxeles)")
        ax.set_ylabel("y (píxeles)")
    fig.suptitle(f"Pieza rectangular simulada — caso: {caso}", fontsize=14)
    fig.savefig(destino, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caso", default="interrumpido",
                        choices=["limpio", "ruido", "interrumpido", "vacio", "solo_ruido"])
    parser.add_argument("--umbral", type=int, default=35)
    parser.add_argument("--salida", type=Path, default=Path("resultados"))
    args = parser.parse_args()
    if args.umbral < 1:
        parser.error("El umbral debe ser un entero positivo.")
    args.salida.mkdir(parents=True, exist_ok=True)
    bordes = crear_bordes(args.caso)
    acumulador, rhos, thetas = transformar_hough(bordes)
    rectas = seleccionar_rectas(acumulador, rhos, thetas, args.umbral)
    nombre = f"{args.caso}_umbral{args.umbral}"
    dibujar_resultado(bordes, acumulador, rhos, rectas,
                     args.salida / f"{nombre}.png", args.caso)
    informe = {"caso": args.caso, "semilla": 21, "umbral_votos": args.umbral,
               "paso_rho_px": 1, "paso_theta_grados": 1,
               "pixeles_borde": int(bordes.sum()), "rectas": rectas,
               "nota": "Theta corresponde a la normal. Las rectas se dibujan completas, no como segmentos."}
    (args.salida / f"{nombre}.json").write_text(
        json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.salida / f"{nombre}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rho_px", "theta_grados", "votos"])
        writer.writeheader()
        writer.writerows(rectas)
    print(f"Caso: {args.caso} | Umbral: {args.umbral} | Rectas: {len(rectas)}")
    for i, recta in enumerate(rectas, 1):
        print(f"R{i}: rho={recta['rho_px']:.0f} px, theta={recta['theta_grados']:.0f} grados, votos={recta['votos']}")
    print(f"Resultados guardados en: {args.salida.resolve()}")


if __name__ == "__main__":
    main()
