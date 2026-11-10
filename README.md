# TP4 - Punto 2: Hough para rectas

Prototipo original en Python. Implementa directamente el acumulador de Hough
con NumPy; Matplotlib genera las figuras. No llama a funciones Hough de OpenCV.

## Ejecución en Windows

1. Extraer el ZIP y abrir una terminal en la carpeta TP4_Punto2.
2. Instalar dependencias:

   python -m pip install -r requirements.txt

3. Ejecutar:

   python hough_rectas.py

Si Windows reconoce `py` en lugar de `python`, usar `py` en ambos comandos.
Las imágenes se guardan en la subcarpeta resultados; no se abre una ventana.
El programa también imprime los parámetros y guarda archivos CSV y JSON.

## Caso propuesto

Mapa binario de 320 x 240 píxeles con los contornos de una pieza rectangular
de 180 x 100 píxeles, centrada en (160,120), girada 12 grados en coordenadas
de imagen. La entrada es un mapa de bordes sintético, no una fotografía.
En estas coordenadas, x crece a la derecha e y hacia abajo.

Por defecto, elimina cada píxel del contorno con probabilidad 0,20 y añade
400 posiciones de ruido al azar (pueden repetirse). La semilla es 21.
Estos son ejemplos reproducibles, no una evaluación estadística de robustez.

## Algoritmo

rho = x*cos(theta) + y*sin(theta)

Se vota con 180 ángulos: 0 a 179 grados, paso 1 grado. Rho se redondea
al píxel más próximo y admite valores negativos. Se buscan máximos con
al menos 35 votos. Tras cada detección se suprime una vecindad de +/-8
píxeles y +/-4 grados. Al cruzar 0/180 grados se invierte el signo de rho.
Se permite un máximo de 8 rectas; llegar a ese límite no prueba que no
existan otros máximos. Las rectas se muestran completas, no como segmentos.

## Comandos para comparar

python hough_rectas.py --caso limpio
python hough_rectas.py --caso ruido
python hough_rectas.py --caso interrumpido
python hough_rectas.py --caso vacio
python hough_rectas.py --caso solo_ruido
python hough_rectas.py --caso interrumpido --umbral 15
python hough_rectas.py --caso interrumpido --umbral 55

## Resultados comprobados

| Caso | Umbral | Rectas reportadas |
| --- | ---: | ---: |
| Contorno completo | 35 | 4 |
| Contorno completo con ruido | 35 | 4 |
| Contorno interrumpido con ruido | 35 | 4 |
| Imagen vacía | 35 | 0 |
| Solamente ruido | 35 | 0 |
| Contorno interrumpido con ruido | 15 | 8 (incluye repeticiones) |
| Contorno interrumpido con ruido | 55 | 2 |

Salida predeterminada:

| Recta | Rho (px) | Theta de la normal (grados) | Votos |
| --- | ---: | ---: | ---: |
| R1 | 134 | 102 | 132 |
| R2 | 34 | 102 | 131 |
| R3 | 270 | 11 | 51 |
| R4 | 91 | 12 | 50 |

La diferencia entre 11 y 12 grados refleja discretización y rasterizado.
Los cuatro bordes esperados se localizaron con diferencia angular máxima
de 1 grado en estas pruebas. Theta es el ángulo de la normal, no la
inclinación directa del borde.

Ver resultados/interrumpido_umbral35.png para la figura principal.

## Funciones para comprender el código

- crear_bordes: genera la entrada y sus variantes.
- transformar_hough: calcula el acumulador.
- seleccionar_rectas: busca máximos y suprime vecinos.
- dibujar_resultado: muestra entrada, acumulador y rectas.
- main: procesa opciones y guarda resultados.

## Alcance

Permite estudiar la detección de líneas en un caso sencillo. No reconoce
semánticamente un motor, no localiza el aro circular, no transforma píxeles
a coordenadas del robot y no calcula ni ejecuta sus movimientos. Para usar
fotografías se necesitarían adquisición, preprocesamiento y detección de
bordes. Para controlar un robot se requerirían calibración y validación.

## Entorno de la ejecución de referencia

Python 3.12.14, NumPy 2.3.5 y Matplotlib 3.10.8.
El código es compatible con Python 3.10 o posterior; la versión instalada
de cada dependencia debe ser compatible con el Python utilizado.

## Consulta teórica

OpenCV, Hough Line Transform:
https://docs.opencv.org/4.x/d6/d10/tutorial_py_houghlines.html

La consulta es conceptual; este programa implementa su propio acumulador.

## Punto 3: Hough para circunferencias

El prototipo detecta el centro de una circunferencia de radio conocido en una imagen simulada con ruido. Utiliza las mismas dependencias que el prototipo de rectas.

### Ejecución

```powershell
python hough_circunferencias.py
```

En el ejemplo con radio de 30 píxeles se obtiene el centro (100, 80), con 200 votos y un error de 0 píxeles.

### Prueba con un radio incorrecto

```powershell
python hough_circunferencias.py --radio 25
```

El máximo obtiene 31 votos y no alcanza el umbral de aceptación de 60.

Las imágenes y los registros se guardan en la carpeta `resultados_circunferencias`.
