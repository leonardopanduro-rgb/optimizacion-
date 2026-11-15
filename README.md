# Optimización con SVM — Dry Bean Dataset

Proyecto de clasificación de siete variedades de frijol seco mediante máquinas de vectores soporte (SVM). Incluye limpieza de datos, validación cruzada, una implementación de SMO, comparación de kernels, estrategias multiclase OvR/OvO, análisis de hiperparámetros y medición del costo computacional.

## Integrantes

Leonardo Panduro · Diego Godoy · Nayara Sernaqué · Jose Cruz

**Profesor:** Víctor Torres Castillo

## Descripción del problema

Cada observación contiene 16 medidas geométricas de un frijol. El objetivo es identificar a cuál de las siete variedades pertenece.

La SVM construye fronteras de separación entre las clases. El parámetro `C` controla la penalización de los errores y `gamma` determina el alcance de la influencia de cada observación en el kernel RBF.

Los datos se dividen en:

- **80 % para entrenamiento:** se usa para ajustar los modelos y seleccionar `C` y `gamma` mediante validación cruzada.
- **20 % para prueba:** permanece reservado hasta la evaluación final.

## Metodología

| Etapa | Procedimiento |
|---|---|
| Duplicados | Se eliminan 68 filas exactas antes del split: 13 611 → 13 543. |
| Faltantes | No se aplica imputación porque las 16 variables no contienen valores faltantes. |
| Validación | 5 pliegues estratificados, semilla 42. |
| Búsqueda | Potencias de 2 en dos etapas: malla amplia y refinamiento local. |
| Caso binario | La pareja se elige por mayor confusión OOF en entrenamiento. Resultó DERMASON–SIRA. |
| Evaluación | Las métricas y tablas se generan directamente durante la ejecución. |
| SMO propio | Kernels lineal/polinomial/RBF, pareja de máxima violación y parada `m-M`. |
| Multiclase | OvR y OvO propios en un subconjunto; comparación completa con `SVC`. |
| E4 | Incluye exactitud, F1 macro y cantidad de vectores soporte. |
| E5 | Tres repeticiones, mediana, IQR, ajuste `T(m)=c·m^theta` y `R²`. |

## Resultados principales de la corrida incluida (Hito II)

- Mejor RBF multiclase por CV: `C=64`, `gamma=0.015625`, F1 macro CV `0.9448`.
- La selección OOF encontró 348 confusiones entre DERMASON y SIRA; la misma pareja salió elegida en las diez particiones adicionales.
- Caso binario: RBF obtuvo F1 macro CV `0.9285` y F1 macro test `0.9163`. En diez particiones nuevas RBF supera al lineal en 9 de 10 (+0.0038 de media).
- Multiclase a configuración común (`C=64`, `gamma=2^-6`): OvR F1 macro test `0.9337`; OvO `0.9350`. La diferencia (+0.0012) tiene IC 95 % bootstrap `[-0.0030, +0.0056]` y cambia de signo entre particiones (media de diez semillas: `-0.0005`). **OvR y OvO no se distinguen en desempeño predictivo**; lo que cambia es el costo (OvO entrena ~2 veces más rápido con 8 000 filas y predice algo más lento).
- SMO propio (E1, 400 filas): 244 iteraciones, residuo KKT global `m-M=0.000904`, |D_propio − D_LIBSVM| = `9.0e-5`, máxima violación KKT puntual `5.5e-4`, concordancia de predicciones `1.0000`, 95 vectores soporte en ambos.
- El buen desempeño en (C, gamma) ocupa una meseta ancha (14 de 33 configuraciones con F1 ≥ 0.940), no un punto.

## Qué cambió en el Hito II

- `SMOBinary` registra el **residuo KKT global** `m-M` y el objetivo dual `D(α)` en cada iteración (`gap_history_`, `dual_history_`); la antigua diferencia del par actualizado se conserva en `pair_gap_history_`. Nuevo método `pointwise_kkt_violation()`.
- `SMOOneVsOne` desempata de forma **lexicográfica** (votos → margen acumulado → orden de clase) en lugar de `votos + 1e-9·margen`.
- E3 añade un **bootstrap pareado** (2 000 réplicas) de la diferencia OvO − OvR y la matriz de confusión de OvR.
- E5 reporta cuartiles de predicción y del barrido por número de clases.
- `results/environment.json` registra SHA, versiones exactas, CPU, núcleos, RAM y SO de cada corrida.
- La caché de la búsqueda multiclase se valida con una huella SHA-256 de datos + rejilla + semilla + versiones (`results/cache_meta.json`); si algo cambia, se recalcula sola.
- `run_seeds.py` repite todo el protocolo (partición, búsqueda, selección del par, E2, E3 con bootstrap) para las semillas 1–10.
- Pruebas: 18 (incluyen los ejemplos a mano del informe: margen duro, dos puntos, camino de la solución en C, curvatura nula, desempate OvO).

## Cómo abrirlo

El archivo principal es [`OPTI.ipynb`](OPTI.ipynb). Ya está ejecutado y contiene tablas, gráficas e interpretaciones.

```powershell
git clone https://github.com/leonardopanduro-rgb/optimizacion-.git
cd optimizacion-
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyter lab
```

Luego abre `OPTI.ipynb`. Para leerlo no necesitas recalcular nada.

## Cómo verificar y regenerar

```powershell
.\.venv\Scripts\python.exe -m pytest -q        # 18 passed
.\.venv\Scripts\python.exe run_experiments.py  # E1-E5 con semilla 42, ~2-4 min
.\.venv\Scripts\python.exe run_seeds.py        # remuestreo de 10 semillas, ~15-30 min
```

La búsqueda multiclase se reutiliza solo si `results/cache_meta.json` coincide con los datos, la rejilla, la semilla y las versiones instaladas; en caso contrario se recalcula automáticamente. Las métricas de SVC/LIBSVM son deterministas; los tiempos de E5 dependen de la máquina (ver `results/environment.json`).

## Estructura

```text
OPTI.ipynb                    análisis completo y ejecutado
data/Dry_Bean_Dataset.xlsx   copia original de los datos
src/smo.py                   SMO propio y wrappers OvR/OvO
src/experiments.py           protocolo E1–E5 y generación de resultados
run_experiments.py           entrada para repetir todos los experimentos
run_seeds.py                 protocolo de remuestreo con 10 semillas (Hito II)
tests/test_smo.py            comparación con LIBSVM y ejemplos a mano del informe
figures/                     gráficas generadas por los experimentos
results/                     tablas CSV, metrics.json, environment.json
docs/informe/                informe del Hito II (Markdown y PDF)
docs/resultados_recibidos/   copia de los CSV que cita el informe
docs/figuras_hito2/          figuras del informe (docs/make_figures_hito2.py)
verificacion_hito2/          reproducción en segunda máquina y registro de pruebas
RESULTADOS.md                relación de resultados, tablas y figuras
```

## Contenido del análisis

1. Tabla de 13 611 filas originales, 68 duplicados, 13 543 filas analíticas y cero faltantes.
2. Distribución por clase antes/después de la limpieza.
3. Protocolo 80/20, semilla 42, CV estratificada de 5 pliegues y test usado una vez.
4. E1: objetivo dual, brecha `m-M`, residuo de igualdad, diferencia máxima de scores y concordancia.
5. Matriz OOF que justifica DERMASON–SIRA.
6. E2: comparación lineal/polinomial/RBF con CV, test y `nSV`.
7. E3: tabla OvR/OvO, reporte por clase y matriz de confusión.
8. E4: mapas de calor de F1, exactitud y vectores soporte.
9. E5: tiempos con mediana/IQR, `theta` y `R²`.
10. Limitaciones y conclusiones prudentes.

La relación exacta entre experimentos y archivos está en [`RESULTADOS.md`](RESULTADOS.md).

## Datos y atribución

El archivo proviene de [UCI Dry Bean](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset), DOI [10.24432/C50S4B](https://doi.org/10.24432/C50S4B), y también se distribuye en [Kaggle](https://www.kaggle.com/datasets/muratkokludataset/dry-bean-dataset). Cita: Koklu, M. y Ozkan, I. A. (2020), *Multiclass classification of dry beans using computer vision and machine learning techniques*.

## Limitaciones

- La implementación NumPy de SMO se valida en subconjuntos por su costo de memoria `O(n²)`; los experimentos completos usan `SVC`/LIBSVM.
- Los tiempos dependen del equipo y no demuestran por sí solos complejidad asintótica.
- El dataset contiene características numéricas extraídas de imágenes, no las fotografías originales.
- El test de la semilla 42 se consultó en el Hito I; por eso el Hito II evalúa la estabilidad de las conclusiones con diez particiones nuevas (semillas 1–10). Esas particiones reutilizan el mismo dataset y no son independientes entre sí.
