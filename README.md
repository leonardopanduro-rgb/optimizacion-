# Optimización con SVM — Dry Bean Dataset

Proyecto de clasificación de siete variedades de frijol seco mediante máquinas de vectores soporte (SVM). Incluye limpieza de datos, validación cruzada, una implementación de SMO, comparación de kernels, estrategias multiclase OvR/OvO, análisis de hiperparámetros y medición del costo computacional.

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

## Resultados principales de la corrida incluida

- Mejor RBF multiclase por CV: `C=64`, `gamma=0.015625`, F1 macro CV `0.9448`.
- La selección OOF encontró 348 confusiones entre DERMASON y SIRA.
- Caso binario: RBF obtuvo F1 macro CV `0.9285` y F1 macro test `0.9163`.
- Multiclase: OvR obtuvo F1 macro test `0.9337`; OvO, `0.9350`.
- SMO propio: concordancia con LIBSVM `1.0000`, brecha KKT `m-M=0.000926` y residuo de igualdad cercano a cero.

La ventaja de OvO es pequeña: el resultado correcto es “fue ligeramente mejor en esta partición”, no “siempre es superior”.

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
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe run_experiments.py
```

La ejecución completa puede tardar varios minutos. Los resultados multiclase ya calculados se reutilizan si existe `results/e4_cv_multiclase_rbf.csv`. Para forzar una búsqueda totalmente nueva, elimina solo ese CSV y vuelve a ejecutar.

## Estructura

```text
OPTI.ipynb                    análisis completo y ejecutado
data/Dry_Bean_Dataset.xlsx   copia original de los datos
src/smo.py                   SMO propio y wrappers OvR/OvO
src/experiments.py           protocolo E1–E5 y generación de resultados
run_experiments.py           entrada para repetir todos los experimentos
tests/test_smo.py            comparación automática con LIBSVM
figures/                     gráficas generadas por los experimentos
results/                     tablas CSV y metrics.json
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
- El test representa una sola partición externa; no debe reutilizarse para volver a ajustar decisiones.
