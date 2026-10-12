# Predicción de baja de clientes — Primera competencia

Florencia Rodriguez · DMEyF 2026

La entrega final es **FlorFinalV88k12000**: veinte listas de **12.000 clientes de agosto de 2021**, generadas con veinte ensambles V88. Cada lista promedia diez modelos de un pool de doce LightGBM. Este paquete reconstruye las variables desde el dataset y reproduce exactamente los veinte CSV enviados. Incluye el reentrenamiento y la optimización que originó los parámetros.

| Referencia | Valor |
|---|---|
| Archivos enviados | Los veinte `FlorV95_V88_dd5114891f_ensambleNN_k12000.csv` de `03_prediccion_final/` |
| Formato | Una columna de IDs enteros, sin encabezado ni índice |
| Filas | 12.000 por archivo |
| Submission ID | 1567-1791770740689465 |
| Media pública / desvío entre archivos | 105,2659 M / 0,7696 M |
| SHA-256 de cada lista | `configuracion.json`, `VERIFICACION.json` y `SHA256SUMS.json` |

El bot no informa qué lista individual obtiene la mayor ganancia. La entrega corresponde al paquete completo; no se identifica una lista individual como ganadora.

## 1. Instalación

Entorno verificado: **Python 3.11**, versiones fijadas en `requirements.txt`. Se recomienda 16 GB de RAM como mínimo y 8 GB libres en disco para matrices y salidas. El dataset no se incluye en el repositorio.

Desde esta carpeta:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

En Windows, activar con `.venv\Scripts\activate`. LightGBM requiere OpenMP; si falta en macOS, instalar `libomp` con Homebrew. En Linux se necesita la biblioteca `libgomp`.

## 2. Reproducir la predicción exacta

Usar el CSV original con los meses **202103–202108** y los nombres originales de sus columnas. Acepta `clase_ternaria` ya incluida. Si no está, reconstruye las etiquetas de marzo–junio por presencia de clientes en los dos meses siguientes; julio/agosto quedan sin etiqueta.

```bash
python reproducir.py --dataset "/ruta/competencia_01.csv" --salida salida
```

El comando:

1. Reconstruye las 762 variables, su orden y sus valores `float32`.
2. Verifica SHA-256 de las seis matrices mensuales, IDs y etiquetas contra la corrida original.
3. Carga los doce modelos guardados y recalcula las probabilidades de los 164.647 clientes de agosto.
4. Comprueba las probabilidades contra la referencia, con tolerancia absoluta `1e-12`.
5. Para cada uno de los veinte subconjuntos fijados, promedia sus diez componentes, ordena por score descendente y por ID ascendente ante empate, y toma los primeros 12.000.
6. Compara los veinte CSV, byte por byte, contra los archivos enviados.

Las salidas son los veinte CSV de `salida/predicciones_finales/`, `salida/VERIFICACION.json` y `salida/datos/VERIFICACION_DATOS.json`. Si alguna lista no coincide, el programa termina con error. La referencia de probabilidades sirve para auditar: **la predicción se calcula desde los datos y los modelos**, no se copia de esa referencia.

La inferencia no necesita credenciales, servicios externos ni volver a optimizar. El tiempo de la reproducción completa está registrado en `VERIFICACION.json`. Las matrices también se verificaron desde el CSV crudo sin target.

## 3. Variables y modelo

- **Target de entrenamiento:** BAJA+1 o BAJA+2 = 1; CONTINUA = 0. La ganancia de evaluación recompensa únicamente BAJA+2: +1.072.500 por acierto y −27.500 por contacto incorrecto.
- **Entrenamiento final:** marzo, abril, mayo y junio de 2021, sin undersampling de clases ni pesos de recencia. El bagging de LightGBM toma 70% de filas por iteración; no es un muestreo específico de CONTINUA.
- **Historia:** lags 1 y 2 y deltas contra esos lags de las variables originales, excepto ID y mes. Los lags generales buscan el mes calendario exacto. Se usa julio para construir la historia de agosto.
- **Calidad:** `ccajas_depositos` pasa a NA en mayo antes de reconstruir su historia. Para reproducir exactamente el entrenamiento, esa familia conserva lags por observaciones previas del cliente; es la única excepción a los lags por calendario. Los faltantes se preservan.
- **Rankings:** después de calcular historia y convertir a `float32`, 365 campos monetarios actuales e históricos se reemplazan por rankings dentro de cada mes. Los positivos quedan en (0, 1], los negativos en [−1, 0), cero conserva cero y NA conserva NA. Los empates usan rango medio. La población de referencia es el conjunto completo de clientes observado en cada mes, sin usar etiquetas.
- **Orden de columnas:** fijado en `configuracion.json`. Se conservan ID, mes, personales y fechas.
- **Parámetros fijos:** learning rate 0,005; max_bin 31; min_data_in_leaf 0; feature_fraction 0,50; feature_fraction_bynode 0,20; bagging_fraction 0,70; bagging_freq 1. `deterministic` y `force_col_wise` están activados.
- **Ensamble final:** veinte subconjuntos de diez componentes del pool de doce, con igual peso dentro de cada subconjunto. Los índices y su orden están fijados en `configuracion.json`; no se vuelven a sortear al ejecutar.

## 4. De dónde salen los hiperparámetros

`01_optimizacion/estudio.sqlite` es la base Optuna original: **12 trials completos y uno interrumpido**. Se incluyen las doce configuraciones, y el paquete final utiliza todos, en subconjuntos de diez.

- Objetivo: media de AUC para ambas bajas en **marzo→mayo** y **marzo–abril→junio**. Se registra además AUC BAJA+2 y ganancia a 10.500 contactos.
- Sampler: TPE, semilla 882026, cuatro trials iniciales.
- Espacio: tope de árboles {2.000, 4.000, 8.000}; hojas {8, 16, 24, 32, 48, 64}; mínimo Hessian de 1 a 100 en escala logarítmica.
- Early stopping: 150 árboles. El trial 0 evaluó AUC nativa en cada árbol. Tras una interrupción se evaluó AUC exacta cada diez árboles, conservando su último valor entre evaluaciones; el AUC final se recalculó sobre la predicción elegida.
- La configuración del trial interrumpido se volvió a encolar. Hubo un reinicio del sampler; por eso repetir una búsqueda nueva con la misma semilla no garantiza la misma secuencia de candidatos.
- Árboles finales por modelo: promedio redondeado de las mejores iteraciones de los dos folds. Se conservaron los doce trials completos: no hubo selección de trials por resultados públicos.

Comprobar el origen sin entrenar:

```bash
python 01_optimizacion/optimizar.py verificar
```

Reevaluar las doce configuraciones originales en sus folds, respetando la evaluación especial del trial 0:

```bash
python 01_optimizacion/optimizar.py repetir --datos salida/datos --salida reevaluacion
```

Ejecutar una búsqueda nueva con el mismo espacio y objetivo:

```bash
python 01_optimizacion/optimizar.py nueva --datos salida/datos --salida nueva_busqueda --trials 12
```

La búsqueda puede tardar varias horas. No es necesaria para reproducir los veinte CSV finales y no sobrescribe parámetros, estudio ni modelos de la entrega. Los folds fueron usados para desarrollo; no son un test independiente.

## 5. Reentrenar los modelos finales

Para entrenar nuevamente los doce componentes con las configuraciones y semillas fijadas:

```bash
python entrenar.py --datos salida/datos --salida reentrenamiento
```

También puede iniciarse desde el CSV con `--dataset "/ruta/competencia_01.csv"` en lugar de `--datos`. La corrida usa todas las filas etiquetadas de marzo–junio. Guarda doce modelos nuevos, mide sus diferencias contra las probabilidades históricas y registra el tiempo de cada uno.

Generar los veinte CSV usando esos nuevos modelos:

```bash
python reproducir.py --dataset "/ruta/competencia_01.csv" --modelos reentrenamiento --salida salida_reentrenada
```

La reproducción exacta verificada se realiza con los modelos incluidos. El reentrenamiento depende también de la versión nativa de LightGBM, arquitectura y aritmética del entorno; por eso tiene comprobaciones explícitas y no se promete igualdad en cualquier plataforma. `entorno.json` registra las versiones de la corrida verificada. Como comprobación adicional, se reentrenó el componente 08 en 10 minutos y 42 segundos: sus probabilidades coincidieron exactamente. No se repitieron los doce entrenamientos ni toda la búsqueda durante la preparación de esta carpeta.

## 6. Construcción y corte del paquete final

Se construyeron veinte combinaciones de diez modelos del pool de doce. Se entregaron las veinte listas, sin seleccionar una semilla por su resultado público. Las combinaciones, pesos, semillas y hashes están fijados en `configuracion.json`.

El corte de 12.000 se fijó tras comparar nueve cortes del mismo ranking, de 9.500 a 13.500. Entre 11.000 y 13.000 la media pública fue aproximadamente plana; se eligió el centro de esa zona en lugar del máximo puntual. La elección de familia se apoyó en experimentos temporales y resultados públicos de desarrollo. El público no garantiza el resultado privado.

El paquete final coincide con los veinte archivos del experimento V95, reenviados bajo el nombre **FlorFinalV88k12000**. Su media pública es del conjunto de listas; no se atribuye a ningún ensamble individual.

## Archivos

```text
README.md
requirements.txt
configuracion.json
entorno.json
reproducir.py
entrenar.py
src/datos.py
01_optimizacion/
    optimizar.py
    estudio.sqlite
    configuraciones.json
02_modelos/
    modelo_XX.txt.gz
03_prediccion_final/
    FlorV95_V88_dd5114891f_ensamble01_k12000.csv
    ...
    FlorV95_V88_dd5114891f_ensamble20_k12000.csv
    predicciones_referencia.npz
VERIFICACION.json
SHA256SUMS.json
```

No hace falta ningún archivo de otros directorios del proyecto. Las salidas y el dataset quedan excluidos mediante `.gitignore`.
