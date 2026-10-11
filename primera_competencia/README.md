# Predicción de baja de clientes — Primera competencia

Florencia Rodriguez · DMEyF 2026

La predicción seleccionada es **V88, ensamble 07**: promedio de diez modelos LightGBM, con **10.500 clientes de agosto de 2021**. Este paquete reconstruye las variables a partir del dataset de la competencia y genera la misma lista usando los modelos guardados. También incluye el reentrenamiento y la optimización que originó los parámetros.

| Referencia | Valor |
|---|---|
| Archivo seleccionado | `03_prediccion_final/prediccion_final.csv` |
| Formato | Una columna de IDs enteros, sin encabezado ni índice |
| Filas | 10.500 |
| SHA-256 | `c8758214110f92cf200ffda43e2960def98a5924d5ddde88b75a86d3972bcef3` |

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
3. Carga los diez modelos guardados y recalcula las probabilidades de los 164.647 clientes de agosto.
4. Comprueba las probabilidades contra la referencia, con tolerancia absoluta `1e-12`.
5. Promedia los diez componentes, ordena por score descendente y por ID ascendente ante empate, y toma los primeros 10.500.
6. Compara el CSV completo, byte por byte, contra el archivo seleccionado.

Las salidas son `salida/prediccion_final.csv`, `salida/VERIFICACION.json` y `salida/datos/VERIFICACION_DATOS.json`. Si la lista no coincide, el programa termina con error. La referencia de probabilidades sirve para auditar: **la predicción se calcula desde los datos y los modelos**, no se copia de esa referencia.

La inferencia no necesita credenciales, servicios externos ni volver a optimizar. La prueba completa tardó aproximadamente **4 minutos y 15 segundos** en el entorno registrado. Las matrices también se verificaron desde el CSV crudo sin target.

## 3. Variables y modelo

- **Target de entrenamiento:** BAJA+1 o BAJA+2 = 1; CONTINUA = 0. La ganancia de evaluación recompensa únicamente BAJA+2: +1.072.500 por acierto y −27.500 por contacto incorrecto.
- **Entrenamiento final:** marzo, abril, mayo y junio de 2021, sin undersampling de clases ni pesos de recencia. El bagging de LightGBM toma 70% de filas por iteración; no es un muestreo específico de CONTINUA.
- **Historia:** lags 1 y 2 y deltas contra esos lags de las variables originales, excepto ID y mes. Los lags generales buscan el mes calendario exacto. Se usa julio para construir la historia de agosto.
- **Calidad:** `ccajas_depositos` pasa a NA en mayo antes de reconstruir su historia. Para reproducir exactamente el entrenamiento, esa familia conserva lags por observaciones previas del cliente; es la única excepción a los lags por calendario. Los faltantes se preservan.
- **Rankings:** después de calcular historia y convertir a `float32`, 365 campos monetarios actuales e históricos se reemplazan por rankings dentro de cada mes. Los positivos quedan en (0, 1], los negativos en [−1, 0), cero conserva cero y NA conserva NA. Los empates usan rango medio. La población de referencia es el conjunto completo de clientes observado en cada mes, sin usar etiquetas.
- **Orden de columnas:** fijado en `configuracion.json`. Se conservan ID, mes, personales y fechas.
- **Parámetros fijos:** learning rate 0,005; max_bin 31; min_data_in_leaf 0; feature_fraction 0,50; feature_fraction_bynode 0,20; bagging_fraction 0,70; bagging_freq 1. `deterministic` y `force_col_wise` están activados.
- **Ensamble final:** diez componentes del pool de doce, índices **0, 1, 2, 3, 4, 6, 7, 8, 10 y 11**, en ese orden y con igual peso. Las semillas y parámetros están registrados; no se vuelven a elegir al ejecutar.

## 4. De dónde salen los hiperparámetros

`01_optimizacion/estudio.sqlite` es la base Optuna original: **12 trials completos y uno interrumpido**. Se incluyen las doce configuraciones, aun cuando el archivo final utiliza diez.

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

La búsqueda puede tardar varias horas. No es necesaria para reproducir el CSV final y no sobrescribe parámetros, estudio ni modelos de la entrega. Los folds fueron usados para desarrollo; no son un test independiente.

## 5. Reentrenar los modelos finales

Para entrenar nuevamente los diez componentes con las configuraciones y semillas fijadas:

```bash
python entrenar.py --datos salida/datos --salida reentrenamiento
```

También puede iniciarse desde el CSV con `--dataset "/ruta/competencia_01.csv"` en lugar de `--datos`. La corrida usa todas las filas etiquetadas de marzo–junio. Guarda diez modelos nuevos, mide sus diferencias contra las probabilidades históricas y registra el tiempo de cada uno.

Generar el CSV usando esos nuevos modelos:

```bash
python reproducir.py --dataset "/ruta/competencia_01.csv" --modelos reentrenamiento --salida salida_reentrenada
```

La reproducción exacta verificada se realiza con los modelos incluidos. El reentrenamiento depende también de la versión nativa de LightGBM, arquitectura y aritmética del entorno; por eso tiene comprobaciones explícitas y no se promete igualdad en cualquier plataforma. `entorno.json` registra las versiones de la corrida verificada. Como comprobación adicional, se reentrenó el componente 08 en 10 minutos y 42 segundos: sus probabilidades coincidieron exactamente. No se repitieron los diez entrenamientos ni toda la búsqueda durante la preparación de esta carpeta.

## 6. Selección de la lista

Se construyeron veinte combinaciones de diez modelos del pool de doce. El ensamble 07 se eligió por tener la mayor coincidencia promedio de clientes con las otras diecinueve listas; los empates se resolvieron por nombre de archivo. Ese criterio no usa etiquetas de agosto ni ganancias públicas individuales.

El corte de 10.500 y la elección de familia se apoyaron en experimentos temporales y resultados públicos de desarrollo. La lista individual no tiene una ganancia pública identificada dentro del paquete; no se le atribuye la media de las veinte listas. El archivo aquí fijado es la predicción seleccionada para la entrega, independientemente de las corridas exploratorias anteriores.

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
    prediccion_final.csv
    predicciones_referencia.npz
VERIFICACION.json
SHA256SUMS.json
```

No hace falta ningún archivo de otros directorios del proyecto. Las salidas y el dataset quedan excluidos mediante `.gitignore`.
