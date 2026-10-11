"""Construcción de variables mensuales y verificación de las matrices."""
from pathlib import Path
import gc
import hashlib
import json
import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'configuracion.json').read_text())


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def rank_signo(values):
    values = np.asarray(values)
    result = np.full(values.shape, np.nan, dtype='float32')
    finite = np.isfinite(values)
    result[finite & (values == 0)] = 0
    for sign in (1, -1):
        use = finite & (values * sign > 0)
        if use.any():
            result[use] = sign * pd.Series(np.abs(values[use])).rank(
                method='average', pct=True).to_numpy(dtype='float32')
    return result


def preparar(dataset, output):
    dataset = Path(dataset).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    months = CONFIG['months']
    con = duckdb.connect()
    con.execute("SET threads=3; SET memory_limit='8GB'")
    relation = con.read_csv(str(dataset), header=True)
    missing = set(CONFIG['raw_features']) - set(relation.columns)
    if missing:
        raise ValueError(f'Faltan columnas: {sorted(missing)}')
    relation.create_view('entrada')
    columns = ','.join('"' + x + '"' for x in CONFIG['raw_features'])
    d = con.execute(f'SELECT {columns} FROM entrada ORDER BY numero_de_cliente, foto_mes').df()
    if d.duplicated(['numero_de_cliente', 'foto_mes']).any():
        raise ValueError('Hay pares cliente/mes duplicados.')
    if sorted(d.foto_mes.unique().tolist()) != months:
        raise ValueError('Se esperan los meses 202103–202108 de la competencia.')
    # Los identificadores conservan int64; el modelo recibe una copia float32.
    ids = {m: d.loc[d.foto_mes.eq(m), 'numero_de_cliente'].to_numpy(dtype='int64') for m in months}
    parts = {m: d[d.foto_mes.eq(m)].set_index('numero_de_cliente', drop=False) for m in months}
    supplied = None
    if 'clase_ternaria' in relation.columns:
        supplied = con.execute('SELECT clase_ternaria FROM entrada ORDER BY numero_de_cliente, foto_mes').df()['clase_ternaria']
    con.close()
    d.loc[d.foto_mes.eq(202105), 'ccajas_depositos'] = np.nan
    # Esta familia conserva la regla de observaciones previas del entrenamiento.
    dep = {'ccajas_depositos': d.ccajas_depositos}
    for lag in (1, 2):
        dep[f'lag{lag}_ccajas_depositos'] = d.groupby('numero_de_cliente', sort=False).ccajas_depositos.shift(lag)
        dep[f'delta{lag}_ccajas_depositos'] = d.ccajas_depositos - dep[f'lag{lag}_ccajas_depositos']
    records = {}
    for month in months:
        current = parts[month]
        index = ids[month]
        values = {c: current[c].to_numpy(dtype='float64', na_value=np.nan) for c in CONFIG['raw_features']}
        for c in CONFIG['raw_features']:
            if c in ('numero_de_cliente', 'foto_mes'):
                continue
            for lag in (1, 2):
                past = month - lag
                prior = parts[past][c].reindex(index).to_numpy(dtype='float64', na_value=np.nan) if past in parts else np.full(len(index), np.nan)
                values[f'lag{lag}_{c}'] = prior
                values[f'delta{lag}_{c}'] = values[c] - prior
        mask = d.foto_mes.eq(month)
        for name, series in dep.items():
            values[name] = series.loc[mask].to_numpy(dtype='float64', na_value=np.nan)
        x = np.asfortranarray(np.column_stack([values[c] for c in CONFIG['features']]), dtype='float32')
        for col in CONFIG['rank_columns']:
            j = CONFIG['features'].index(col)
            x[:, j] = rank_signo(x[:, j])
        if not np.isfinite(x[~np.isnan(x)]).all():
            raise ValueError('La matriz contiene valores infinitos.')
        y = np.full(len(index), 255, dtype='uint8')
        if month <= 202106:
            if supplied is not None:
                label = supplied.loc[mask]
                if not label.isin(['CONTINUA', 'BAJA+1', 'BAJA+2']).all():
                    raise ValueError(f'Etiquetas incompletas en {month}.')
                y = label.map({'CONTINUA': 0, 'BAJA+1': 1, 'BAJA+2': 2}).to_numpy(dtype='uint8')
            else:
                next1 = np.isin(index, ids[month + 1])
                next2 = np.isin(index, ids[month + 2])
                y = np.where(next2, 0, np.where(next1, 2, 1)).astype('uint8')
        np.save(output / f'x_{month}.npy', x)
        np.save(output / f'ids_{month}.npy', index)
        np.save(output / f'y_{month}.npy', y)
        record = {}
        for prefix, key in [('x', 'matrix_sha256'), ('ids', 'ids_sha256'), ('y', 'labels_sha256')]:
            digest = sha256(output / f'{prefix}_{month}.npy')
            expected = CONFIG[key][str(month)]
            if digest != expected:
                raise ValueError(f'No coincide {prefix}_{month}: {digest}, esperado {expected}')
            record[prefix] = digest
        records[str(month)] = record
        print(f'Mes {month}: {len(index):,} filas, {x.shape[1]} variables, hashes verificados.', flush=True)
        del x, values
        gc.collect()
    (output / 'VERIFICACION_DATOS.json').write_text(json.dumps({'dataset_sha256': sha256(dataset), 'months': records}, indent=2) + '\n')
    return output
