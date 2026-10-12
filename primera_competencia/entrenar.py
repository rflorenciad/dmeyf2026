"""Reentrenamiento de los doce componentes del ensamble final."""
import argparse
import gc
import json
import time
from pathlib import Path
import lightgbm as lgb
import numpy as np
from src.datos import ROOT, CONFIG, preparar, sha256


def cargar_entrenamiento(data, months, threads):
    arrays = [np.load(data / f'x_{m}.npy', mmap_mode='r') for m in months]
    x = np.concatenate(arrays)
    labels = np.concatenate([np.load(data / f'y_{m}.npy') for m in months])
    assert (labels != 255).all()
    d = lgb.Dataset(x, label=np.isin(labels, [1, 2]).astype('int8'),
                    feature_name=CONFIG['features'], free_raw_data=True,
                    params={'max_bin': 31, 'feature_pre_filter': False, 'num_threads': threads})
    d.construct()
    return d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--dataset', type=Path)
    source.add_argument('--datos', type=Path, help='Matrices creadas por reproducir.py.')
    parser.add_argument('--salida', type=Path, default=Path('reentrenamiento'))
    parser.add_argument('--indices', type=int, nargs='+', help='Subconjunto opcional para comprobar una corrida.')
    args = parser.parse_args()
    args.salida.mkdir(parents=True, exist_ok=True)
    data = preparar(args.dataset, args.salida / 'datos') if args.dataset else args.datos.resolve()
    for m in CONFIG['months']:
        assert sha256(data / f'x_{m}.npy') == CONFIG['matrix_sha256'][str(m)]
        assert sha256(data / f'y_{m}.npy') == CONFIG['labels_sha256'][str(m)]
    configs = json.loads((ROOT / '01_optimizacion/configuraciones.json').read_text())
    entries = [m for m in CONFIG['models'] if args.indices is None or m['index'] in args.indices]
    if not entries:
        raise ValueError('No hay componentes seleccionados.')
    reference = np.load(ROOT / '03_prediccion_final/predicciones_referencia.npz')
    xt = np.load(data / 'x_202108.npy', mmap_mode='r')
    d = cargar_entrenamiento(data, CONFIG['train_months'], entries[0]['threads'])
    report = []
    for entry in entries:
        start = time.monotonic()
        i = entry['index']
        c = configs[i]
        params = dict(CONFIG['fixed_params'], num_leaves=c['num_leaves'],
                      min_sum_hessian_in_leaf=c['min_sum_hessian_in_leaf'],
                      seed=c['seed'], num_threads=entry['threads'])
        b = lgb.train(params, d, num_boost_round=c['num_iterations'])
        path = args.salida / f'modelo_{i:02d}.txt'
        b.save_model(str(path))
        pred = b.predict(xt, num_threads=entry['threads'])
        j = CONFIG['selected_model_indices'].index(i)
        error = float(np.max(np.abs(pred - reference['probabilities'][j])))
        record = {'index': i, 'sha256': sha256(path), 'max_abs_prediction_error': error,
                  'seconds': round(time.monotonic() - start, 2)}
        report.append(record)
        (args.salida / 'VERIFICACION_ENTRENAMIENTO.json').write_text(json.dumps(report, indent=2) + '\n')
        print(record, flush=True)
        if error > 1e-12:
            raise ValueError('Reentrenamiento distinto; revisar versiones y entorno antes de continuar.')
        del b
        gc.collect()


if __name__ == '__main__':
    main()
