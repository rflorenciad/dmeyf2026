"""Genera y verifica el CSV final desde el dataset de la competencia."""
import argparse
import gzip
import json
import time
from pathlib import Path
import lightgbm as lgb
import numpy as np
from src.datos import ROOT, CONFIG, preparar, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--dataset', type=Path)
    source.add_argument('--datos', type=Path, help='Matrices previamente reconstruidas y verificadas.')
    p.add_argument('--salida', type=Path, default=Path('salida'))
    p.add_argument('--modelos', type=Path, help='Carpeta alternativa de modelos reentrenados.')
    args = p.parse_args()
    start = time.monotonic()
    output = args.salida.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = preparar(args.dataset, output / 'datos') if args.dataset else args.datos.resolve()
    if args.datos:
        for month in CONFIG['months']:
            for prefix, key in [('x', 'matrix_sha256'), ('ids', 'ids_sha256'), ('y', 'labels_sha256')]:
                assert sha256(data / f'{prefix}_{month}.npy') == CONFIG[key][str(month)]
    x = np.load(data / 'x_202108.npy', mmap_mode='r')
    ids = np.load(data / 'ids_202108.npy')
    reference = np.load(ROOT / '03_prediccion_final/predicciones_referencia.npz')
    assert np.array_equal(ids, reference['ids'])
    scores = []
    checks = []
    for j, entry in enumerate(CONFIG['models']):
        if args.modelos:
            path = args.modelos / f'modelo_{entry["index"]:02d}.txt'
            b = lgb.Booster(model_file=str(path))
        else:
            path = ROOT / entry['file']
            assert sha256(path) == entry['sha256_compressed']
            with gzip.open(path, 'rt') as f:
                b = lgb.Booster(model_str=f.read())
        assert b.feature_name() == CONFIG['features']
        pred = b.predict(x, num_threads=entry['threads'])
        assert np.isfinite(pred).all()
        error = float(np.max(np.abs(pred - reference['probabilities'][j])))
        if error > 1e-12:
            raise ValueError(f'Predicciones distintas para modelo {entry["index"]}: {error}')
        scores.append(pred)
        checks.append({'index': entry['index'], 'max_abs_error': error, 'file_sha256': sha256(path)})
        print(f'Modelo {entry["index"]:02d}: predicciones verificadas.', flush=True)
    scores = np.stack(scores)
    positions = {entry['index']: j for j, entry in enumerate(CONFIG['models'])}
    destinations = output / 'predicciones_finales'
    destinations.mkdir(exist_ok=True)
    files = []
    for entry in CONFIG['ensembles']:
        mean = scores[[positions[i] for i in entry['model_indices']]].mean(axis=0)
        selected = ids[np.lexsort((ids, -mean))[:CONFIG['cutoff']]]
        assert len(selected) == len(np.unique(selected)) == CONFIG['cutoff']
        expected = ROOT / entry['file']
        destination = destinations / expected.name
        destination.write_bytes(''.join(f'{int(client)}\n' for client in selected).encode('ascii'))
        digest = sha256(destination)
        if destination.read_bytes() != expected.read_bytes() or digest != entry['sha256']:
            raise ValueError(f'El ensamble {entry["ensemble"]} no coincide con la referencia.')
        files.append({'ensemble': entry['ensemble'], 'file': destination.name,
                      'rows': len(selected), 'sha256': digest, 'expected_sha256': entry['sha256']})
        print(f'Ensamble {entry["ensemble"]:02d}: CSV idéntico byte por byte.', flush=True)
    assert len(files) == 1
    report = {'status': 'identico_byte_por_byte', 'candidate_models': len(files),
              'rows_per_file': CONFIG['cutoff'], 'files': files, 'models': checks,
              'seconds': round(time.monotonic() - start, 2)}
    (output / 'VERIFICACION.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'OK: CSV final verificado en {destinations}', flush=True)


if __name__ == '__main__':
    main()
