"""Genera y verifica la lista final desde el CSV de la competencia."""
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
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--salida', type=Path, default=Path('salida'))
    p.add_argument('--modelos', type=Path, help='Carpeta alternativa de modelos reentrenados.')
    args = p.parse_args()
    start = time.monotonic()
    output = args.salida.resolve()
    data = preparar(args.dataset, output / 'datos')
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
    mean = np.stack(scores).mean(axis=0)
    selected = ids[np.lexsort((ids, -mean))[:CONFIG['cutoff']]]
    assert len(selected) == len(np.unique(selected)) == 10500
    destination = output / 'prediccion_final.csv'
    destination.write_bytes(''.join(f'{int(client)}\n' for client in selected).encode('ascii'))
    digest = sha256(destination)
    expected = ROOT / '03_prediccion_final/prediccion_final.csv'
    identical = destination.read_bytes() == expected.read_bytes()
    report = {'status': 'identico_byte_por_byte' if identical else 'no_coincide', 'rows': len(selected), 'sha256': digest, 'expected_sha256': CONFIG['final_sha256'], 'models': checks, 'seconds': round(time.monotonic() - start, 2)}
    (output / 'VERIFICACION.json').write_text(json.dumps(report, indent=2) + '\n')
    if not identical or digest != CONFIG['final_sha256']:
        raise ValueError('La lista final no coincide con la referencia.')
    print(f'OK: {destination}\nSHA-256: {digest}', flush=True)


if __name__ == '__main__':
    main()
