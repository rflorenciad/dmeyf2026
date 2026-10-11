"""Origen de parámetros, reevaluación temporal y búsqueda nueva con Optuna."""
import argparse
import gc
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from src.datos import ROOT, CONFIG, preparar, sha256
from entrenar import cargar_entrenamiento

HERE = Path(__file__).resolve().parent


def estudio_original():
    names = optuna.study.get_all_study_names(storage=f'sqlite:///{HERE / "estudio.sqlite"}')
    assert len(names) == 1
    return optuna.load_study(study_name=names[0], storage=f'sqlite:///{HERE / "estudio.sqlite"}')


def verificar():
    study = estudio_original()
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    configs = json.loads((HERE / 'configuraciones.json').read_text())
    assert len(complete) == len(configs) == 12
    for c, t in zip(configs, complete):
        assert c['trial'] == t.number and c['seed'] == t.user_attrs['seed']
        assert c['num_leaves'] == t.params['num_leaves']
        assert c['min_sum_hessian_in_leaf'] == t.params['min_sum_hessian_in_leaf']
        assert c['num_iterations'] == round(np.mean([f['best_iteration'] for f in t.user_attrs['folds']]))
        assert c['folds'] == t.user_attrs['folds'] and c['auc'] == t.value
    print('Las doce configuraciones coinciden con el estudio original.', flush=True)
    return study, configs


def evaluate(params, cap, folds, destination, trial, native=False):
    metrics = []
    for month, data, x, y, valid, ids in folds:
        counter, cached = [0], [0.0]
        def periodic_auc(pred, dataset):
            counter[0] += 1
            if counter[0] == 1 or counter[0] % 10 == 0:
                cached[0] = float(roc_auc_score(dataset.get_label(), pred))
            return 'auc', cached[0], True
        metric = {'metric': 'auc'} if native else {'metric': 'None'}
        b = lgb.train(dict(params, **metric), data, num_boost_round=cap,
                      valid_sets=[valid], feval=None if native else periodic_auc,
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        pred = b.predict(x, num_threads=3)
        order = np.lexsort((ids, -pred))[:10500]
        best = b.best_iteration or b.current_iteration()
        metrics.append({'month': month, 'best_iteration': best, 'cap': cap,
                        'cap_reached': best >= cap,
                        'auc_both': roc_auc_score(np.isin(y, [1, 2]), pred),
                        'auc_baja2': roc_auc_score(y == 2, pred),
                        'gain_m': (1100000 * int((y[order] == 2).sum()) - 27500 * 10500) / 1e6})
        b.save_model(str(destination / f'trial{trial:02d}_{month}.txt'))
        print(f'Trial {trial}, mes {month}: {metrics[-1]}', flush=True)
        del b
        gc.collect()
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('modo', choices=['verificar', 'repetir', 'nueva'])
    group = p.add_mutually_exclusive_group()
    group.add_argument('--dataset', type=Path)
    group.add_argument('--datos', type=Path)
    p.add_argument('--salida', type=Path, default=Path('optimizacion_nueva'))
    p.add_argument('--trials', type=int, default=12)
    args = p.parse_args()
    old, configs = verificar()
    if args.modo == 'verificar':
        return
    if args.dataset is None and args.datos is None:
        p.error('Indicar --dataset o --datos para entrenar.')
    args.salida.mkdir(parents=True, exist_ok=True)
    data = preparar(args.dataset, args.salida / 'datos') if args.dataset else args.datos.resolve()
    for month in CONFIG['months']:
        assert sha256(data / f'x_{month}.npy') == CONFIG['matrix_sha256'][str(month)]
        assert sha256(data / f'y_{month}.npy') == CONFIG['labels_sha256'][str(month)]
    folds = []
    for months, month in [([202103], 202105), ([202103, 202104], 202106)]:
        d = cargar_entrenamiento(data, months, 3)
        x = np.load(data / f'x_{month}.npy', mmap_mode='r')
        y = np.load(data / f'y_{month}.npy')
        valid = lgb.Dataset(x, label=np.isin(y, [1, 2]).astype('int8'), reference=d)
        folds.append((month, d, x, y, valid, np.load(data / f'ids_{month}.npy')))
    if args.modo == 'repetir':
        report = []
        for c in configs:
            t = old.trials[c['trial']]
            params = dict(CONFIG['fixed_params'], seed=c['seed'], num_leaves=c['num_leaves'],
                          min_sum_hessian_in_leaf=c['min_sum_hessian_in_leaf'])
            metrics = evaluate(params, t.params['num_iterations'], folds, args.salida, t.number, native=t.number == 0)
            report.append({'trial': t.number, 'folds': metrics, 'original_folds': c['folds']})
            (args.salida / 'reevaluacion.json').write_text(json.dumps(report, indent=2) + '\n')
        return
    study = optuna.create_study(study_name='rank_directo',
            storage=f'sqlite:///{args.salida.resolve() / "estudio.sqlite"}',
            direction='maximize', sampler=optuna.samplers.TPESampler(seed=882026, n_startup_trials=4))
    def objective(t):
        cap = t.suggest_categorical('num_iterations', [2000, 4000, 8000])
        leaves = t.suggest_categorical('num_leaves', [8, 16, 24, 32, 48, 64])
        hessian = t.suggest_float('min_sum_hessian_in_leaf', 1, 100, log=True)
        seed = 882026 + t.number * 1009
        params = dict(CONFIG['fixed_params'], num_leaves=leaves, min_sum_hessian_in_leaf=hessian, seed=seed)
        metrics = evaluate(params, cap, folds, args.salida, t.number)
        t.set_user_attr('folds', metrics)
        t.set_user_attr('seed', seed)
        return float(np.mean([m['auc_both'] for m in metrics]))
    study.optimize(objective, n_trials=args.trials)
    print(f'Búsqueda nueva guardada en {args.salida}. No modifica la configuración final.')


if __name__ == '__main__':
    main()
