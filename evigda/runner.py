import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback

import torch
import torch_geometric

from evigda.config import TASKS, PRIMARY_SELECTION
from evigda.data import load_target
from evigda.trainer import adapt


ROOT = Path(__file__).resolve().parent.parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def environment():
    return dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
                torch_geometric=torch_geometric.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                platform=platform.platform(), command=sys.argv,
                nondeterminism='CUDA sparse reductions may vary slightly across runs/devices.')


def source_paths(config):
    if config.get('checkpoint_dir'):
        directory = ROOT / config['checkpoint_dir']
        return [directory / name / ('model_' + name + '.pth') for name in config['sources']]
    if config['task'] == 'graph':
        return [ROOT / 'mypretrain' / ('model_' + name + '.pth') for name in config['sources']]
    return [ROOT / 'mypretrain' / config['target'] / name / ('model_' + name + '.pth')
            for name in config['sources']]


def evaluate(predictions, labels, config):
    predicted = predictions['predicted_labels']
    correct = int((predicted == labels).sum())
    accuracy = 100.0 * correct / labels.numel()
    classes = predictions['logits'].shape[1]
    confusion = torch.bincount(labels * classes + predicted, minlength=classes * classes).reshape(classes, classes)
    return dict(
        target=config['target'], seed=config['seed'],
        epochs=config['epochs'], accuracy_percent=accuracy, correct=correct, total=labels.numel(),
        evaluation_scope='all target instances, as specified in Appendix I',
        checkpoint_selection=config['checkpoint_policy'],
        source_checkpoint_provenance='supplied historical weights; original source-model selection not independently verified',
        confusion_matrix=confusion.tolist(),
        source_accuracies={name: 100.0 * (value == labels).sum().item() / labels.numel()
                           for name, value in zip(config['sources'], predictions['source_predictions'])},
        source_ensemble_accuracy=100.0 * (predictions['source_ensemble_prediction'] == labels).sum().item() / labels.numel(),
    )


def write_summary(output, results):
    dump_json(output / 'summary.json', results)
    fields = ['target', 'seed', 'epochs', 'accuracy_percent', 'correct', 'total']
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', nargs='+', default=['all'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--config', type=Path, required=True, help='Complete frozen configurations')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; refusing a silent CPU fallback.')
    targets = list(TASKS) if args.targets == ['all'] else args.targets
    unknown = set(targets) - set(TASKS)
    if unknown:
        parser.error('Unknown targets: {}'.format(sorted(unknown)))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dump_json(output / 'environment.json', environment())
    configurations = json.loads(args.config.read_text(encoding='utf-8-sig'))
    results = []
    failures = []
    # Keep one immutable source snapshot for this process, including when
    # multiple targets run sequentially and files are later edited elsewhere.
    code_paths = sorted((ROOT / 'evigda').glob('*.py')) + [ROOT / 'main.py']
    code_contents = {path: path.read_bytes() for path in code_paths}
    for target in targets:
        run_dir = output / target
        if run_dir.exists():
            raise FileExistsError('Refusing to overwrite an existing run: {}'.format(run_dir))
        run_dir.mkdir()
        config = configurations[target]
        if config['target'] != target or config['primary_selection'] != PRIMARY_SELECTION:
            raise ValueError('Configuration differs from the retained Table 1 method.')
        if config['epochs'] < 1:
            raise ValueError('Epoch count must be positive.')
        dump_json(run_dir / 'config.json', config)
        try:
            device = torch.device(args.device)
            inputs, labels, dataset_path = load_target(ROOT, config, device)
            checkpoints = source_paths(config)
            snapshot = run_dir / 'code_snapshot'
            for source in code_paths:
                destination = snapshot / source.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(code_contents[source])
            provenance = dict(dataset=dict(path=str(dataset_path.relative_to(ROOT)), sha256=sha256(dataset_path)),
                              source_checkpoints=[dict(path=str(path.relative_to(ROOT)), sha256=sha256(path)) for path in checkpoints],
                              code={str(path.relative_to(ROOT)): hashlib.sha256(code_contents[path]).hexdigest() for path in code_paths},
                              config_sha256=sha256(run_dir / 'config.json'))
            dump_json(run_dir / 'provenance.json', provenance)
            predictions, metadata, duration = adapt(inputs, checkpoints, config, run_dir, device)
            # First and only access to labels for scoring: adaptation and checkpoint saving are complete.
            metrics = evaluate(predictions, labels, config)
            metrics.update(primary=metadata['primary'], duration_seconds=duration,
                           selected_epoch=metadata['selected_epoch'],
                           checkpoint_sha256=sha256(run_dir / 'model_final.pt'),
                           predictions_sha256=sha256(run_dir / 'predictions.pt'))
            dump_json(run_dir / 'metrics.json', metrics)
            results.append(metrics)
            print('[RESULT] {} {:.4f}%'.format(target, metrics['accuracy_percent']), flush=True)
            del inputs, labels, predictions
        except Exception:
            detail = traceback.format_exc()
            (run_dir / 'error.txt').write_text(detail, encoding='utf-8')
            failures.append(target)
            print(detail, flush=True)
        finally:
            torch.cuda.empty_cache()
            write_summary(output, results)
    dump_json(output / 'status.json', dict(completed=[row['target'] for row in results], failed=failures))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
