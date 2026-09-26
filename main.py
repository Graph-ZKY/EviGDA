import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from evigda.config import TASKS, PRIMARY_SELECTION, read_configuration
from evigda.data import load_target
from evigda.losses import decision_logits
from evigda.models import EvidentialGCN
from evigda.runner import dump_json, environment, sha256

ROOT = Path(__file__).resolve().parent


def relative_path(value):
    """Read archived Windows paths on Windows and Linux."""
    return ROOT / value.replace('\\', '/')


def verify_files():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text(encoding='utf-8'))
    failures = []
    for item in manifest['files']:
        path = relative_path(item['path'])
        if not path.is_file() or path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
            failures.append(item['path'])
    if failures:
        raise RuntimeError('Missing or changed release files: {}'.format(', '.join(failures)))
    print('[CHECK] All {} release files match SHA-256.'.format(len(manifest['files'])), flush=True)


def read_reference():
    values = json.loads((ROOT / 'results/reference_results.json').read_text(encoding='utf-8'))
    records = {item['target']: item for item in values['results']}
    if any(item['primary_selection'] != PRIMARY_SELECTION for item in records.values()):
        raise ValueError('Reference checkpoints do not match the requested selection rule.')
    return records


def score_checkpoint(directory, device):
    checkpoint = torch.load(str(directory / 'model_final.pt'), map_location='cpu')
    config, metadata = checkpoint['config'], checkpoint['metadata']
    architecture = metadata['architectures'][metadata['primary_index']]
    model = EvidentialGCN(dropout=config['dropout'], **architecture).to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    inputs, labels, _ = load_target(ROOT, config, device)
    with torch.no_grad():
        classification, evidence, _ = model(inputs.x, inputs.adjacency, inputs.batch)
        logits = decision_logits(classification, evidence, config.get('prediction_rule', 'classification')).cpu()
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError('Non-finite predictions for ' + config['target'])
    predicted = logits.argmax(1)
    stored = torch.load(str(directory / 'predictions.pt'), map_location='cpu')
    if not torch.equal(predicted, stored['predicted_labels']):
        raise RuntimeError('Reloaded predictions differ from saved predictions: ' + str(directory))
    classes = logits.shape[1]
    confusion = torch.bincount(labels * classes + predicted, minlength=classes ** 2).reshape(classes, classes)
    recall = confusion.diag().double() / confusion.sum(1).double().clamp_min(1)
    total, correct = labels.numel(), int((predicted == labels).sum())
    result = dict(target=config['target'], primary=metadata['primary'],
                  primary_selection=config['primary_selection'], correct=correct, total=total,
                  accuracy_percent=100.0 * correct / total,
                  balanced_accuracy_percent=100.0 * recall.mean().item(),
                  maximum_predicted_class_fraction=torch.bincount(predicted).max().item() / total,
                  checkpoint_epoch=checkpoint['epoch'], prediction_rule=config.get('prediction_rule', 'classification'),
                  parameter_average_start=config.get('parameter_average_start', 0),
                  parameter_average_updates=metadata.get('parameter_averaging', {}).get('updates', 0),
                  confusion_matrix=confusion.tolist(), checkpoint_sha256=sha256(directory / 'model_final.pt'),
                  reloaded_predictions_identical=True,
                  maximum_logit_error=(logits - stored['logits']).abs().max().item())
    del inputs, labels, logits, model, stored, checkpoint
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result, predicted


def write_results(output, records):
    dump_json(output / 'comparison.json', records)
    fields = ['target', 'primary', 'primary_selection', 'reference_primary_selection',
              'accuracy_percent', 'reference_accuracy_percent', 'delta_reference',
              'correct', 'total', 'balanced_accuracy_percent',
              'maximum_predicted_class_fraction', 'checkpoint_epoch', 'prediction_rule',
              'parameter_average_start', 'parameter_average_updates',
              'matches_reference_correct_count', 'reference_prediction_disagreements',
              'reloaded_predictions_identical', 'maximum_logit_error']
    with (output / 'comparison.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['evaluate', 'train', 'check'], default='evaluate')
    parser.add_argument('--targets', nargs='+', default=['all'], help='all, or exact target names from README')
    parser.add_argument('--device', default='cuda:0', help='cuda:0 for the tested GPU setup; cpu for checkpoint evaluation')
    parser.add_argument('--output', type=Path, help='New output directory, relative to this release or absolute')
    args = parser.parse_args()
    verify_files()
    if args.mode == 'check':
        return
    targets = list(TASKS) if args.targets == ['all'] else args.targets
    if len(set(targets)) != len(targets) or set(targets) - set(TASKS):
        parser.error('Supply unique target names from: ' + ', '.join(TASKS))
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA is unavailable. For checkpoint evaluation, pass --device cpu.')
    torch.set_num_threads(4)
    variant = PRIMARY_SELECTION
    output = (ROOT / (args.output or Path('outputs') / (args.mode + '_' + variant))).resolve()
    if output.exists():
        raise FileExistsError('Output already exists. Choose a new --output: {}'.format(output))
    output.mkdir(parents=True)
    dump_json(output / 'environment.json', environment())
    references = read_reference()
    if args.mode == 'train':
        config = read_configuration()
        effective = output / 'effective_configs.json'
        dump_json(effective, config)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0')
        command = [sys.executable, '-B', '-u', '-m', 'evigda.runner',
                   '--targets'] + targets + ['--config', str(effective), '--output', str(output / 'training'),
                                            '--device', args.device, '--threads', '4']
        subprocess.run(command, cwd=str(ROOT), env=env, check=True)
    results = []
    for target in targets:
        reference = references[target]
        reference_dir = relative_path(reference['checkpoint_directory'])
        model_dir = output / 'training' / target if args.mode == 'train' else reference_dir
        result, predicted = score_checkpoint(model_dir, torch.device(args.device))
        if result['primary_selection'] != variant:
            raise ValueError('Checkpoint selection rule differs from the requested method: ' + target)
        original = torch.load(str(reference_dir / 'predictions.pt'), map_location='cpu')['predicted_labels']
        result.update(reference_accuracy_percent=reference['accuracy_percent'],
                      delta_reference=result['accuracy_percent'] - reference['accuracy_percent'],
                      matches_reference_correct_count=result['correct'] == reference['correct'],
                      reference_prediction_disagreements=int((predicted != original).sum()),
                      mode=args.mode, variant=result['primary_selection'],
                      reference_primary_selection=reference['primary_selection'])
        results.append(result)
        write_results(output, results)
        print('[{}] {} {:.4f}% | reference {:.4f}% | delta {:+.4f}'.format(
            args.mode.upper(), target, result['accuracy_percent'], reference['accuracy_percent'],
            result['delta_reference']), flush=True)
        if args.mode == 'evaluate' and (not result['matches_reference_correct_count'] or
                                        result['reference_prediction_disagreements']):
            raise RuntimeError('The supplied checkpoint did not reproduce its reference predictions.')
    dump_json(output / 'verification.json', dict(
        primary_selection=variant, targets=targets, mode=args.mode,
        all_match_reference_correct_counts=all(result['matches_reference_correct_count'] for result in results),
        all_match_reference_predictions=all(result['reference_prediction_disagreements'] == 0 for result in results)))
    print('Saved comparison.csv and comparison.json to {}'.format(output), flush=True)


if __name__ == '__main__':
    main()
