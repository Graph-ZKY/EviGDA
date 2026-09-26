"""Unsupervised adaptation; this module never receives target labels."""
import json
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.nn import APPNP

from .losses import aggregation_loss, concentration, decision_logits, evidential_loss, information_loss, local_divergence
from .memory import PredictionMemory
from .averaging import ParameterAverage
from .models import load_model


PRIMARY_SELECTION_RULES = ('inverse_max_argmin',)
PRIMARY_SCORE_DEFINITIONS = {'inverse_max_argmin': 'mean(1 / max_c(alpha_c))'}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def predict(model, inputs):
    model.eval()
    return model(inputs.x, inputs.adjacency, inputs.batch)


def select_primary(predictions, rule='inverse_max_argmin'):
    """Select once from unlabeled target evidence, without task-specific rules.

The same inverse-maximum rule is used for every retained Table 1 task.
Exact ties retain the first source in the recorded input order.
"""
    if rule not in PRIMARY_SELECTION_RULES:
        raise ValueError('Unknown primary selection rule: ' + str(rule))
    if not predictions:
        raise ValueError('At least one source prediction is required.')
    scores = []
    for prediction in predictions:
        alpha = concentration(prediction[1])
        value = alpha.max(dim=1).values.reciprocal().mean()
        scores.append(value.item())
    if not np.isfinite(scores).all():
        raise ValueError('Source selection scores must be finite.')
    primary_idx = int(np.argmin(scores))
    return primary_idx, scores


def source_teacher_probabilities(predictions, primary_index, rule='primary', temperature=0.05):
    """Build a fixed soft target from unlabeled source predictions only."""
    if rule == 'primary':
        indices = [primary_index]
    elif rule == 'complementary_information':
        indices = [index for index in range(len(predictions)) if index != primary_index]
    else:
        raise ValueError('Unknown anchor target: {}'.format(rule))
    if not indices:
        raise ValueError('A complementary teacher requires at least two source models.')
    probabilities = torch.stack([predictions[index][0].softmax(1) for index in indices])
    if rule == 'complementary_information':
        if temperature <= 0:
            raise ValueError('Teacher temperature must be positive.')
        # This weights only the frozen complementary predictions; the trainable
        # primary was already selected by the configured evidence-only rule.
        scores = torch.stack([information_loss(predictions[index][0], diversity=True) for index in indices])
        weights = (-scores / temperature).softmax(0)
        return (probabilities * weights[:, None, None]).sum(0).detach()
    return probabilities.mean(0).detach()


def adapt(inputs, checkpoint_paths, config, run_dir, device):
    """Optimize one expert for the predetermined Table 1 epoch budget.

Source data and target labels are not arguments. Frozen experts are evaluated
once; their cached outputs provide the same deterministic supervision each step.
"""
    if config['checkpoint_policy'] != 'fixed_final_epoch':
        raise ValueError('Only predetermined final checkpoints are supported.')
    seed_everything(config['seed'])
    models, architectures, initial = [], [], []
    for path in checkpoint_paths:
        model, architecture = load_model(path, config['dropout'], device, config.get('ego_mix', 0.0))
        if architecture['input_dim'] != inputs.x.shape[1]:
            raise ValueError('Checkpoint input dimension differs from target features.')
        models.append(model)
        architectures.append(architecture)
        if config.get('center_source_logits', False):
            # Optional domain-bias correction based only on target features.
            # The linear head's bias absorbs the mean shift, so a saved model
            # performs the same operation without a separate inference rule.
            with torch.no_grad():
                source_logits, _, _ = predict(model, inputs)
                model.clf.fc.bias.sub_(source_logits.mean(0))
        initial.append(predict(model, inputs))
    primary_index, selection_scores = select_primary(initial, config['primary_selection'])
    primary = models[primary_index]
    teacher_outputs = []
    for index, model in enumerate(models):
        if index != primary_index:
            teacher_outputs.append((initial[index][0], concentration(initial[index][1])))
            for parameter in model.parameters():
                parameter.requires_grad_(False)
    rule = config.get('prediction_rule', 'classification')
    initial_decisions = [decision_logits(value[0], value[1], rule) for value in initial]
    source_predictions = [value.argmax(1).cpu() for value in initial_decisions]
    ensemble_prediction = torch.stack([value.softmax(1) for value in initial_decisions]).mean(0).argmax(1).cpu()
    del initial_decisions
    metadata = dict(primary=config['sources'][primary_index], primary_index=primary_index,
                    selection_rule=config['primary_selection'],
                    selection_score_definition=PRIMARY_SCORE_DEFINITIONS[config['primary_selection']],
                    selection_direction='minimize',
                    selection_scores=selection_scores, architectures=architectures,
                    trainable_parameters=sum(p.numel() for p in primary.active_parameters()))
    (run_dir / 'source_selection.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('[SELECT] {}: {} | scores={}'.format(config['target'], metadata['primary'], selection_scores), flush=True)

    groups = [
        dict(params=primary.backbone.parameters(), lr=config['lr'] * config['backbone_lr_scale']),
        dict(params=primary.clf.parameters(), lr=config['lr']),
        dict(params=primary.evi.parameters(), lr=config['lr']),
    ]
    if config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(groups, momentum=config['momentum'],
                                    weight_decay=config['weight_decay'], nesterov=True)
    else:
        optimizer = torch.optim.Adam(groups, weight_decay=config['weight_decay'])
    initial_lrs = [group['lr'] for group in optimizer.param_groups]
    average_start = config.get('parameter_average_start', 0)
    average_interval = config.get('parameter_average_interval', 10)
    parameter_average = ParameterAverage() if average_start else None
    if parameter_average is not None and (not 1 <= average_start <= config['epochs'] or
            average_interval < 1 or config['checkpoint_policy'] != 'fixed_final_epoch'):
        raise ValueError('Parameter averaging requires a valid predetermined tail and fixed final epoch.')
    propagation = APPNP(K=config['propagation_steps'], alpha=config['teleport'], cached=True).to(device)
    initial_logits, _, initial_features = initial[primary_index]
    anchor_probabilities = source_teacher_probabilities(initial, primary_index, config.get('anchor_target', 'primary'), config.get('teacher_temperature', 0.05))
    memory_teacher_weight = config.get('memory_teacher_weight', 0.0)
    pseudo_query = config.get('pseudo_query', 'train')
    if pseudo_query not in ('train', 'eval'):
        raise ValueError('pseudo_query must be train or eval.')
    if not 0.0 <= memory_teacher_weight <= 1.0:
        raise ValueError('memory_teacher_weight must lie in [0, 1].')
    memory_probabilities = ((1.0 - memory_teacher_weight) * initial_logits.softmax(1) +
                            memory_teacher_weight * anchor_probabilities)
    memory_features = initial_features
    if config['topology_memory'] and inputs.batch is None:
        memory_features = propagation(initial_features, inputs.adjacency)
    memory = PredictionMemory(memory_features, memory_probabilities,
                              config['memory_momentum'], config['warm_start'],
                              config['exclude_self'], config['chunk_size'])
    # Retain only the primary network and cached teacher tensors on the GPU.
    del models, initial, initial_features, initial_logits, memory_features
    start_time = time.monotonic()
    history_path = run_dir / 'history.jsonl'
    last_losses = {}
    selected_epoch = config['epochs']
    for epoch in range(1, config['epochs'] + 1):
        eval_logits, eval_evidence, eval_features = predict(primary, inputs)
        with torch.no_grad():
            eval_alpha = concentration(eval_evidence)
            if config['topology_memory'] and inputs.batch is None:
                eval_features = propagation(eval_features, inputs.adjacency)
        primary.train()
        logits, evidence_logits, features = primary(inputs.x, inputs.adjacency, inputs.batch)
        alpha = concentration(evidence_logits)
        with torch.no_grad():
            if config['topology_memory'] and inputs.batch is None:
                features = propagation(features.detach(), inputs.adjacency)
            # Memory stores evaluation-mode features. A deterministic query can
            # avoid matching heavily dropped-out features against that memory.
            query_features = eval_features if pseudo_query == 'eval' else features
            pseudo_labels = memory.pseudo_labels(query_features, config['neighbors'])
            memory.update(eval_features, (1.0 - memory_teacher_weight) * eval_logits.softmax(1) +
                          memory_teacher_weight * anchor_probabilities)

        cross_entropy = F.cross_entropy(logits, pseudo_labels)
        evidence = evidential_loss(alpha, pseudo_labels, epoch, config['annealing_epochs'],
                                   config['evidence_loss'], config['evidence_kl_weight'])
        voting, fusion, mask_fractions = aggregation_loss(
            logits, alpha, eval_logits, eval_alpha, teacher_outputs, config)
        local = logits.sum() * 0
        if config['local_weight'] and inputs.batch is None:
            with torch.no_grad():
                propagated_logits = propagation(logits.detach(), inputs.adjacency)
            local = local_divergence(logits, propagated_logits, config['local_divergence'])
        entropy = information_loss(logits, config['diversity'])
        anchor = F.kl_div(logits.log_softmax(1), anchor_probabilities, reduction='batchmean')
        loss = (config['pseudo_weight'] * cross_entropy + config['edl_weight'] * evidence +
                config['voting_weight'] * voting + config['fusion_weight'] * fusion +
                config['local_weight'] * local + config['entropy_weight'] * entropy +
                config['source_anchor_weight'] * anchor)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite loss at epoch {}'.format(epoch))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(primary.active_parameters(), config['gradient_clip'])
        step = epoch - 1 if config['task'] == 'graph' else epoch
        decay = (1 + 10 * step / config['epochs']) ** (-0.75)
        for group, initial_lr in zip(optimizer.param_groups, initial_lrs):
            group['lr'] = initial_lr * decay
        optimizer.step()

        if parameter_average is not None and epoch >= average_start and (
                (epoch - average_start) % average_interval == 0 or epoch == config['epochs']):
            parameter_average.update(primary)

        if epoch == 1 or epoch % config['log_every'] == 0 or epoch == config['epochs']:
            last_losses = dict(epoch=epoch, loss=loss.item(), ce=cross_entropy.item(),
                               edl=evidence.item(), voting=voting.item(), fusion=fusion.item(),
                               local=local.item(), entropy=entropy.item(), anchor=anchor.item(), mask_fractions=mask_fractions,
                               pseudo_counts=torch.bincount(pseudo_labels, minlength=alpha.shape[1]).tolist(),
                               elapsed_seconds=time.monotonic() - start_time)
            with history_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(last_losses) + '\n')
            print('[TRAIN] {} epoch={}/{} loss={:.5f} time={:.1f}s'.format(
                config['target'], epoch, config['epochs'], loss.item(), last_losses['elapsed_seconds']), flush=True)

    if parameter_average is not None:
        parameter_average.apply(primary)
        metadata['parameter_averaging'] = dict(start_epoch=average_start, interval=average_interval,
                                               updates=parameter_average.updates)
    metadata['selected_epoch'] = selected_epoch
    metadata['checkpoint_policy'] = config['checkpoint_policy']
    (run_dir / 'source_selection.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    logits, evidence_logits, features = predict(primary, inputs)
    classification_logits = logits
    logits = decision_logits(classification_logits, evidence_logits, rule)
    checkpoint = dict(state_dict=primary.state_dict(), config=config, metadata=metadata,
                      epoch=selected_epoch, selection=config['checkpoint_policy'])
    torch.save(checkpoint, str(run_dir / 'model_final.pt'))
    predictions = dict(logits=logits.cpu(), evidence_logits=evidence_logits.cpu(),
                       classification_logits=classification_logits.cpu(),
                       predicted_labels=logits.argmax(1).cpu(),
                       source_predictions=source_predictions,
                       source_ensemble_prediction=ensemble_prediction)
    torch.save(predictions, str(run_dir / 'predictions.pt'))
    return predictions, metadata, time.monotonic() - start_time
