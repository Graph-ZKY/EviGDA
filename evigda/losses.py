import torch
from torch.nn import functional as F

EPS = 1e-8


def concentration(evidence_logits):
    return F.softplus(evidence_logits, beta=5) + 1.0


def decision_logits(classification_logits, evidence_logits, rule='classification'):
    """Explicit inference rule; adaptation losses still use both original heads."""
    if rule == 'classification':
        return classification_logits
    alpha = concentration(evidence_logits)
    probabilities = alpha / alpha.sum(1, keepdim=True)
    if rule == 'hybrid':
        probabilities = 0.5 * (probabilities + classification_logits.softmax(1))
    else:
        raise ValueError('Unknown inference rule: {}'.format(rule))
    return probabilities.clamp_min(EPS).log()


def uncertainty(alpha, mode='sum'):
    if mode not in ('sum', 'max', 'relative_sum'):
        raise ValueError('Unknown uncertainty mode: {}'.format(mode))
    strength = alpha.max(1).values if mode == 'max' else alpha.sum(1)
    value = alpha.shape[1] / strength.clamp_min(EPS)
    if mode == 'relative_sum':
        # Unlabeled, model-specific scaling for cross-expert comparisons.
        # The ordering of uncertainty within each expert is unchanged.
        value = value / value.mean().clamp_min(EPS)
    return value


def uniform_dirichlet_kl(alpha):
    strength = alpha.sum(1, keepdim=True)
    classes = alpha.shape[1]
    log_normalizer = torch.lgamma(strength) - torch.lgamma(alpha).sum(1, keepdim=True)
    uniform_normalizer = torch.lgamma(alpha.new_tensor(float(classes)))
    expectation = ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(strength))).sum(1, keepdim=True)
    return (log_normalizer - uniform_normalizer + expectation).squeeze(1)


def evidential_loss(alpha, pseudo_labels, epoch, horizon=50, kind='digamma', kl_weight=0.1):
    labels = F.one_hot(pseudo_labels, num_classes=alpha.shape[1]).to(alpha.dtype)
    strength = alpha.sum(1, keepdim=True)
    if kind == 'mse':
        fit = ((labels - alpha / strength) ** 2 +
               alpha * (strength - alpha) / (strength.square() * (strength + 1))).sum(1)
    else:
        fit = (labels * (torch.digamma(strength) - torch.digamma(alpha))).sum(1)
    incorrect_alpha = (alpha - 1) * (1 - labels) + 1
    anneal = min(1.0, epoch / max(horizon, 1))
    return (fit + kl_weight * anneal * uniform_dirichlet_kl(incorrect_alpha)).mean()


def fuse_evidence(alpha_a, alpha_b, legacy_scaling=False):
    """Dempster-Shafer fusion with explicit class-axis conflict calculation.

    The old positional torch.diagonal call treated -2 as an offset instead of
    a dimension. Computing the diagonal product directly avoids this bug.
legacy_scaling retains the original confidence reweighting as an explicit option.
"""
    classes = alpha_a.shape[1]
    opinions = []
    for alpha in (alpha_a, alpha_b):
        if legacy_scaling:
            alpha = alpha * alpha.max(1, keepdim=True).values / alpha.sum(1, keepdim=True)
        strength = alpha.sum(1, keepdim=True).clamp_min(EPS)
        belief = (alpha - 1).clamp_min(0) / strength
        ignorance = (classes / strength).clamp(EPS, 1.0)
        opinions.append((belief, ignorance))
    (belief_a, ignorance_a), (belief_b, ignorance_b) = opinions
    conflict = (belief_a.sum(1) * belief_b.sum(1) -
                (belief_a * belief_b).sum(1)).clamp(0.0, 1.0 - 1e-3)
    normalizer = (1 - conflict).unsqueeze(1)
    belief = (belief_a * belief_b + belief_a * ignorance_b +
              belief_b * ignorance_a) / normalizer
    ignorance = ignorance_a * ignorance_b / normalizer
    fused = belief * (classes / ignorance.clamp_min(EPS)) + 1
    if legacy_scaling:
        fused = fused / fused.max(1, keepdim=True).values * fused.sum(1, keepdim=True)
    return fused, conflict


def information_loss(logits, diversity=False):
    probabilities = logits.softmax(1)
    entropy = -(probabilities * probabilities.clamp_min(EPS).log()).sum(1).mean()
    if diversity:
        mean = probabilities.mean(0)
        entropy = entropy + (mean * mean.clamp_min(EPS).log()).sum()
    return entropy


def local_divergence(logits, propagated_logits, kind='kl'):
    """Prediction consistency with a bounded JS option for negative weights."""
    if kind == 'kl':
        return F.kl_div(logits.log_softmax(1), propagated_logits.softmax(1), reduction='batchmean')
    first, second = logits.softmax(1), propagated_logits.softmax(1).detach()
    mixture = 0.5 * (first + second)
    log_mixture = mixture.clamp_min(EPS).log()
    # F.kl_div differentiates its target through log(target); exact softmax zeros
    # can therefore produce NaN gradients. Log-softmax remains finite here.
    first_kl = (first * (logits.log_softmax(1) - log_mixture)).sum(1)
    second_kl = (second * (propagated_logits.detach().log_softmax(1) - log_mixture)).sum(1)
    return 0.5 * (first_kl + second_kl).mean()


def aggregation_loss(logits, alpha, eval_logits, eval_alpha, teachers, config):
    voting = logits.sum() * 0
    evidential = alpha.sum() * 0
    selected = []
    for teacher_logits, teacher_alpha in teachers:
        with torch.no_grad():
            mask = (uncertainty(teacher_alpha, config['uncertainty']) <
                    uncertainty(eval_alpha, config['uncertainty']) - config['margin']).float()
            pseudo = (eval_logits + teacher_logits).argmax(1)
        voting = voting + (F.cross_entropy(logits, pseudo, reduction='none') * mask).mean()
        fused, _ = fuse_evidence(alpha, teacher_alpha, config['legacy_fusion_scaling'])
        # Preserve KL(primary categorical || fused categorical).
        divergence = F.kl_div((fused / fused.sum(1, keepdim=True)).log(),
                             alpha / alpha.sum(1, keepdim=True), reduction='none').sum(1)
        if config['fusion_class_mean']:
            divergence = divergence / alpha.shape[1]
        evidential = evidential + (divergence * mask).mean()
        selected.append(mask.mean().item())
    return voting, evidential, selected
