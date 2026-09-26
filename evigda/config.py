import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / 'configs' / 'table1.json'
PRIMARY_SELECTION = 'inverse_max_argmin'
TASKS = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))


def read_configuration():
    """Return independent copies of the settings used for the verified runs."""
    values = copy.deepcopy(TASKS)
    for target, config in values.items():
        if config['target'] != target or config['primary_selection'] != PRIMARY_SELECTION:
            raise ValueError('Invalid Table 1 configuration: ' + target)
        if config['checkpoint_policy'] != 'fixed_final_epoch':
            raise ValueError('Table 1 uses predetermined final checkpoints only.')
    return values
