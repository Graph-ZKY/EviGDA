import torch


class ParameterAverage:
    def __init__(self):
        self.state = None
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        current = model.state_dict()
        self.updates += 1
        if self.state is None:
            self.state = {key: value.detach().clone() for key, value in current.items()}
            return
        for key, value in current.items():
            if value.is_floating_point():
                self.state[key].add_((value - self.state[key]) / self.updates)
            else:
                self.state[key].copy_(value)

    @torch.no_grad()
    def apply(self, model):
        if self.state is None:
            raise ValueError('No predetermined averaging step was reached.')
        model.load_state_dict(self.state, strict=True)

