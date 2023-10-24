import torch
import torch.nn as nn
import numpy as np
from itertools import product
from torch.nn.utils.parametrize import register_parametrization as rpm


def state_mask(n_agents, n_0, n_1, gru=False):
    # Mask for the state to state connections between layers
    mask = torch.eye(n_agents)
    mask = mask.repeat_interleave(n_0, 0).repeat_interleave(n_1, 1)
    if gru:
        mask = torch.concat([m for m in mask.unsqueeze(0).repeat_interleave(3, 0)])
    return mask


def sparse_mask(sparsity, n_in, n_out):
    if sparsity >= 0:
        assert sparsity <= 1
        nb_non_zero = int(sparsity * n_in * n_out)
    else:
        nb_non_zero = -sparsity
    w_mask = np.zeros((n_in, n_out), dtype=bool)
    # ind_in = rd.choice(np.arange(in_features),size=self.nb_non_zero)
    # ind_out = rd.choice(np.arange(out_features),size=self.nb_non_zero)

    ind_in, ind_out = np.unravel_index(
        np.random.choice(np.arange(n_in * n_out), nb_non_zero, replace=False),
        (n_in, n_out),
    )
    w_mask[ind_in, ind_out] = True
    w_mask = torch.tensor(w_mask)

    return w_mask


def comms_mask(sparsity, n_agents, n_hidden, gru=False):
    comms_mask = torch.zeros((n_agents * n_hidden, n_agents * n_hidden))
    rec_mask = torch.zeros((n_agents * n_hidden, n_agents * n_hidden))

    for i, j in product(range(n_agents), repeat=2):
        if i != j:
            comms_mask[
                i * n_hidden : (i + 1) * n_hidden, j * n_hidden : (j + 1) * n_hidden
            ] = sparse_mask(sparsity, n_hidden, n_hidden)
        else:
            rec_mask[
                i * n_hidden : (i + 1) * n_hidden, j * n_hidden : (j + 1) * n_hidden
            ] = 1 - torch.eye(n_hidden)

    masks = [comms_mask, rec_mask]
    if gru:
        masks = [
            torch.concat([m for m in mask.unsqueeze(0).repeat_interleave(3, 0)])
            for mask in masks
        ]

    return masks


cell_types_dict = {str(t): t for t in [nn.RNN, nn.GRU]}


class Masked_weight(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, W):
        return W * self.mask


def reccursive_readout(input, readout, common_readout, output_size):
    if isinstance(readout, nn.ModuleList):
        out = [
            reccursive_readout(input, r, common_readout, size)
            for r, size in zip(readout, output_size)
        ]
    else:
        out = process_readout(input, readout, common_readout, output_size)

    return out


def process_readout(input, readout, common_readout, output_size):
    output = readout(input)
    if not common_readout:
        output = torch.stack(output.split(output_size, -1), 1)
    return output


def reccursive_rpm(model, mask):
    if isinstance(model, nn.ModuleList):
        [reccursive_rpm(m, mask) for m in model]
    elif isinstance(model, nn.Sequential):
        rpm(model[0], "weight", Masked_weight(mask))
    else:
        rpm(model, "weight", Masked_weight(mask))
    return model


class Community(nn.Module):
    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()

        (
            self.input_config,
            self.agents_config,
            self.connections_config,
            self.readout_config,
        ) = [config[k] for k in ["input", "agents", "connections", "readout"]]

        self.is_community = True
        self.input_size, self.common_input = [
            self.input_config[k] for k in ["input_size", "common_input"]
        ]
        self.n_agents, self.hidden_size, self.n_layers, self.dropout, self.cell_type = [
            self.agents_config[k]
            for k in ["n_agents", "hidden_size", "n_layers", "dropout", "cell_type"]
        ]
        self.sparsity = self.connections_config["sparsity"]
        self.output_size, self.common_readout = [
            self.readout_config[k] for k in ["output_size", "common_readout"]
        ]

        gru = "GRU" in self.cell_type
        rec_masks = comms_mask(self.sparsity, self.n_agents, self.hidden_size, gru=gru)

        self.masks = {
            "input_mask": state_mask(
                self.n_agents, self.hidden_size, self.input_size, gru=gru
            )
            if not self.common_input
            else torch.ones_like(
                state_mask(self.n_agents, self.hidden_size, self.input_size, gru=gru)
            ),
            "state_mask": state_mask(
                self.n_agents, self.hidden_size, self.hidden_size, gru=gru
            ),
            "rec_mask": rec_masks[1],
            "comms_mask": rec_masks[0],
            "output_mask": state_mask(self.n_agents, self.output_size, self.hidden_size)
            if not isinstance(self.output_size, list)
            else torch.stack(
                [
                    state_mask(self.n_agents, o, self.hidden_size)
                    for o in self.output_size
                ]
            ),
        }

        self.core = cell_types_dict[self.cell_type](
            input_size=self.input_size * self.n_agents,
            hidden_size=self.hidden_size * self.n_agents,
            num_layers=self.n_layers,
            batch_first=False,
            bias=False,
            dropout=self.dropout,
        )

        for n, m in self.masks.items():
            self.register_buffer(n, m)

        if self.common_readout:
            self.readout = (
                nn.Linear(self.n_agents * self.hidden_size, self.output_size)
                if not self.multi_readout
                else nn.ModuleList(
                    [
                        nn.Linear(self.n_agents * self.hidden_size, o)
                        for o in self.output_size
                    ]
                )
            )
        else:
            self.readout = (
                nn.Linear(
                    self.n_agents * self.hidden_size,
                    self.output_size * self.n_agents,
                    bias=False,
                )
                if not self.multi_readout
                else nn.ModuleList(
                    [
                        nn.Linear(
                            self.n_agents * self.hidden_size,
                            o * self.n_agents,
                            bias=False,
                        )
                        for o in self.output_size
                    ]
                )
            )

        for n in dict(self.core.named_parameters()).copy().keys():
            if "weight_hh" in n:
                if n[-1] == str(self.n_layers - 1):
                    rpm(self.core, n, Masked_weight(self.comms_mask + self.rec_mask))
                else:
                    rpm(self.core, n, Masked_weight(self.rec_mask))
            elif "weight_ih" in n and n[-1] != "0":
                rpm(self.core, n, Masked_weight(self.state_mask))
            elif "weight_ih" in n and n[-1] == "0":
                rpm(self.core, n, Masked_weight(self.input_mask))

        reccursive_rpm(self.readout, self.output_mask)

    @property
    def multi_readout(self):
        if hasattr(self, "readout"):
            return isinstance(self.readout, nn.ModuleList)
        else:
            return isinstance(self.output_size, list)

    def forward(self, input):
        output, states = self.core(input)

        output = reccursive_readout(
            output,
            self.readout,
            self.common_readout,
            self.output_size,
        )
        return output, states
