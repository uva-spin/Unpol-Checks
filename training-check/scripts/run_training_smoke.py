#!/usr/bin/env python3
"""Run real optimization steps on the archived 329-point fit input.

This is a smoke test: it verifies the neural model, cached W kernel, data,
nuisance normalizations, backpropagation, and optimizer work together.  It is
not the 96-start production ensemble.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


class FiLMBlock(nn.Module):
    def __init__(self, width: int, cond_width: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(width, width)
        self.lin2 = nn.Linear(width, width)
        self.to_gamma = nn.Linear(cond_width, width)
        self.to_beta = nn.Linear(cond_width, width)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        u = torch.tanh(self.lin1(h))
        gamma = F.softplus(self.to_gamma(c)).unsqueeze(1) + 1e-6
        beta = torch.tanh(self.to_beta(c)).unsqueeze(1)
        return torch.tanh(self.lin2(gamma * u + beta) + h)


class NonperturbativeFactor(nn.Module):
    """The archived monotone FiLM nonperturbative factor."""
    def __init__(self) -> None:
        super().__init__()
        self.radial = nn.Linear(4, 48)
        self.cond = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU())
        self.blocks = nn.ModuleList([FiLMBlock(48, 32) for _ in range(3)])
        self.head = nn.Linear(48, 1)

    @staticmethod
    def _features(x: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b2 = b.unsqueeze(0).expand(x.numel(), -1)
        radial = torch.stack([b2, b2.square(), torch.sqrt(b2 + 1e-8), torch.log1p(b2)], dim=-1)
        x = torch.clamp(x, 1e-6, 1.0 - 1e-6)
        condition = torch.stack([x, torch.log(x / (1.0 - x))], dim=-1)
        return radial, condition

    @staticmethod
    def _smooth(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        sigma = 0.45
        db = b[1:] - b[:-1]
        weights = torch.cat((0.5 * db[:1], 0.5 * (b[2:] - b[:-2]), 0.5 * db[-1:]))
        delta = b[:, None] - b[None, :]
        kernel = torch.exp(-0.5 * (delta / sigma).square()) * weights[None, :]
        return a @ (kernel / kernel.sum(dim=1, keepdim=True)).T

    def forward(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        radial, condition = self._features(x, b)
        h = torch.tanh(self.radial(radial))
        c = self.cond(condition)
        for block in self.blocks:
            h = block(h, c)
        a = self._smooth(F.softplus(self.head(h).squeeze(-1)), b)
        a = a + 0.08 * torch.sigmoid((b - 3.5) / 0.25)[None, :]
        integrand = 2.0 * b[None, :] * a
        area = 0.5 * (integrand[:, 1:] + integrand[:, :-1]) * (b[1:] - b[:-1])[None, :]
        exponent = -torch.cat((torch.zeros_like(area[:, :1]), torch.cumsum(area, dim=1)), dim=1)
        return torch.exp(torch.clamp(exponent, -40.0, 40.0))


def load_kernel(rows: pd.DataFrame, dtype: torch.dtype) -> tuple[np.ndarray, torch.Tensor]:
    tab = pd.read_csv(DATA / "w_kernel.csv")
    ids = rows.row_id.astype(str).tolist()
    tab = tab[tab.row_id.astype(str).isin(ids)]
    b = np.sort(tab.bT.unique())
    matrix = tab.pivot(index="row_id", columns="bT", values="Wpert_CS").reindex(ids).to_numpy()
    if matrix.shape != (len(rows), len(b)) or not np.isfinite(matrix).all():
        raise ValueError("cached W kernel is incomplete for the fit rows")
    weights = np.empty_like(b)
    weights[0], weights[-1] = 0.5 * (b[1] - b[0]), 0.5 * (b[-1] - b[-2])
    weights[1:-1] = 0.5 * (b[2:] - b[:-2])
    argument = rows.qT.to_numpy()[:, None] * b[None, :]
    j0 = torch.special.bessel_j0(torch.tensor(argument, dtype=dtype)).numpy()
    return b, torch.tensor(weights[None, :] * b[None, :] * j0 * matrix, dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=2, help="optimizer updates (default: 2)")
    parser.add_argument("--start", choices=("frozen", "fresh"), default="frozen", help="checkpoint compatibility or fresh-start check")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "training-smoke.json")
    args = parser.parse_args()
    if args.updates < 1:
        raise ValueError("--updates must be positive")
    torch.manual_seed(303)
    dtype = torch.float32
    rows = pd.read_csv(DATA / "fit_rows.csv")
    required = {"row_id", "qT", "QM", "x1", "x2", "dataset", "target_used", "sigma_used", "norm_rel_used"}
    missing = required.difference(rows.columns)
    if missing or len(rows) != 329:
        raise ValueError(f"fit rows do not have the expected 329-row schema; missing={sorted(missing)}")
    b, kernel = load_kernel(rows, dtype)
    model = NonperturbativeFactor().to(dtype)
    if args.start == "frozen":
        state = torch.load(DATA / "initial_model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict({key.removeprefix("np_factor."): value for key, value in state.items() if key.startswith("np_factor.")})
    else:
        nn.init.zeros_(model.head.weight)
        nn.init.constant_(model.head.bias, math.log(math.expm1(0.05)))
    datasets = list(dict.fromkeys(rows.dataset.astype(str)))
    index = torch.tensor(rows.dataset.astype(str).map({name: i for i, name in enumerate(datasets)}).to_numpy(), dtype=torch.long)
    initial = pd.read_csv(DATA / "initial_normalizations.csv").set_index("dataset").norm_scale
    scale_values = [initial[name] for name in datasets] if args.start == "frozen" else [1.0] * len(datasets)
    scales = nn.Parameter(torch.tensor(scale_values, dtype=dtype))
    x1, x2 = (torch.tensor(rows[name].to_numpy(), dtype=dtype) for name in ("x1", "x2"))
    target, sigma = (torch.tensor(rows[name].to_numpy(), dtype=dtype) for name in ("target_used", "sigma_used"))
    norm_rel = torch.tensor([rows.loc[rows.dataset == name, "norm_rel_used"].iloc[0] for name in datasets], dtype=dtype)
    optimizer = torch.optim.Adam(list(model.parameters()) + [scales], lr=2e-5)

    def objective() -> torch.Tensor:
        b_tensor = torch.tensor(b, dtype=dtype)
        prediction = (kernel * model(x1, b_tensor) * model(x2, b_tensor)).sum(dim=1) * scales[index]
        chi2 = ((prediction - target) / sigma).square().mean()
        constrained = norm_rel > 0
        nuisance = ((scales[constrained] - 1.0) / norm_rel[constrained]).square().sum() / len(rows)
        return chi2 + nuisance

    before = float(objective().detach())
    for _ in range(args.updates):
        optimizer.zero_grad()
        loss = objective()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [scales], 10.0)
        optimizer.step()
    after = float(objective().detach())
    if not (math.isfinite(before) and math.isfinite(after)):
        raise RuntimeError("training produced a non-finite objective")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"rows": len(rows), "start": args.start, "updates": args.updates, "objective_before": before, "objective_after": after, "datasets": datasets}, indent=2) + "\n")
    print(f"PASS: {args.start} start, {len(rows)} rows, {args.updates} updates, objective {before:.8f} -> {after:.8f}")


if __name__ == "__main__":
    main()
