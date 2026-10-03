"""Deep Ensemble model used for the Layer-2 capacity-utilization surrogates.

The implementation follows the manuscript description:
- five independently initialized MLP members by default;
- hidden layers 256-256-128;
- a mean head and a Softplus variance head;
- heteroscedastic Gaussian negative log-likelihood;
- Adam optimizer with cosine-annealing learning-rate schedule;
- input/output standardization fitted on the training subset only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence
import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


OUTPUT_NAMES = ("eta_head", "eta_mid_max")


@dataclass
class DEConfig:
    n_models: int = 5
    hidden_dims: tuple[int, ...] = (256, 256, 128)
    dropout_min: float = 0.05
    dropout_max: float = 0.15
    lr: float = 1e-3
    epochs: int = 300
    batch_size: int = 64
    weight_decay: float = 1e-3
    patience: int = 60
    random_seed: int = 42
    variance_floor: float = 1e-6
    device: str = "auto"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def read_numeric_csv(path: str | Path, min_cols: Optional[int] = None) -> pd.DataFrame:
    """Read a numeric CSV with or without a header row."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    raw = pd.read_csv(path, header=None)
    num = raw.apply(pd.to_numeric, errors="coerce")

    if len(num) >= 2 and num.iloc[0].isna().any() and num.iloc[1:].notna().all().all():
        num = num.iloc[1:].reset_index(drop=True)

    if num.isna().any().any():
        r, c = np.argwhere(num.isna().to_numpy())[0]
        raise ValueError(
            f"{path} contains a non-numeric or missing value at row {r + 1}, col {c + 1}."
        )
    if min_cols is not None and num.shape[1] < min_cols:
        raise ValueError(f"{path} must contain at least {min_cols} columns.")
    return num.astype(np.float64)


def load_xy(x_path: str | Path, y_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    X = read_numeric_csv(x_path).to_numpy(dtype=np.float64)
    Y = read_numeric_csv(y_path, min_cols=2).iloc[:, :2].to_numpy(dtype=np.float64)
    if len(X) != len(Y):
        raise ValueError(f"X/Y sample count mismatch: X={len(X)}, Y={len(Y)}")
    return X, Y


class MLPMember(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Sequence[int] = (256, 256, 128),
        dropout: float = 0.10,
        variance_floor: float = 1e-6,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev, out_dim)
        self.var_head = nn.Linear(prev, out_dim)
        self.softplus = nn.Softplus()
        self.variance_floor = variance_floor

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        mean = self.mean_head(feat)
        variance = self.softplus(self.var_head(feat)) + self.variance_floor
        return mean, variance


def gaussian_nll(mean: torch.Tensor, variance: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Heteroscedastic Gaussian negative log-likelihood (constant omitted)."""
    return 0.5 * (torch.log(variance) + (target - mean) ** 2 / variance).mean()


class DeepEnsemble:
    def __init__(self, cfg: DEConfig) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.members: list[MLPMember] = []
        self.member_dropouts: list[float] = []
        self.training_histories: list[pd.DataFrame] = []
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        self.in_dim: Optional[int] = None
        self.out_dim: Optional[int] = None

    def _make_loader(
        self, X: np.ndarray, Y: np.ndarray, seed: int, shuffle: bool
    ) -> DataLoader:
        generator = torch.Generator().manual_seed(seed)
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.float32),
        )
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            generator=generator if shuffle else None,
        )

    @staticmethod
    def _evaluate_nll(model: MLPMember, loader: DataLoader, device: str) -> float:
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                mu, var = model(xb)
                loss = gaussian_nll(mu, var, yb)
                total += float(loss.item()) * len(xb)
                count += len(xb)
        return total / max(count, 1)

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
    ) -> None:
        set_seed(self.cfg.random_seed)
        X_train = np.asarray(X_train, dtype=np.float64)
        Y_train = np.asarray(Y_train, dtype=np.float64)
        if X_train.ndim != 2 or Y_train.ndim != 2:
            raise ValueError("X_train and Y_train must be 2-D arrays.")
        if len(X_train) != len(Y_train):
            raise ValueError("Training X/Y sample counts do not match.")

        self.in_dim = X_train.shape[1]
        self.out_dim = Y_train.shape[1]
        Xtr = self.scaler_X.fit_transform(X_train)
        Ytr = self.scaler_Y.fit_transform(Y_train)

        use_val = X_val is not None and Y_val is not None
        if use_val:
            X_val = np.asarray(X_val, dtype=np.float64)
            Y_val = np.asarray(Y_val, dtype=np.float64)
            if len(X_val) != len(Y_val):
                raise ValueError("Validation X/Y sample counts do not match.")
            Xva = self.scaler_X.transform(X_val)
            Yva = self.scaler_Y.transform(Y_val)
        else:
            Xva = Yva = None

        self.members.clear()
        self.member_dropouts.clear()
        self.training_histories.clear()

        for m in range(self.cfg.n_models):
            seed = self.cfg.random_seed + 1000 * m
            set_seed(seed)
            rng = np.random.default_rng(seed)
            dropout = float(rng.uniform(self.cfg.dropout_min, self.cfg.dropout_max))
            model = MLPMember(
                self.in_dim,
                self.out_dim,
                hidden_dims=self.cfg.hidden_dims,
                dropout=dropout,
                variance_floor=self.cfg.variance_floor,
            ).to(self.device)

            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.cfg.epochs
            )
            train_loader = self._make_loader(Xtr, Ytr, seed, shuffle=True)
            val_loader = (
                self._make_loader(Xva, Yva, seed, shuffle=False) if use_val else None
            )

            best_state = None
            best_val = np.inf
            stale = 0
            rows = []

            for epoch in range(1, self.cfg.epochs + 1):
                model.train()
                total = 0.0
                count = 0
                for xb, yb in train_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    optimizer.zero_grad()
                    mu, var = model(xb)
                    loss = gaussian_nll(mu, var, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total += float(loss.item()) * len(xb)
                    count += len(xb)
                train_nll = total / max(count, 1)
                val_nll = (
                    self._evaluate_nll(model, val_loader, self.device)
                    if val_loader is not None
                    else np.nan
                )
                rows.append(
                    {
                        "member": m + 1,
                        "seed": seed,
                        "dropout": dropout,
                        "epoch": epoch,
                        "train_nll": train_nll,
                        "val_nll": val_nll,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )

                if val_loader is not None:
                    if val_nll < best_val - 1e-8:
                        best_val = val_nll
                        best_state = copy.deepcopy(model.state_dict())
                        stale = 0
                    else:
                        stale += 1
                    if stale >= self.cfg.patience:
                        break
                scheduler.step()

            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            self.members.append(model)
            self.member_dropouts.append(dropout)
            self.training_histories.append(pd.DataFrame(rows))

    def predict(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.members:
            raise RuntimeError("The Deep Ensemble has not been trained or loaded.")
        X = np.asarray(X, dtype=np.float64)
        Xsc = self.scaler_X.transform(X)
        Xt = torch.tensor(Xsc, dtype=torch.float32, device=self.device)

        all_means = []
        all_vars = []
        with torch.no_grad():
            for model in self.members:
                model = model.to(self.device)
                model.eval()
                mu_sc, var_sc = model(Xt)
                all_means.append(mu_sc.cpu().numpy())
                all_vars.append(var_sc.cpu().numpy())

        all_means = np.asarray(all_means)
        all_vars = np.asarray(all_vars)
        mean_sc = all_means.mean(axis=0)
        var_ep_sc = all_means.var(axis=0)
        var_al_sc = all_vars.mean(axis=0)
        var_total_sc = var_ep_sc + var_al_sc

        mean = self.scaler_Y.inverse_transform(mean_sc)
        scale = self.scaler_Y.scale_
        std_ep = np.sqrt(np.maximum(var_ep_sc, 0.0)) * scale
        std_al = np.sqrt(np.maximum(var_al_sc, 0.0)) * scale
        std_total = np.sqrt(np.maximum(var_total_sc, 0.0)) * scale
        return mean, std_total, std_ep, std_al

    def predict_member_means(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Xsc = self.scaler_X.transform(X)
        Xt = torch.tensor(Xsc, dtype=torch.float32, device=self.device)
        values = []
        with torch.no_grad():
            for model in self.members:
                mu_sc, _ = model.to(self.device)(Xt)
                values.append(self.scaler_Y.inverse_transform(mu_sc.cpu().numpy()))
        return np.asarray(values)

    def save(self, path: str | Path) -> None:
        if self.in_dim is None or self.out_dim is None:
            raise RuntimeError("Cannot save an uninitialized Deep Ensemble.")
        payload = {
            "format_version": 1,
            "config": asdict(self.cfg),
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "member_dropouts": self.member_dropouts,
            "member_state_dicts": [m.cpu().state_dict() for m in self.members],
            "scaler_X": {
                "mean": self.scaler_X.mean_,
                "scale": self.scaler_X.scale_,
                "var": self.scaler_X.var_,
                "n_features_in": self.scaler_X.n_features_in_,
            },
            "scaler_Y": {
                "mean": self.scaler_Y.mean_,
                "scale": self.scaler_Y.scale_,
                "var": self.scaler_Y.var_,
                "n_features_in": self.scaler_Y.n_features_in_,
            },
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        self.device = resolve_device(self.cfg.device)
        for member in self.members:
            member.to(self.device)

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> "DeepEnsemble":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")

        cfg_dict = dict(payload["config"])
        cfg_dict["hidden_dims"] = tuple(cfg_dict["hidden_dims"])
        cfg_dict["device"] = device
        cfg = DEConfig(**cfg_dict)
        obj = cls(cfg)
        obj.in_dim = int(payload["in_dim"])
        obj.out_dim = int(payload["out_dim"])
        obj.member_dropouts = [float(x) for x in payload["member_dropouts"]]

        obj.members = []
        for dropout, state in zip(obj.member_dropouts, payload["member_state_dicts"]):
            member = MLPMember(
                obj.in_dim,
                obj.out_dim,
                hidden_dims=cfg.hidden_dims,
                dropout=dropout,
                variance_floor=cfg.variance_floor,
            )
            member.load_state_dict(state)
            member.to(obj.device)
            member.eval()
            obj.members.append(member)

        for scaler, key in [(obj.scaler_X, "scaler_X"), (obj.scaler_Y, "scaler_Y")]:
            state = payload[key]
            scaler.mean_ = np.asarray(state["mean"])
            scaler.scale_ = np.asarray(state["scale"])
            scaler.var_ = np.asarray(state["var"])
            scaler.n_features_in_ = int(state["n_features_in"])
            scaler.n_samples_seen_ = 1
        return obj
