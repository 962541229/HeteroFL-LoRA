# coding: UTF-8


import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
try:
    from transformers import AutoModel
except Exception:  # pragma: no cover
    AutoModel = None

try:
    from SVFLinear import SVFLinear
except Exception:  # pragma: no cover
    SVFLinear = None


@dataclass
class PublicSubspace:
    """One server-side public subspace for one semantic module/layer group."""

    key: str
    module_type: str
    layer_id: int
    U: torch.Tensor              # [out_dim_p, r_p], CPU float tensor
    V: torch.Tensor              # [in_dim_p, r_p], CPU float tensor
    delta_sigma: torch.Tensor    # [r_p, r_p], CPU float tensor
    source_name: str = ""

    @property
    def r(self) -> int:
        return int(self.delta_sigma.size(0))

    @property
    def out_features(self) -> int:
        return int(self.U.size(0))

    @property
    def in_features(self) -> int:
        return int(self.V.size(0))


def _safe_int(x: Optional[str], default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def canonical_module_type(module_name: str) -> Optional[str]:

    name = module_name.lower()

    if name.endswith("attention.self.query") or name.endswith("attention.q_lin") or name.endswith("q_lin"):
        return "query"

    if name.endswith("attention.self.value") or name.endswith("attention.v_lin") or name.endswith("v_lin"):
        return "value"

    if name.endswith("intermediate.dense") or name.endswith("ffn.lin1") or name.endswith("lin1"):
        return "intermediate"

    if name.endswith("attention.output.dense") or name.endswith("attention.out_lin") or name.endswith("out_lin"):
        return None

    if name.endswith("output.dense") or name.endswith("ffn.lin2") or name.endswith("lin2"):
        return "output"

    return None


def extract_layer_id(module_name: str) -> int:
    """Extract layer id from common Transformer module names."""

    patterns = [
        r"encoder\.layer\.(\d+)",
        r"transformer\.layer\.(\d+)",
        r"layers\.(\d+)",
        r"layer\.(\d+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, module_name)
        if m is not None:
            return _safe_int(m.group(1), default=-1)
    return -1


def make_public_key(layer_id: int, module_type: str) -> str:
    return f"layer_{layer_id:02d}.{module_type}"


def truncated_svd_bases(weight: torch.Tensor, r: int) -> Tuple[torch.Tensor, torch.Tensor]:

    with torch.no_grad():
        W = weight.detach().float().cpu()
        max_rank = min(W.size(0), W.size(1), int(r))
        if max_rank <= 0:
            raise ValueError(f"Invalid rank {r} for weight shape {tuple(W.shape)}")
        U, _, Vh = torch.linalg.svd(W, full_matrices=False)
        U_r = U[:, :max_rank].contiguous()
        V_r = Vh[:max_rank, :].T.contiguous()
    return U_r, V_r


def _fix_qr_sign(Q: torch.Tensor) -> torch.Tensor:
    """Make QR output deterministic up to signs."""

    if Q.numel() == 0:
        return Q
    # For each column, make the element with largest magnitude positive.
    idx = torch.argmax(torch.abs(Q), dim=0)
    signs = torch.sign(Q[idx, torch.arange(Q.size(1))])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return Q * signs.view(1, -1)


def orthonormalize_columns(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return a matrix with approximately orthonormal columns."""

    X = X.detach().float().cpu()
    rows, cols = X.shape
    if rows < cols:
        raise ValueError(f"Cannot orthonormalize columns: rows {rows} < cols {cols}")
    if X.norm().item() < eps:
        eye = torch.eye(rows, cols, dtype=torch.float32)
        return eye.contiguous()
    Q, _ = torch.linalg.qr(X, mode="reduced")
    Q = _fix_qr_sign(Q)
    return Q[:, :cols].contiguous()


def resize_basis_rows(
    basis: torch.Tensor,
    target_rows: int,
    mode: str = "row_resize_qr",
) -> torch.Tensor:

    basis = basis.detach().float().cpu()
    if basis.dim() != 2:
        raise ValueError(f"basis must be 2-D, got {basis.dim()}-D")

    source_rows, rank = basis.shape
    target_rows = int(target_rows)
    if target_rows <= 0:
        raise ValueError(f"target_rows must be positive, got {target_rows}")
    if target_rows < rank:
        raise ValueError(
            f"target_rows={target_rows} is smaller than rank={rank}; choose a smaller svf_r."
        )

    if source_rows == target_rows:
        # Keep the exact original basis for equal-dimensional projection.
        return basis.contiguous()

    if mode == "identity_or_fail":
        raise ValueError(f"Ambient dimension mismatch: {source_rows} -> {target_rows}")
    if mode not in {"row_resize", "row_resize_qr"}:
        raise ValueError(f"Unknown cross-dimensional bridge mode: {mode}")

    resized = torch.zeros(target_rows, rank, dtype=torch.float32)
    overlap = min(source_rows, target_rows)
    resized[:overlap, :] = basis[:overlap, :]

    # Preserve expected energy after truncation/padding. This is mild and helps
    # avoid very small cross-Gram scores when 1024 -> 768 or 4096 -> 3072.
    if overlap > 0:
        resized = resized * math.sqrt(float(max(source_rows, target_rows)) / float(overlap))

    if mode == "row_resize_qr":
        resized = orthonormalize_columns(resized)

    return resized.contiguous()


def cross_basis_gram(
    source_basis: torch.Tensor,
    target_basis: torch.Tensor,
    bridge_mode: str = "row_resize_qr",
) -> torch.Tensor:

    source_in_target = resize_basis_rows(source_basis, target_basis.size(0), mode=bridge_mode)
    return target_basis.detach().float().cpu().T @ source_in_target


def subspace_similarity(
    U_a: torch.Tensor,
    U_b: torch.Tensor,
    bridge_mode: str = "row_resize_qr",
) -> float:

    U_a = U_a.detach().float().cpu()
    U_b = U_b.detach().float().cpu()
    if U_a.dim() != 2 or U_b.dim() != 2:
        return float("-inf")
    try:
        G = cross_basis_gram(U_a, U_b, bridge_mode=bridge_mode)
        denom = max(1, min(int(U_a.size(1)), int(U_b.size(1))))
        score = torch.norm(G, p="fro").pow(2) / denom
        if torch.isnan(score) or torch.isinf(score):
            return float("-inf")
        return float(score.item())
    except Exception:
        return float("-inf")


def combined_subspace_similarity(
    U_c: torch.Tensor,
    V_c: torch.Tensor,
    U_p: torch.Tensor,
    V_p: torch.Tensor,
    bridge_mode: str = "row_resize_qr",
) -> float:
    """Average the left-basis and right-basis similarities."""

    s_u = subspace_similarity(U_c, U_p, bridge_mode=bridge_mode)
    s_v = subspace_similarity(V_c, V_p, bridge_mode=bridge_mode)
    if s_u == float("-inf") or s_v == float("-inf"):
        return float("-inf")
    return 0.5 * (s_u + s_v)


def _check_delta_shape(delta_sigma: torch.Tensor, rank: int, name: str) -> None:
    if tuple(delta_sigma.shape) != (rank, rank):
        raise ValueError(f"{name} shape {tuple(delta_sigma.shape)} does not match rank {rank}")


def project_client_to_public(
    U_c: torch.Tensor,
    V_c: torch.Tensor,
    delta_sigma_c: torch.Tensor,
    U_p: torch.Tensor,
    V_p: torch.Tensor,
    bridge_mode: str = "row_resize_qr",
) -> torch.Tensor:

    U_c = U_c.detach().float().cpu()
    V_c = V_c.detach().float().cpu()
    delta_sigma_c = delta_sigma_c.detach().float().cpu()
    U_p = U_p.detach().float().cpu()
    V_p = V_p.detach().float().cpu()

    _check_delta_shape(delta_sigma_c, int(U_c.size(1)), "delta_sigma_c")
    if U_c.size(1) != V_c.size(1):
        raise ValueError(f"Client U/V rank mismatch: {U_c.size(1)} vs {V_c.size(1)}")
    if U_p.size(1) != V_p.size(1):
        raise ValueError(f"Public U/V rank mismatch: {U_p.size(1)} vs {V_p.size(1)}")

    left = cross_basis_gram(U_c, U_p, bridge_mode=bridge_mode)       # [r_p, r_c]
    right = cross_basis_gram(V_c, V_p, bridge_mode=bridge_mode)      # [r_p, r_c]
    return (left @ delta_sigma_c @ right.T).contiguous()             # [r_p, r_p]


def project_public_to_client(
    U_p: torch.Tensor,
    V_p: torch.Tensor,
    delta_sigma_p: torch.Tensor,
    U_c: torch.Tensor,
    V_c: torch.Tensor,
    bridge_mode: str = "row_resize_qr",
) -> torch.Tensor:

    U_p = U_p.detach().float().cpu()
    V_p = V_p.detach().float().cpu()
    delta_sigma_p = delta_sigma_p.detach().float().cpu()
    U_c = U_c.detach().float().cpu()
    V_c = V_c.detach().float().cpu()

    _check_delta_shape(delta_sigma_p, int(U_p.size(1)), "delta_sigma_p")
    if U_c.size(1) != V_c.size(1):
        raise ValueError(f"Client U/V rank mismatch: {U_c.size(1)} vs {V_c.size(1)}")
    if U_p.size(1) != V_p.size(1):
        raise ValueError(f"Public U/V rank mismatch: {U_p.size(1)} vs {V_p.size(1)}")

    left = cross_basis_gram(U_p, U_c, bridge_mode=bridge_mode)       # [r_c, r_p]
    right = cross_basis_gram(V_p, V_c, bridge_mode=bridge_mode)      # [r_c, r_p]
    return (left @ delta_sigma_p @ right.T).contiguous()             # [r_c, r_c]


def aggregate_delta_sigmas(
    delta_sigma_list: Iterable[torch.Tensor],
    weights: Optional[Iterable[float]] = None,
) -> torch.Tensor:
    """Weighted average of projected public-space updates."""

    delta_sigma_list = [x.detach().float().cpu() for x in delta_sigma_list]
    if len(delta_sigma_list) == 0:
        raise ValueError("Cannot aggregate an empty delta_sigma_list.")

    shapes = {tuple(x.shape) for x in delta_sigma_list}
    if len(shapes) != 1:
        raise ValueError(f"Cannot aggregate tensors with different shapes: {shapes}")

    if weights is None:
        stacked = torch.stack(delta_sigma_list, dim=0)
        return stacked.mean(dim=0).contiguous()

    weights = list(weights)
    if len(weights) != len(delta_sigma_list):
        raise ValueError("weights length must match delta_sigma_list length.")
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.sum().clamp_min(1e-12)
    stacked = torch.stack(delta_sigma_list, dim=0)
    return torch.sum(stacked * w.view(-1, 1, 1), dim=0).contiguous()


def refine_public_basis(
    U_p: torch.Tensor,
    V_p: torch.Tensor,
    delta_sigma_p: torch.Tensor,
    transform_delta_to_new_basis: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    U_p = U_p.detach().float().cpu()
    V_p = V_p.detach().float().cpu()
    delta_sigma_p = delta_sigma_p.detach().float().cpu()

    if delta_sigma_p.norm().item() < 1e-12:
        return U_p.contiguous(), V_p.contiguous(), delta_sigma_p.contiguous()

    Q, _, _ = torch.linalg.svd(delta_sigma_p, full_matrices=False)
    U_new = (U_p @ Q).contiguous()
    V_new = (V_p @ Q).contiguous()

    if transform_delta_to_new_basis:
        delta_new = (Q.T @ delta_sigma_p @ Q).contiguous()
    else:
        delta_new = delta_sigma_p.contiguous()

    return U_new, V_new, delta_new


class PublicSpaceManager:

    def __init__(
        self,
        spaces: Dict[str, PublicSubspace],
        rank: int,
        bridge_mode: str = "row_resize_qr",
    ):
        self.spaces = spaces
        self.rank = int(rank)
        self.bridge_mode = bridge_mode
        self.type_to_keys: Dict[str, List[str]] = {}
        for key, space in spaces.items():
            self.type_to_keys.setdefault(space.module_type, []).append(key)
        for module_type in self.type_to_keys:
            self.type_to_keys[module_type] = sorted(
                self.type_to_keys[module_type],
                key=lambda k: self.spaces[k].layer_id,
            )

    @classmethod
    def from_public_model(
        cls,
        public_model_path: str,
        rank: int,
        device: str = "cpu",
        verbose: bool = True,
        bridge_mode: str = "row_resize_qr",
    ) -> "PublicSpaceManager":
        """Initialize public spaces from a public base LFM, e.g. bert-base-uncased."""

        if AutoModel is None:
            raise RuntimeError("transformers.AutoModel is unavailable. Please install transformers in your training environment.")
        public_model = AutoModel.from_pretrained(public_model_path)
        public_model.to(device)
        public_model.eval()

        spaces: Dict[str, PublicSubspace] = {}

        for name, module in public_model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            module_type = canonical_module_type(name)
            if module_type is None:
                continue
            layer_id = extract_layer_id(name)
            if layer_id < 0:
                continue

            U, V = truncated_svd_bases(module.weight, rank)
            r_eff = U.size(1)
            key = make_public_key(layer_id, module_type)
            spaces[key] = PublicSubspace(
                key=key,
                module_type=module_type,
                layer_id=layer_id,
                U=U,
                V=V,
                delta_sigma=torch.zeros(r_eff, r_eff, dtype=torch.float32),
                source_name=name,
            )

        del public_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if len(spaces) == 0:
            raise RuntimeError(
                f"No public subspaces were initialized from {public_model_path}. "
                "Please check public_model_path and canonical_module_type()."
            )

        if verbose:
            print(f"[PublicSpace] Initialized {len(spaces)} public subspaces from {public_model_path}")
            counts = {}
            for s in spaces.values():
                counts[s.module_type] = counts.get(s.module_type, 0) + 1
            print(f"[PublicSpace] Type counts: {counts}")
            print(f"[PublicSpace] Cross-dimensional bridge mode: {bridge_mode}")

        return cls(spaces=spaces, rank=rank, bridge_mode=bridge_mode)

    def state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            key: {
                "U": space.U.clone(),
                "V": space.V.clone(),
                "delta_sigma": space.delta_sigma.clone(),
                "layer_id": torch.tensor(space.layer_id),
            }
            for key, space in self.spaces.items()
        }

    def iter_client_svf_modules(self, client_model: nn.Module):
        if SVFLinear is None:
            raise RuntimeError("SVFLinear could not be imported.")
        for name, module in client_model.named_modules():
            if isinstance(module, SVFLinear):
                yield name, module

    def match_client_model(self, client_model: nn.Module, verbose: bool = False) -> Dict[str, str]:

        match_map: Dict[str, str] = {}

        for name, module in self.iter_client_svf_modules(client_model):
            module_type = canonical_module_type(name)
            if module_type is None:
                if verbose:
                    print(f"[PublicSpace] Skip unmatched client module: {name}")
                continue

            candidate_keys = self.type_to_keys.get(module_type, [])
            if len(candidate_keys) == 0:
                raise RuntimeError(f"No public subspace candidates for module_type={module_type}, client module={name}")

            U_c = module.U.detach().float().cpu()
            V_c = module.V.detach().float().cpu()

            best_key = None
            best_score = float("-inf")
            for key in candidate_keys:
                space = self.spaces[key]
                score = combined_subspace_similarity(
                    U_c,
                    V_c,
                    space.U,
                    space.V,
                    bridge_mode=self.bridge_mode,
                )
                if score > best_score:
                    best_score = score
                    best_key = key

            # Robust fallback: layer-proportional matching within the same type.
            if best_key is None or best_score == float("-inf"):
                client_layer = extract_layer_id(name)
                if client_layer < 0:
                    chosen_index = 0
                else:
                    chosen_index = min(client_layer, len(candidate_keys) - 1)
                best_key = candidate_keys[chosen_index]

            match_map[name] = best_key

            if verbose:
                c_shape = f"U{tuple(U_c.shape)} V{tuple(V_c.shape)}"
                p = self.spaces[best_key]
                p_shape = f"U{tuple(p.U.shape)} V{tuple(p.V.shape)}"
                print(
                    f"[PublicSpace] Match {name} {c_shape} -> {best_key} {p_shape} | "
                    f"score={best_score:.6f}"
                )

        if len(match_map) == 0:
            raise RuntimeError("No SVFLinear modules were matched. Please check model injection and module names.")

        return match_map

    def collect_projected_client_updates(
        self,
        client_model: nn.Module,
        match_map: Dict[str, str],
        sample_weight: Optional[float] = None,
    ) -> Dict[str, List[Tuple[torch.Tensor, float]]]:

        updates: Dict[str, List[Tuple[torch.Tensor, float]]] = {}
        weight = 1.0 if sample_weight is None else float(sample_weight)

        for name, module in self.iter_client_svf_modules(client_model):
            if name not in match_map:
                continue
            public_key = match_map[name]
            space = self.spaces[public_key]
            delta_c_to_p = project_client_to_public(
                U_c=module.U,
                V_c=module.V,
                delta_sigma_c=module.delta_sigma,
                U_p=space.U,
                V_p=space.V,
                bridge_mode=self.bridge_mode,
            )
            updates.setdefault(public_key, []).append((delta_c_to_p, weight))

        return updates

    def aggregate_and_refine(
        self,
        client_updates: Iterable[Dict[str, List[Tuple[torch.Tensor, float]]]],
        refine: bool = True,
    ) -> Dict[str, int]:

        bucket: Dict[str, List[torch.Tensor]] = {}
        weight_bucket: Dict[str, List[float]] = {}

        for one_client_updates in client_updates:
            for key, items in one_client_updates.items():
                for delta, weight in items:
                    bucket.setdefault(key, []).append(delta.detach().float().cpu())
                    weight_bucket.setdefault(key, []).append(float(weight))

        update_counts: Dict[str, int] = {}

        for key, deltas in bucket.items():
            weights = weight_bucket.get(key, None)
            delta_p = aggregate_delta_sigmas(deltas, weights=weights)

            space = self.spaces[key]
            if refine:
                U_new, V_new, delta_new = refine_public_basis(space.U, space.V, delta_p)
                space.U = U_new
                space.V = V_new
                space.delta_sigma = delta_new
            else:
                space.delta_sigma = delta_p.contiguous()

            update_counts[key] = len(deltas)

        return update_counts

    @torch.no_grad()
    def broadcast_to_client(self, client_model: nn.Module, match_map: Dict[str, str]) -> None:

        for name, module in self.iter_client_svf_modules(client_model):
            if name not in match_map:
                continue
            public_key = match_map[name]
            space = self.spaces[public_key]
            delta_p_to_c = project_public_to_client(
                U_p=space.U,
                V_p=space.V,
                delta_sigma_p=space.delta_sigma,
                U_c=module.U,
                V_c=module.V,
                bridge_mode=self.bridge_mode,
            )
            module.set_delta_sigma(delta_p_to_c.to(device=module.delta_sigma.device, dtype=module.delta_sigma.dtype))
