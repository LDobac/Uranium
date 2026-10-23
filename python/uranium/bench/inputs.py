"""Synthetic model inputs for the bench suite: observations of the live feature schema (030) with
random features and masks, and labels sampled by the network itself so every head's label is legal
under the masks (teacher forcing then exercises the same paths as real data)."""

import json

import torch

from uranium.model.config import InputDims

PIXELS_PER_CELL = 32


def feature_dims() -> dict:
    from uranium import _core

    return json.loads(_core.feature_dims())


def random_obs(b: int, n: int, inp: InputDims, device, gen: torch.Generator, valid: int | None = None) -> dict:
    """`b` rows of `n` entity slots, the first `valid` (default all) of them real."""
    d = feature_dims()
    m = d["map"]
    cpu = torch.device("cpu")
    r = lambda *s: torch.rand(*s, generator=gen, device=cpu)  # noqa: E731
    ri = lambda hi, *s: torch.randint(0, hi, s, generator=gen, device=cpu)  # noqa: E731
    valid = n if valid is None else valid
    mask = torch.zeros(b, n, dtype=torch.bool)
    mask[:, :valid] = True
    cat = torch.stack([ri(inp.unit_types, b, n), ri(inp.owners, b, n), ri(inp.orders, b, n), ri(inp.actions, b, n)], 2)
    pos = r(b, n, 2) * (m * PIXELS_PER_CELL)
    cell = (pos[..., 1] // PIXELS_PER_CELL).long() * m + (pos[..., 0] // PIXELS_PER_CELL).long()
    type_mask = r(b, d["actions"]) < 0.3
    type_mask[:, 0] = True
    obs = {
        "entity_cat": cat, "entity_cont": torch.randn(b, n, d["entity_cont"], generator=gen), "entity_mask": mask,
        "entity_cell": cell, "entity_pos": pos, "spatial": torch.randn(b, d["spatial"], m, m, generator=gen),
        "scalar": torch.randn(b, d["scalar"], generator=gen),
        "scalar_cat": torch.stack([ri(inp.races, b), ri(inp.map_families, b), ri(inp.start_locs, b)], 1),
        "z_build": ri(d["actions"], b, d["z_build"]), "z_units": torch.randn(b, d["z_units"], generator=gen),
        "z_present": r(b) < 0.5, "type_mask": type_mask, "unit_cap": r(b, n, d["caps"]) < 0.5,
        "loc_class_mask": r(b, d["loc_classes"], m, m) < 0.5, "is_building": r(b, n) < 0.2,
    }
    return {k: v.to(device) for k, v in obs.items()}


@torch.no_grad()
def sampled_labels(net, obs: dict, gen_seed: int) -> dict:
    """Labels drawn from the network's own distributions (legal under every mask)."""
    dev = obs["entity_mask"].device
    g = torch.Generator(device=dev)
    g.manual_seed(gen_seed)
    out = net.act(obs, net.init_state(obs["entity_mask"].shape[0], dev), 1.0, g)
    return {k: v.clone() for k, v in out["action"].items()}
