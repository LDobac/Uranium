"""Burn module paths <-> PyTorch parameters (032-AC10, AMB-000-4; used by the parity check until
Burn is removed). The PyTorch model keeps Burn's module names, so the map is mechanical:
- `nn.Linear`: Burn stores the weight [in, out] (PyTorch [out, in]): transposed.
- `nn.LayerNorm`: Burn `gamma` / `beta` = PyTorch `weight` / `bias`.
- `nn.LSTM`: Burn has one Linear pair per gate (`<gate>_gate.{input,hidden}_transform`); PyTorch
  packs them in the order input, forget, cell, output (i, f, g, o).
- Conv2d, ConvTranspose2d, Embedding and plain parameters: the same layout and name."""

import torch
import torch.nn as nn

LSTM_GATES = ("input", "forget", "cell", "output")  # PyTorch nn.LSTM order i, f, g, o


def burn_to_torch(model: nn.Module, burn: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """A `state_dict` for `model` from Burn tensors by module path. Fails on a missing or unused
    Burn tensor, so a name drift shows at once."""
    out: dict[str, torch.Tensor] = {}
    used: set[str] = set()

    def take(key: str) -> torch.Tensor:
        used.add(key)
        return burn[key]

    for name, mod in model.named_modules():
        p = f"{name}." if name else ""
        if isinstance(mod, nn.Linear):
            out[p + "weight"] = take(p + "weight").t().contiguous()
            if mod.bias is not None:
                out[p + "bias"] = take(p + "bias")
        elif isinstance(mod, nn.LayerNorm):
            out[p + "weight"] = take(p + "gamma")
            out[p + "bias"] = take(p + "beta")
        elif isinstance(mod, nn.LSTM):
            g = lambda gate, part, kind: take(f"{p}{gate}_gate.{part}_transform.{kind}")  # noqa: E731
            out[p + "weight_ih_l0"] = torch.cat([g(x, "input", "weight").t() for x in LSTM_GATES], 0).contiguous()
            out[p + "weight_hh_l0"] = torch.cat([g(x, "hidden", "weight").t() for x in LSTM_GATES], 0).contiguous()
            out[p + "bias_ih_l0"] = torch.cat([g(x, "input", "bias") for x in LSTM_GATES], 0)
            out[p + "bias_hh_l0"] = torch.cat([g(x, "hidden", "bias") for x in LSTM_GATES], 0)
        elif isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d, nn.Embedding)):
            for k, _ in mod.named_parameters(recurse=False):
                out[p + k] = take(p + k)
    for k, _ in model.named_parameters():
        if k not in out:
            out[k] = take(k)
    missing = [k for k in model.state_dict() if k not in out]
    unused = sorted(set(burn) - used)
    if missing or unused:
        raise KeyError(f"burn -> torch: missing {missing[:8]}, unused Burn tensors {unused[:8]}")
    return out


def torch_to_burn(model: nn.Module, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The inverse layout map, for comparing PyTorch gradients / parameters with Burn's by path."""
    out: dict[str, torch.Tensor] = {}
    done: set[str] = set()
    for name, mod in model.named_modules():
        p = f"{name}." if name else ""
        if isinstance(mod, nn.Linear):
            out[p + "weight"] = tensors[p + "weight"].t()
            done.add(p + "weight")
            if mod.bias is not None:
                out[p + "bias"] = tensors[p + "bias"]
                done.add(p + "bias")
        elif isinstance(mod, nn.LayerNorm):
            out[p + "gamma"], out[p + "beta"] = tensors[p + "weight"], tensors[p + "bias"]
            done |= {p + "weight", p + "bias"}
        elif isinstance(mod, nn.LSTM):
            for torch_k, part, kind in (("weight_ih_l0", "input", "weight"), ("weight_hh_l0", "hidden", "weight"), ("bias_ih_l0", "input", "bias"), ("bias_hh_l0", "hidden", "bias")):
                for gate, chunk in zip(LSTM_GATES, tensors[p + torch_k].chunk(4, 0)):
                    out[f"{p}{gate}_gate.{part}_transform.{kind}"] = chunk.t() if kind == "weight" else chunk
                done.add(p + torch_k)
    for k, v in tensors.items():
        if k not in done:
            out[k] = v
    return out
