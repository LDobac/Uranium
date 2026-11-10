"""Activations at the cue-flow stages during one teacher-forced pass, `UraniumNet.action_seq` (the
trunk of `forward_seq` and its ActionType head; the other heads, whose location head runs
convolutions over a varying row count, are not needed). Forward hooks only, the model is not
changed:

- `E`: entity encoder output [rows, N, d] (the focus token is picked from it), `e_pool` its pool
- `sp_pool`: the spatial path into the core input, `c`: the scalar encoder output
- `core_in`: the core input [e_pool, sp_pool, c] (+ the outer LSTM's addition), `x_in`: after the
  GTrXL input projection
- per GTrXL layer l (1-based, as reports/jepa_survey.md 10.3): `L{l}.a_pre` attention output,
  `L{l}.a` its ReLU (gate 1's y), `L{l}.g1` after gate 1, `L{l}.f_pre` FFN output, `L{l}.f` its
  ReLU (gate 2's y), `L{l}.out` after gate 2 (the last layer's out is h_t)
- `h`: the core output the heads read

With the outer LSTM (032 §2.7) the encoders and the core run once per segment; the pieces are
joined back to b-major rows over the whole sequence."""

import torch

CORE_SETS = {
    "none": (),
    "gate": ("x_in", "a", "g1", "f", "out"),  # what the gate decomposition needs
    "full": ("core_in", "x_in", "a_pre", "a", "g1", "f_pre", "f", "out"),
}


class Capture:
    def __init__(self, net, core: str = "gate"):
        self.net = net
        self.want = set(CORE_SETS[core])
        self.on = False
        self.cap: dict[str, list] = {}
        self.handles = []
        hk = self.handles
        hk.append(net.entity.register_forward_hook(lambda m, i, o: self._put("e_pool", o[1])))
        core = net.core
        inp = getattr(core, "input", None)
        if inp is not None and self.want & {"core_in", "x_in"}:
            hk.append(inp.register_forward_hook(lambda m, i, o: (self._put("core_in", i[0]), self._put("x_in", o)) and None))
        for l, layer in enumerate(getattr(core, "layers", []), 1):
            if self.want & {"a_pre"}:
                hk.append(layer.attn.register_forward_hook(lambda m, i, o, l=l: self._put(f"L{l}.a_pre", o)))
            if self.want & {"f_pre"}:
                hk.append(layer.ffn.register_forward_hook(lambda m, i, o, l=l: self._put(f"L{l}.f_pre", o)))
            if self.want & {"a", "g1"}:
                hk.append(layer.gate1.register_forward_hook(lambda m, i, o, l=l: (self._put(f"L{l}.a", i[1]), self._put(f"L{l}.g1", o)) and None))
            if self.want & {"f", "out"}:
                hk.append(layer.gate2.register_forward_hook(lambda m, i, o, l=l: (self._put(f"L{l}.f", i[1]), self._put(f"L{l}.out", o)) and None))
        self.layers = len(getattr(core, "layers", []))

    def keys(self) -> list[str]:
        """The core keys this capture records (in order)."""
        out = [k for k in ("core_in", "x_in") if k in self.want and getattr(self.net.core, "input", None) is not None]
        for l in range(1, self.layers + 1):
            out += [f"L{l}.{p}" for p in ("a_pre", "a", "g1", "f_pre", "f", "out") if p in self.want]
        return out

    def _put(self, k, v):
        if self.on:
            self.cap.setdefault(k, []).append(v.detach())

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def run(self, obs, s0, t: int):
        """`UraniumNet.action_seq` (the trunk and the ActionType head, as `forward_seq` computes
        them) with the hooks on; -> ({"state", "action": ActionType log-probs}, {key: [rows, ...]
        b-major})."""
        self.cap = {}
        self.on = True
        try:
            tr, lp = self.net.action_seq(obs, s0, t)
        finally:
            self.on = False
        bt = obs["entity_cat"].shape[0]
        b = bt // t
        # what the heads read (forward_seq: Heads(labels, temp, h, c, e, sp, ...))
        acts = {"h": tr["h"].detach(), "c": tr["sc"]["c"].detach(), "E": tr["e"].detach(), "sp_pool": tr["sp"]["pool"].detach()}
        for k, lst in self.cap.items():
            # each piece is [b*seg, d] (encoders) or [b, seg, d] (core): back to b-major [b*t, d]
            acts[k] = torch.cat([x.reshape(b, -1, x.shape[-1]) for x in lst], 1).reshape(bt, -1)
        self.cap = {}
        return {"state": tr["state"], "action": lp}, acts
