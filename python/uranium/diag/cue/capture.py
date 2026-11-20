import torch

CORE_SETS = {
    "none": (),
    "gate": ("x_in", "a", "g1", "f", "out"),
    "full": ("core_in", "x_in", "a_pre", "a", "g1", "f_pre", "f", "out"),
}


class Capture:
    def __init__(self, net, core: str = "gate"):
        self.net = net
        self.has_pool = not getattr(net, "patch", 0)
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

    def stages(self, names) -> list[str]:
        return [k for k in names if k != "sp_pool" or self.has_pool]

    def keys(self) -> list[str]:
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
        self.cap = {}
        self.on = True
        try:
            tr, lp = self.net.action_seq(obs, s0, t)
        finally:
            self.on = False
        bt = obs["entity_cat"].shape[0]
        b = bt // t
        acts = {"h": tr["h"].detach(), "c": tr["sc"]["c"].detach(), "E": tr["e"].detach()}
        if self.has_pool:
            acts["sp_pool"] = tr["sp"]["pool"].detach()
        for k, lst in self.cap.items():
            acts[k] = torch.cat([x.reshape(b, -1, x.shape[-1]) for x in lst], 1).reshape(bt, -1)
        self.cap = {}
        return {"state": tr["state"], "action": lp}, acts
