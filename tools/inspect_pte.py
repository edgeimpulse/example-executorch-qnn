#!/usr/bin/env python3
"""Show what is inside an ExecuTorch .pte: which backend(s) it was lowered to,
how much of the graph they own, what still runs as portable CPU kernels, and
the input/output tensors.

    python tools/inspect_pte.py model.pte [model_qnn.pte ...]

Works on any machine with the `executorch` pip package (x86 or Arm). It only
parses the file, so it can inspect a QNN .pte even where the QNN backend is not
available. Example output for the two artifacts of this project:

    model.pte      delegates: XnnpackBackend x1   portable ops: none
    model_qnn.pte  delegates: QnnBackend x1       portable ops: (whatever fell back)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import Counter

_DTYPES = {0: "uint8", 1: "int8", 2: "int16", 3: "int32", 4: "int64", 5: "float16",
           6: "float32", 7: "float64", 11: "bool"}


def _tensor_desc(value) -> str:
    t = getattr(value, "val", None)
    sizes = getattr(t, "sizes", None)
    if sizes is None:
        return type(t).__name__
    dtype = _DTYPES.get(int(getattr(t, "scalar_type", -1)), str(getattr(t, "scalar_type", "?")))
    return f"{dtype}{list(sizes)}"


def inspect(path: str) -> None:
    from executorch.exir._serialize._program import deserialize_pte_binary

    data = open(path, "rb").read()
    program = deserialize_pte_binary(data).program
    print(f"{path}")
    print(f"  size      : {len(data) / 1024:.1f} KB   sha256 {hashlib.sha256(data).hexdigest()[:16]}")
    for plan in program.execution_plan:
        backends = Counter(d.id for d in plan.delegates)
        ops = Counter(op.name + (f".{op.overload}" if op.overload else "") for op in plan.operators)
        ins = [_tensor_desc(plan.values[i]) for i in plan.inputs]
        outs = [_tensor_desc(plan.values[i]) for i in plan.outputs]
        print(f"  method    : {plan.name}")
        print(f"  inputs    : {', '.join(ins)}")
        print(f"  outputs   : {', '.join(outs)}")
        print("  delegates : " + (", ".join(f"{b} x{n}" for b, n in backends.items()) or
                                  "none (everything runs on portable CPU kernels)"))
        if ops:
            print("  portable ops (not delegated, run on the CPU):")
            for name, n in sorted(ops.items()):
                print(f"      {name}  x{n}")
        else:
            print("  portable ops: none -- the whole graph is inside the delegate(s)")
    if program.backend_delegate_data or program.segments:
        sizes = [s.size for s in program.segments]
        if sizes:
            print(f"  segments  : {len(sizes)} (largest {max(sizes) / 1024:.1f} KB; "
                  "delegate payloads such as a QNN context binary live here)")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect ExecuTorch .pte files")
    ap.add_argument("pte", nargs="+")
    for p in ap.parse_args().pte:
        if not os.path.exists(p):
            sys.exit(f"not found: {p}")
        inspect(p)


if __name__ == "__main__":
    main()
