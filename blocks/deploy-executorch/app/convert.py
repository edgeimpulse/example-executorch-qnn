"""Offline ONNX -> ExecuTorch (.pte) converter.

Provided so users can reproduce the conversion the deployment block performs, or
run it themselves when the block packaged only `model.onnx`.

Usage:
    python app/convert.py model.onnx model.pte --backend xnnpack
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ONNX to ExecuTorch .pte")
    parser.add_argument("onnx", help="Path to the input .onnx model")
    parser.add_argument("pte", help="Path to write the output .pte program")
    parser.add_argument("--backend", default="xnnpack",
                        choices=["xnnpack", "portable"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import onnx
    import torch
    from onnx2torch import convert
    from executorch.exir import to_edge_transform_and_lower

    onnx_model = onnx.load(args.onnx)
    torch_model = convert(onnx_model).eval()

    dims = onnx_model.graph.input[0].type.tensor_type.shape.dim
    shape = [(d.dim_value if d.dim_value > 0 else 1) for d in dims]
    example = (torch.randn(*shape),)

    exported = torch.export.export(torch_model, example)
    if args.backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )
        program = to_edge_transform_and_lower(
            exported, partitioner=[XnnpackPartitioner()]
        ).to_executorch()
    else:
        program = to_edge_transform_and_lower(exported).to_executorch()

    with open(args.pte, "wb") as f:
        f.write(program.buffer)
    print(f"Wrote {args.pte} (backend={args.backend}).")


if __name__ == "__main__":
    main()
