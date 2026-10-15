# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

# Code added to a public ExecuTorch block's train.py by the porting recipe.
# Sections are separated by "# ---8<---" lines; porting/apply_port.py inserts them.
# ---8<---
    # --- Qualcomm NPU (QNN) export, see docs/porting-to-other-blocks.md ---
    parser.add_argument("--export-qnn", action="store_true", default=False)
    parser.add_argument("--soc-model", type=str, default="QCS9100")
    parser.add_argument("--quant-scheme", type=str, default="8a8w",
                        choices=["8a8w", "16a8w", "16a16w", "16a4w"])
    parser.add_argument("--calibration-samples", type=int, default=200)
# ---8<---
def load_class_names(info_file, n_classes: int) -> list:
    """Class names from train_input.json (Studio passes --info-file); indices otherwise."""
    import json

    try:
        with open(info_file) as f:
            info = json.load(f)
        names = info.get("classes") or info.get("classNames") or []
        if len(names) == n_classes:
            return [str(n) for n in names]
    except Exception:  # noqa: BLE001 - names are a convenience
        pass
    return [str(i) for i in range(n_classes)]


def export_qnn_pte(model, x_train, args, *, eval_x=None, eval_y=None,
                   eval_fn=None, metric_name="accuracy", labels=None) -> dict:
    """Write model_qnn.pte + export_report.json for the Qualcomm NPU.

    `model` must output LOGITS (the trained module, not the softmax export
    wrapper). x_train is already in the model's input layout. Never fails training.
    """
    from qnn_export import export_qnn

    rng = np.random.default_rng(SEED)
    n = min(args.calibration_samples, len(x_train))
    idx = rng.choice(len(x_train), size=n, replace=False)
    calibration = np.ascontiguousarray(x_train[idx], dtype=np.float32)
    sample = torch.from_numpy(np.ascontiguousarray(x_train[:1], dtype=np.float32))
    report = export_qnn(
        model.eval(), sample, calibration,
        os.path.join(args.out_directory, "model_qnn.pte"),
        soc_model=args.soc_model, quant_scheme=args.quant_scheme,
        eval_x=eval_x, eval_y=eval_y, eval_fn=eval_fn, metric_name=metric_name,
        report_path=os.path.join(args.out_directory, "export_report.json"),
        extra_report={"labels": labels, "input_dtype": "float32",
                      "output_semantics": "logits (apply softmax on host)"},
    )
    print(f"QNN export status: {report['status']}", flush=True)
    return report

# ---8<---
    if args.export_qnn:
        # `model` outputs logits; ClassifierExport only adds the softmax Studio needs.
        has_test = len(x_test) > 0
        export_qnn_pte(
            model, x_train, args,
            eval_x=x_test if has_test else None,
            eval_y=label_test if has_test else None,
            labels=load_class_names(args.info_file, num_classes),
        )
# ---8<---
    if args.export_qnn:
        def object_cell_accuracy(m) -> float:
            """evaluate()'s metric on any callable model (the QNN export passes the
            float module, then the int8 graph, which only supports calling)."""
            correct = total = 0
            for x, t in zip(x_test, t_test):
                pred = m(torch.from_numpy(np.ascontiguousarray(x[None]))).argmax(dim=1)[0]
                target = torch.from_numpy(t)
                fg = target > 0
                total += int(fg.sum())
                correct += int((pred[fg] == target[fg]).sum())
            return correct / total if total else float("nan")

        # `model` outputs per-cell logits; FomoExport only adds the softmax Studio needs.
        export_qnn_pte(
            model, x_train, args,
            eval_fn=object_cell_accuracy if len(x_test) else None,
            metric_name="object_cell_accuracy",
            labels=["background"] + load_class_names(args.info_file, num_classes - 1),
        )
