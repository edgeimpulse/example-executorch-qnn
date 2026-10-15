# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

# Use a slim Python base. CPU-only training keeps the image small; switch to an
# nvidia/cuda base image if you need GPU acceleration.
# QNN lowering only ships for x86_64 Linux, so pin the platform (matters when
# building on Apple Silicon).
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# libatomic1 is required by the libc++ that ExecuTorch downloads for the QNN
# libraries. Do NOT install the apt libc++1/libc++abi1: they are too old
# (missing __cxa_init_primary_exception) and shadow the downloaded one.
# System deps kept minimal. Add build tooling here only if a wheel needs it.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libatomic1 libgomp1 unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

ENV TORCH_HOME=/opt/torch
RUN python3 -c "from torchvision.models import mobilenet_v2, MobileNet_V2_Weights; mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)"

# Bake the Qualcomm AI Runtime SDK (the version this ExecuTorch release pins)
# and the libc++ it needs into the image. Blocks run without network access, so
# both must be fetched at build time. QNN_STAGING_DIR makes the location
# deterministic.
ENV QNN_STAGING_DIR=/opt/qnn
RUN python3 -c "from executorch.backends.qualcomm.utils.qnn_sdk_setup import setup_qnn_sdk; setup_qnn_sdk()" \
    && ls /opt/qnn

# The QNN support libraries (libQnnHtpPrepare, libQnnSystem, libc++ ...) are
# resolved by the dynamic loader BY NAME, and a process only reads
# LD_LIBRARY_PATH at start-up -- setting it from Python afterwards has no
# effect. So export it in the image, like the ExecuTorch docs do in the shell.
# ExecuTorch stages libc++ as libc++.so.1.0 only; the loader needs the soname.
RUN cd /opt/qnn/libcxx-14.0.0 \
    && ln -sf libc++.so.1.0 libc++.so.1 \
    && ln -sf libc++abi.so.1.0 libc++abi.so.1 \
    && ls -l /opt/qnn/libcxx-14.0.0
ENV QNN_SDK_ROOT=/opt/qnn/sdk-2.37.0.250724
ENV LD_LIBRARY_PATH=/opt/qnn/sdk-2.37.0.250724/lib/x86_64-linux-clang:/opt/qnn/libcxx-14.0.0

COPY model.py qnn_export.py qnn_smoke.py train.py ./

# Fail the BUILD (not a training run) if QNN lowering does not work.
RUN python3 qnn_smoke.py

# Edge Impulse calls the entrypoint with --data-directory, --out-directory,
# --info-file and any custom parameters from parameters.json.
ENTRYPOINT ["python3", "train.py"]