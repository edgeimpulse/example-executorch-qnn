# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

# x86_64 only: ExecuTorch's QNN ahead-of-time compiler ships in the x86_64 wheel.
# Pinning the platform matters when building on Apple Silicon.
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# libatomic1 is required by the libc++ that ExecuTorch downloads for the QNN
# libraries. Do NOT install the apt libc++1/libc++abi1: they are too old
# (missing __cxa_init_primary_exception) and shadow the downloaded one.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libatomic1 libgomp1 unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Blocks run without network access, so download the Qualcomm AI Runtime SDK
# (the version this ExecuTorch release pins) and its libc++ at BUILD time.
ENV QNN_STAGING_DIR=/opt/qnn
RUN python3 -c "from executorch.backends.qualcomm.utils.qnn_sdk_setup import setup_qnn_sdk; setup_qnn_sdk()" \
    && ls /opt/qnn

# The loader reads LD_LIBRARY_PATH only at process start, so set it here, not
# from Python. ExecuTorch stages libc++ as libc++.so.1.0 only; add the sonames.
RUN cd /opt/qnn/libcxx-14.0.0 \
    && ln -sf libc++.so.1.0 libc++.so.1 \
    && ln -sf libc++abi.so.1.0 libc++abi.so.1
ENV QNN_SDK_ROOT=/opt/qnn/sdk-2.37.0.250724
ENV LD_LIBRARY_PATH=/opt/qnn/sdk-2.37.0.250724/lib/x86_64-linux-clang:/opt/qnn/libcxx-14.0.0

COPY model.py train.py qnn_export.py qnn_smoke.py ./

# Fail the BUILD (not a training run) if QNN lowering does not work.
RUN python3 qnn_smoke.py

ENTRYPOINT ["python3", "train.py"]
