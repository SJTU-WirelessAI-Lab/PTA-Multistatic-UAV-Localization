"""ONNX Runtime inference called from MATLAB (see onnx_infer.m)."""

import ctypes
import glob
import os
import sys

# MATLAB overrides LD_LIBRARY_PATH; preload cuDNN from the active Python env if present
for _lib in glob.glob(os.path.join(sys.prefix, 'lib', 'libcudnn.so.*')):
    try:
        ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)
        break
    except OSError:
        pass

import numpy as np
import onnxruntime as ort

_sessions = {}


def load_model(model_path, device='cpu'):
    model_path, device = str(model_path), str(device)
    key = (model_path, device)
    if key not in _sessions:
        if device.startswith('cuda'):
            device_id = int(device.split(':')[1]) if ':' in device else 0
            providers = [('CUDAExecutionProvider', {'device_id': device_id}), 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        _sessions[key] = ort.InferenceSession(model_path, providers=providers)
    return _sessions[key]


def infer(model_path, x, device='cpu'):
    session = load_model(model_path, device)
    x = np.asarray(x, dtype=np.float32)
    return session.run(None, {session.get_inputs()[0].name: x})[0]
