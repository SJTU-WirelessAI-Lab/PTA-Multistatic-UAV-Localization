import argparse
import json
import os

import numpy as np
import onnx
import onnxruntime
import torch
import yaml

from src.models import create_model
from src.utils import load_norm_params, to_builtin_types


def main():
    parser = argparse.ArgumentParser(description='Export a paper model and MATLAB sidecar.')
    parser.add_argument('--experiment_dir', required=True, help='Training output directory.')
    args = parser.parse_args()
    with open(os.path.join(args.experiment_dir, 'config.yaml')) as stream:
        config = yaml.safe_load(stream)
    name = config['output']['experiment_name']
    model = create_model(config)
    model.load_state_dict(torch.load(
        os.path.join(args.experiment_dir, f'{name}.pth'), map_location='cpu',
        weights_only=True
    ), strict=True)
    model.eval()
    norm = load_norm_params(os.path.join(args.experiment_dir, 'norm_params.yaml'))
    sidecar = {'model': 'pfmlp', 'scene_L': norm['L']}
    if config['model']['name'] == 'SignalFusionNetLite':
        dummy_input = torch.randn(1, 2, 1, config['model']['params']['Nc'], requires_grad=True)
        sidecar.update(model='sfcnn', db_offset=norm['db_offset'],
                       db_half_range=norm['db_half_range'])
    else:
        dummy_input = torch.randn(1, 4, requires_grad=True)
    onnx_path = os.path.join(args.experiment_dir, f'{name}.onnx')
    torch.onnx.export(
        model, dummy_input, onnx_path, export_params=True, opset_version=12,
        do_constant_folding=True, input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        dynamo=False
    )
    onnx.checker.check_model(onnx.load(onnx_path))
    session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_output = session.run(None, {session.get_inputs()[0].name: dummy_input.detach().numpy()})[0]
    with torch.no_grad():
        torch_output = model(dummy_input).numpy()
    np.testing.assert_allclose(torch_output, ort_output, rtol=1e-3, atol=1e-5)
    with open(os.path.splitext(onnx_path)[0] + '.json', 'w') as stream:
        json.dump(to_builtin_types(sidecar), stream, indent=2)
        stream.write('\n')
    print(f'Exported {onnx_path} and JSON sidecar.')
    print(f'PyTorch/ONNX max absolute difference: {np.max(np.abs(torch_output - ort_output)):.9g}')


if __name__ == '__main__':
    main()
