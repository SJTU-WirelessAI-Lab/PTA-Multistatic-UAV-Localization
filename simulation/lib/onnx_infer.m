function out = onnx_infer(model_file, x, device)
% Run an ONNX model through Python onnxruntime (onnx_inference.py in this folder).
% device: 'cpu', 'cuda' or 'cuda:<id>'. Sessions are cached on the Python side.

    lib_dir = fileparts(mfilename('fullpath'));
    if count(py.sys.path, lib_dir) == 0
        insert(py.sys.path, int32(0), lib_dir);
    end
    module = py.importlib.import_module('onnx_inference');
    % atleast_2d: a MATLAB row vector arrives as a 1-D numpy array
    out = double(module.infer(model_file, py.numpy.atleast_2d(py.numpy.array(single(x))), device));
    out = out(:)';
end
