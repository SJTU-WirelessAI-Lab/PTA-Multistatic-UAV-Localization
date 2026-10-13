function pos_est = fuse_sf_cnn(Y1, Y2, p, model_file, device)
% SF-CNN: signal-level fusion network on the received PTA amplitudes |y_m| of both Rx.
% The dB normalization of the training set is read from the JSON file next to the model
% (written by model_training/export_to_onnx.py).

    persistent norm_cache
    if isempty(norm_cache)
        norm_cache = containers.Map();
    end
    if ~isKey(norm_cache, model_file)
        [folder, name] = fileparts(model_file);
        norm_cache(model_file) = jsondecode(fileread(fullfile(folder, [name '.json'])));
    end
    nrm = norm_cache(model_file);

    db1 = 20 * log10(abs(Y1(:, 1).') + 1e-20);
    db2 = 20 * log10(abs(Y2(:, 1).') + 1e-20);
    x = zeros(1, 2, 1, p.Nc, 'single');
    x(1, 1, 1, :) = single((db1 - nrm.db_offset) / nrm.db_half_range);
    x(1, 2, 1, :) = single((db2 - nrm.db_offset) / nrm.db_half_range);

    out = onnx_infer(model_file, x, device);
    x_center = sqrt(3) * p.L / 3;
    pos_est = [out(1) * x_center + x_center, out(2) * (p.L / 2)];
end
