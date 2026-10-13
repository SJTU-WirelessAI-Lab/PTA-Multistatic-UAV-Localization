function pos_est = fuse_pf_mlp(r1, r2, p, model_file, device)
% PF-MLP: parameter-level fusion network, input [d1, theta1, d2, theta2] -> position.
% Normalization matches model_training/src/data_loader.py (FusionDataset).

    L = p.L;
    x = [(r1.d_l - 1.5 * L) / (0.5 * L), (deg2rad(r1.aoa_gcs_deg) - pi) / pi, ...
         (r2.d_l - 1.5 * L) / (0.5 * L), (deg2rad(r2.aoa_gcs_deg) - pi) / pi];
    out = onnx_infer(model_file, x, device);
    pos_est = denormalize_position(out, L);
end

function pos = denormalize_position(out, L)
    x_center = sqrt(3) * L / 3;
    pos = [out(1) * x_center + x_center, out(2) * (L / 2)];
end
