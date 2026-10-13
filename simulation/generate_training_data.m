% Training data for the two networks, generated from the same random targets:
%   PF-MLP: PTA parameter estimates [d1, theta1 (rad), d2, theta2 (rad)]
%   SF-CNN: received PTA amplitudes |y_m| of Rx1 and Rx2 (2 x Nc values)
% Labels are the true positions. Output: data/pfmlp_Ptx<P>dBm.mat, data/sfcnn_Ptx<P>dBm.mat,
% read by model_training/train.py. Run once per Ptx for the power-matched models.

clear; clc;
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'lib'));

%% Settings
num_samples = 100000;
ptx = 52;                      % dBm
seed = 42;
out_dir = fullfile(root, '..', 'model_training', 'data');

%% Generate
p = system_params();
Ptx = 10 ^ ((ptx - 30) / 10);
rng(seed, 'twister');
pf_inputs = zeros(num_samples, 4, 'single');
sf_inputs = zeros(num_samples, 2 * p.Nc, 'single');
labels = zeros(num_samples, 2, 'single');
tic_all = tic;
for t = 1:num_samples
    [pos, vel] = generate_random_target(p.area, p.velocity_range);
    [Y1, Y2, N1, N2] = simulate_element_signals(pos, vel, Ptx, p);
    Ypta1 = rx_observation('PTA', Y1, N1, p);
    Ypta2 = rx_observation('PTA', Y2, N2, p);
    [d1, a1] = estimate_params_pta(Ypta1, p, p.orientation_rx(1));
    [d2, a2] = estimate_params_pta(Ypta2, p, p.orientation_rx(2));

    pf_inputs(t, :) = single([d1, deg2rad(a1), d2, deg2rad(a2)]);
    sf_inputs(t, :) = single([abs(Ypta1(:)); abs(Ypta2(:))]');
    labels(t, :) = single(pos);
    if mod(t, 5000) == 0
        fprintf('%d / %d samples (%.0f s)\n', t, num_samples, toc(tic_all));
    end
end

%% Save
if ~isfolder(out_dir), mkdir(out_dir); end
metadata = struct('scene_L', p.L, 'Nc', p.Nc, 'Ns', p.Ns, 'Ptx_dBm', ptx, 'random_seed', seed);
training_dataset = struct('inputs', pf_inputs, 'labels', labels, 'metadata', metadata);
save(fullfile(out_dir, sprintf('pfmlp_Ptx%ddBm.mat', ptx)), 'training_dataset');
training_dataset = struct('inputs', sf_inputs, 'labels', labels, 'metadata', metadata);
save(fullfile(out_dir, sprintf('sfcnn_Ptx%ddBm.mat', ptx)), 'training_dataset', '-v7.3');
fprintf('Saved to %s\n', out_dir);
