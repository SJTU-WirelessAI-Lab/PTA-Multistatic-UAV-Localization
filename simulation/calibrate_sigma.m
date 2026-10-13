% Calibrate the error std of the PTA bistatic range and AoA estimates per link and Ptx.
% The result goes into p.sigma_d / p.sigma_theta_deg of lib/system_params.m (used by
% GDOP-WLS and the GDOP-weighted initialization). Errors from all seeds are pooled.

clear; clc;
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'lib'));

%% Settings
num_trials = 10000;            % per seed and Ptx
ptx_list = 42:2:52;            % dBm
seeds = 48:52;

%% Monte Carlo
p = system_params();
sigma_d = nan(numel(ptx_list), 2);
sigma_theta_deg = nan(numel(ptx_list), 2);
for ip = 1:numel(ptx_list)
    Ptx = 10 ^ ((ptx_list(ip) - 30) / 10);
    range_err = nan(num_trials * numel(seeds), 2);
    angle_err = nan(num_trials * numel(seeds), 2);
    row = 0;
    for s = seeds
        rng(s, 'twister');
        for t = 1:num_trials
            [pos, vel] = generate_random_target(p.area, p.velocity_range);
            [Y1, Y2, N1, N2] = simulate_element_signals(pos, vel, Ptx, p);
            [d1, a1] = estimate_params_pta(rx_observation('PTA', Y1, N1, p), p, p.orientation_rx(1));
            [d2, a2] = estimate_params_pta(rx_observation('PTA', Y2, N2, p), p, p.orientation_rx(2));

            d_tx = norm(pos - p.pos_tx);
            v1 = pos - p.pos_rx1;
            v2 = pos - p.pos_rx2;
            row = row + 1;
            range_err(row, :) = [d1 - (d_tx + norm(v1)), d2 - (d_tx + norm(v2))];
            aoa_true = mod(rad2deg([atan2(v1(2), v1(1)), atan2(v2(2), v2(1))]), 360);
            angle_err(row, :) = mod([a1, a2] - aoa_true + 180, 360) - 180;
        end
        fprintf('Ptx %d dBm, seed %d done\n', ptx_list(ip), s);
    end
    for i = 1:2
        sigma_d(ip, i) = std(range_err(isfinite(range_err(:, i)), i));
        sigma_theta_deg(ip, i) = std(angle_err(isfinite(angle_err(:, i)), i));
    end
end

%% Output
T = table(ptx_list(:), sigma_d(:, 1), sigma_d(:, 2), sigma_theta_deg(:, 1), sigma_theta_deg(:, 2), ...
    'VariableNames', {'Ptx_dBm', 'SigmaD1_m', 'SigmaD2_m', 'SigmaTheta1_deg', 'SigmaTheta2_deg'});
disp(T);
if ~isfolder(fullfile(root, 'results')), mkdir(fullfile(root, 'results')); end
writetable(T, fullfile(root, 'results', 'sigma_table.csv'));
