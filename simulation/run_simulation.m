% Monte Carlo evaluation of PTA-enabled multistatic UAV localization.
%   - Fusion schemes on the PTA receiver over the Ptx sweep: Table III, Figs. 3-5
%   - FD / PTA / PA receiver comparison at rx_compare_ptx: Table II
% Results are saved to results/ and summarized at the end; figures: plot_results.m.

clear; clc;
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'lib'));

%% Settings
num_trials = 10000;
ptx_list = 42:2:52;            % transmit power (dBm)
rx_compare_ptx = 52;           % FD and PA receivers are simulated at this power only
seed = 47;

% Networks (optional; schemes whose model file is missing are skipped).
% "Generalized" models are trained at nn_train_ptx and used at every power,
% "power-matched" models are trained at each Ptx.
model_dir = fullfile(root, 'models');
nn_train_ptx = 52;
pfmlp_file = @(ptx) fullfile(model_dir, sprintf('pfmlp_Ptx%ddBm.onnx', ptx));
sfcnn_file = @(ptx) fullfile(model_dir, sprintf('sfcnn_Ptx%ddBm.onnx', ptx));
nn_device = 'cpu';             % 'cpu' or 'cuda:<id>'
python_exe = fullfile(root, '..', 'model_training', '.venv', 'bin', 'python');   % uv environment, see README

%% Setup
p = system_params();
methods = {'GI_PL', 'GDOP_PL', 'GDOP_WLS', 'PF_MLP', 'PF_MLP_matched', 'SF_CNN', 'SF_CNN_matched'};
num_ptx = numel(ptx_list);

if isfile(python_exe) && pyenv().Status == "NotLoaded"
    pyenv('Version', python_exe);
end
models = unique([arrayfun(pfmlp_file, [nn_train_ptx, ptx_list], 'UniformOutput', false), ...
                 arrayfun(sfcnn_file, [nn_train_ptx, ptx_list], 'UniformOutput', false)]);
for k = 1:numel(models)
    if isfile(models{k})
        % Warm-up so that session creation is not counted in the latency
        if contains(models{k}, 'pfmlp')
            onnx_infer(models{k}, zeros(1, 4), nn_device);
        else
            onnx_infer(models{k}, zeros(1, 2, 1, p.Nc), nn_device);
        end
    else
        fprintf('Model not found, scheme skipped: %s\n', models{k});
    end
end

res = struct('ptx_list', ptx_list, 'num_trials', num_trials, 'seed', seed, 'methods', {methods});
res.geometry = struct('L', p.L, 'pos_tx', p.pos_tx, 'pos_rx1', p.pos_rx1, 'pos_rx2', p.pos_rx2);
res.true_pos = nan(num_trials, 2, num_ptx);
res.pta_params = nan(num_trials, 4, num_ptx);    % [d1, aoa1, d2, aoa2] (m, deg)
for m = 1:numel(methods)
    res.est.(methods{m}) = nan(num_trials, 2, num_ptx);
    res.time.(methods{m}) = nan(num_trials, num_ptx);
end
for arch = {'FD', 'PA'}
    res.rx.(arch{1}).est_pos = nan(num_trials, 2);
    res.rx.(arch{1}).params = nan(num_trials, 4);
    res.rx.(arch{1}).time = nan(num_trials, 1);
end

rng(seed, 'twister');

%% Monte Carlo
for ip = 1:num_ptx
    ptx = ptx_list(ip);
    Ptx = 10 ^ ((ptx - 30) / 10);
    is = find(p.sigma_ptx == ptx);
    sigma = struct('d', p.sigma_d(is, :), 'theta', p.sigma_theta_deg(is, :));
    nn_file = struct('PF_MLP', pfmlp_file(nn_train_ptx), 'PF_MLP_matched', pfmlp_file(ptx), ...
        'SF_CNN', sfcnn_file(nn_train_ptx), 'SF_CNN_matched', sfcnn_file(ptx));
    fprintf('\n==== Ptx = %d dBm ====\n', ptx);
    tic_ptx = tic;

    for t = 1:num_trials
        [target_pos, target_vel] = generate_random_target(p.area, p.velocity_range);
        [Y1, Y2, N1, N2] = simulate_element_signals(target_pos, target_vel, Ptx, p);
        res.true_pos(t, :, ip) = target_pos;

        % PTA bistatic parameter estimation (the two Rx run in parallel: latency = max)
        Ypta1 = rx_observation('PTA', Y1, N1, p);
        Ypta2 = rx_observation('PTA', Y2, N2, p);
        tic_e = tic; [d1, a1] = estimate_params_pta(Ypta1, p, p.orientation_rx(1)); t1 = toc(tic_e);
        tic_e = tic; [d2, a2] = estimate_params_pta(Ypta2, p, p.orientation_rx(2)); t2 = toc(tic_e);
        t_param = max(t1, t2);
        res.pta_params(t, :, ip) = [d1, a1, d2, a2];
        [r1, r2] = link_structs(d1, a1, d2, a2, sigma);

        for m = 1:numel(methods)
            name = methods{m};
            if isfield(nn_file, name) && ~isfile(nn_file.(name))
                continue;
            end
            pos = [NaN, NaN];
            tic_f = tic;
            try
                switch name
                    case 'GI_PL'
                        pos = fuse_pl_weighted(r1, r2, p.pos_tx, p.pos_rx1, p.pos_rx2, 'ray');
                    case 'GDOP_PL'
                        pos = fuse_pl_weighted(r1, r2, p.pos_tx, p.pos_rx1, p.pos_rx2, 'gdop');
                    case 'GDOP_WLS'
                        pos = fuse_gdop_wls(r1, r2, p.pos_tx, p.pos_rx1, p.pos_rx2);
                    case {'PF_MLP', 'PF_MLP_matched'}
                        pos = fuse_pf_mlp(r1, r2, p, nn_file.(name), nn_device);
                    case {'SF_CNN', 'SF_CNN_matched'}
                        pos = fuse_sf_cnn(Ypta1, Ypta2, p, nn_file.(name), nn_device);
                end
            catch err
                warning('%s failed at trial %d: %s', name, t, err.message);
                pos = [NaN, NaN];
            end
            t_fuse = toc(tic_f);
            if all(isfinite(pos))
                res.est.(name)(t, :, ip) = pos;
                % SF-CNN latency is the network call only; the others include parameter estimation
                res.time.(name)(t, ip) = t_fuse + ~startsWith(name, 'SF_CNN') * t_param;
            end
        end

        % Receiver architecture comparison on the same targets and noise
        if ptx == rx_compare_ptx
            Yfd1 = rx_observation('FD', Y1, N1, p);
            Yfd2 = rx_observation('FD', Y2, N2, p);
            tic_e = tic; [fd1, fa1] = estimate_params_fd(Yfd1, p, p.orientation_rx(1)); t1 = toc(tic_e);
            tic_e = tic; [fd2, fa2] = estimate_params_fd(Yfd2, p, p.orientation_rx(2)); t2 = toc(tic_e);
            [q1, q2] = link_structs(fd1, fa1, fd2, fa2, sigma);   % FD uses the same link sigma as PTA
            tic_f = tic; pos = fuse_gdop_wls(q1, q2, p.pos_tx, p.pos_rx1, p.pos_rx2); t_fuse = toc(tic_f);
            res.rx.FD.params(t, :) = [fd1, fa1, fd2, fa2];
            res.rx.FD.est_pos(t, :) = pos;
            res.rx.FD.time(t) = max(t1, t2) + t_fuse;

            Ypa1 = rx_observation('PA', Y1, N1, p);
            Ypa2 = rx_observation('PA', Y2, N2, p);
            tic_e = tic; pd1 = estimate_params_pa(Ypa1, p); t1 = toc(tic_e);
            tic_e = tic; pd2 = estimate_params_pa(Ypa2, p); t2 = toc(tic_e);
            tic_f = tic; pos = fuse_pa_range_only(pd1, pd2, p); t_fuse = toc(tic_f);
            res.rx.PA.params(t, :) = [pd1, NaN, pd2, NaN];
            res.rx.PA.est_pos(t, :) = pos;
            res.rx.PA.time(t) = max(t1, t2) + t_fuse;
        end

        if mod(t, 1000) == 0
            fprintf('  %d / %d trials (%.0f s)\n', t, num_trials, toc(tic_ptx));
        end
    end
end

%% Save and summarize
if ~isfolder(fullfile(root, 'results')), mkdir(fullfile(root, 'results')); end
result_file = fullfile(root, 'results', sprintf('sim_%s.mat', datestr(now, 'yyyymmdd_HHMMSS')));
save(result_file, '-struct', 'res');
fprintf('\nSaved %s\n', result_file);
summarize_results(res, p, rx_compare_ptx);

%% Local functions
function [r1, r2] = link_structs(d1, a1, d2, a2, sigma)
    r1 = struct('d_l', d1, 'aoa_gcs_deg', a1, 'sigma_d', sigma.d(1), 'sigma_angle_deg', sigma.theta(1));
    r2 = struct('d_l', d2, 'aoa_gcs_deg', a2, 'sigma_d', sigma.d(2), 'sigma_angle_deg', sigma.theta(2));
end

function summarize_results(res, p, rx_compare_ptx)
    ip = find(res.ptx_list == rx_compare_ptx);
    if isempty(ip)
        print_rmse_vs_ptx(res);
        return;
    end
    truth = true_link_params(res.true_pos(:, :, ip), p);

    fprintf('\nTable II: receiver architectures at %d dBm\n', rx_compare_ptx);
    fprintf('%-6s %10s %10s %10s %13s\n', 'Arch', 'AoA (deg)', 'Range (m)', 'Pos. (m)', 'Latency (ms)');
    rows = {'FD', res.rx.FD.params, res.rx.FD.est_pos, res.rx.FD.time;
            'PTA', res.pta_params(:, :, ip), res.est.GDOP_WLS(:, :, ip), res.time.GDOP_WLS(:, ip);
            'PA', res.rx.PA.params, res.rx.PA.est_pos, res.rx.PA.time};
    for k = 1:size(rows, 1)
        prm = rows{k, 2};
        range_err = prm(:, [1, 3]) - truth(:, [1, 3]);
        angle_err = mod(prm(:, [2, 4]) - truth(:, [2, 4]) + 180, 360) - 180;
        fprintf('%-6s %10.2f %10.2f %10.2f %13.2f\n', rows{k, 1}, rms_valid(angle_err(:)), ...
            rms_valid(range_err(:)), rms_valid(position_errors(rows{k, 3}, res.true_pos(:, :, ip))), ...
            1e3 * mean(rows{k, 4}, 'omitnan'));
    end

    fprintf('\nTable III: fusion schemes at %d dBm\n', rx_compare_ptx);
    fprintf('%-16s %9s %10s %13s\n', 'Scheme', 'RMSE (m)', 'P90 (m)', 'Latency (ms)');
    for m = 1:numel(res.methods)
        e = position_errors(res.est.(res.methods{m})(:, :, ip), res.true_pos(:, :, ip));
        fprintf('%-16s %9.2f %10.2f %13.2f\n', res.methods{m}, rms_valid(e), ...
            percentile(e, 90), 1e3 * mean(res.time.(res.methods{m})(:, ip), 'omitnan'));
    end
    print_rmse_vs_ptx(res);
end

function print_rmse_vs_ptx(res)
    fprintf('\nFig. 5: position RMSE (m) vs. Ptx\n%-16s', 'Scheme');
    fprintf('%8d', res.ptx_list);
    fprintf('\n');
    for m = 1:numel(res.methods)
        fprintf('%-16s', res.methods{m});
        for i = 1:numel(res.ptx_list)
            fprintf('%8.2f', rms_valid(position_errors(res.est.(res.methods{m})(:, :, i), res.true_pos(:, :, i))));
        end
        fprintf('\n');
    end
end

function truth = true_link_params(pos, p)
    d_tx = vecnorm(pos - p.pos_tx, 2, 2);
    rx = {p.pos_rx1, p.pos_rx2};
    truth = zeros(size(pos, 1), 4);
    for i = 1:2
        v = pos - rx{i};
        truth(:, 2 * i - 1) = d_tx + vecnorm(v, 2, 2);
        truth(:, 2 * i) = mod(rad2deg(atan2(v(:, 2), v(:, 1))), 360);
    end
end

function e = position_errors(est, truth)
    e = vecnorm(est - truth, 2, 2);
end

function r = rms_valid(e)
    e = e(isfinite(e));
    r = sqrt(mean(e .^ 2));
end

function v = percentile(e, pct)
    % Same definition as prctile (linear interpolation between midpoints)
    e = sort(e(isfinite(e)));
    n = numel(e);
    if n == 0
        v = NaN;
        return;
    end
    v = interp1([0; ((1:n)' - 0.5) / n; 1] * 100, [e(1); e; e(end)], pct);
end
