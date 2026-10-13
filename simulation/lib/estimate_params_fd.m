function [d_est, aoa_gcs_deg] = estimate_params_fd(Y, p, rx_orientation_deg)
% Fully digital bistatic parameter estimation for one Rx (single target):
% 2D FFT over (antenna, subcarrier) and the peak inside the range window and beam coverage.

    if size(Y, 3) > 1
        Y = mean(Y, 3);
    end
    Y = Y(:, :, 1) .* reshape(conj(p.s_data(:, 1)).', 1, []);

    n_range = max(p.range_fft_n, p.Nc);
    n_angle = max(p.fd_angle_fft_n, p.Nr);
    score = abs(fftshift(fft(ifft(Y, n_range, 2), n_angle, 1), 1)) .^ 2;

    range_axis = (0:n_range - 1) * p.c0 / (n_range * p.fdelta);
    spatial_freq = (-floor(n_angle / 2):(ceil(n_angle / 2) - 1)) / n_angle;
    sin_theta = -spatial_freq * p.lambda / p.d;
    valid_sin = abs(sin_theta) <= 1;
    angle_lcs_deg = zeros(size(sin_theta));
    angle_lcs_deg(valid_sin) = rad2deg(asin(sin_theta(valid_sin)));

    range_idx = find(range_axis >= p.range_window(1) & range_axis <= p.range_window(2));
    angle_idx = find(valid_sin & angle_lcs_deg >= p.theta_start_deg & angle_lcs_deg <= p.theta_end_deg);

    [~, k] = max(reshape(score(angle_idx, range_idx), [], 1));
    [ia, ir] = ind2sub([numel(angle_idx), numel(range_idx)], k);
    d_est = range_axis(range_idx(ir));
    aoa_gcs_deg = mod(rx_orientation_deg + angle_lcs_deg(angle_idx(ia)), 360);
end
