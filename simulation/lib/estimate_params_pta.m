function [d_est, aoa_gcs_deg] = estimate_params_pta(Y, p, rx_orientation_deg)
% PTA bistatic parameter estimation for one Rx (single target).
% AoA: strongest subcarrier mapped through the rainbow-beam frequency-angle relation.
% Range: zero-padded IFFT over a sub-band of 2*half_width+1 subcarriers around that peak.

    if isempty(Y) || norm(Y(:)) < eps
        d_est = NaN;
        aoa_gcs_deg = NaN;
        return;
    end

    y = Y(:, 1);
    [~, peak_idx] = max(abs(y) .^ 2);
    aoa_gcs_deg = carrier_to_angle_deg(peak_idx, p) + rx_orientation_deg;

    hw = p.pta_subband_half_width;
    sub = (max(1, peak_idx - hw):min(p.Nc, peak_idx + hw))';
    y_sub = Y(sub, 1);
    if norm(y_sub) < eps
        d_est = NaN;
        return;
    end
    y_sub = y_sub / max(norm(y_sub), eps);

    n_fft = max(round(p.range_fft_n), numel(y_sub));
    range_profile = abs(ifft(y_sub, n_fft, 1)) .^ 2;
    range_axis = (0:n_fft - 1) * p.c0 / (n_fft * p.fdelta);
    valid = find(range_axis >= p.range_window(1) & range_axis <= p.range_window(2));
    [~, k] = max(range_profile(valid));
    d_est = range_axis(valid(k));
end

function angle_deg = carrier_to_angle_deg(carrier_idx, p)
    % Inverse of the rainbow-beam mapping: subcarrier index -> local AoA
    W = (p.Nc - 1) * p.fdelta;
    f0 = p.fc - W / 2;
    tilde_fm = (carrier_idx - 1) * p.fdelta;
    fm = f0 + tilde_fm;
    termA = ((W - tilde_fm) * f0) / (W * fm);
    termB = ((W + f0) * tilde_fm) / (W * fm);
    sin_theta = termA * sin(deg2rad(p.theta_start_deg)) + termB * sin(deg2rad(p.theta_end_deg));
    sin_theta = max(min(sin_theta, 1), -1);
    % The beam angle refers to the arrival (target-to-Rx) direction; the minus sign
    % converts it to the Rx-to-target AoA used by the fusion schemes
    angle_deg = rad2deg(-asin(sin_theta));
end
