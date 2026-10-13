function d_est = estimate_params_pa(Y, p)
% Phased-array bistatic range estimation for one Rx: IFFT over all subcarriers, peak in the range window.

    y = Y(:, 1);
    if norm(y) < eps
        d_est = NaN;
        return;
    end
    y = y .* conj(p.s_data(:, 1));

    n_fft = max(p.range_fft_n, p.Nc);
    range_profile = abs(ifft(y, n_fft, 1)) .^ 2;
    range_axis = (0:n_fft - 1) * p.c0 / (n_fft * p.fdelta);
    valid = find(range_axis >= p.range_window(1) & range_axis <= p.range_window(2));
    [~, k] = max(range_profile(valid));
    d_est = range_axis(valid(k));
end
