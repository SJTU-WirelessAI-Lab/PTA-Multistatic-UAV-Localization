function H_echo = generate_ISAC_Hecho_channel_vec(Nt, Nr, Nc, Ns, fc, lambda, d_tx, d_rx, Ts, ...
        target_pos_xy, target_vel_xy, target_rcs, fdelta, rx_pos_xy, Ge_t, Ge_r, rx_orientation_deg, Ptx)
% Wideband bistatic echo channel of a point target, Tx at the origin.
% Output H_echo is Nr x Nt x Ns x Nc (Rx antenna, Tx antenna, symbol, subcarrier).

    c0 = 3e8;
    pos_tx = [0, 0];

    vec_tx_to_target = target_pos_xy - pos_tx;
    r_tx = max(norm(vec_tx_to_target), 1e-6);
    theta_tx = atan2(vec_tx_to_target(2), vec_tx_to_target(1));

    vec_rx_to_target = target_pos_xy - rx_pos_xy;
    r_rx = max(norm(vec_rx_to_target), 1e-6);
    vec_target_to_rx = rx_pos_xy - target_pos_xy;
    aoa_gcs_rad = atan2(vec_target_to_rx(2), vec_target_to_rx(1));
    aoa_lcs_rad = aoa_gcs_rad - deg2rad(rx_orientation_deg);

    % Bistatic radar-equation amplitude and Doppler velocity
    alpha = sqrt(lambda ^ 2 * target_rcs / ((4 * pi) ^ 3)) / (r_tx * r_rx);
    v_comp_tx = dot(target_vel_xy, vec_tx_to_target / r_tx);
    v_comp_rx = dot(target_vel_xy, vec_rx_to_target / r_rx);
    doppler_vel = v_comp_tx + v_comp_rx;
    common_gain = sqrt(Ptx * Ge_t * Ge_r) * alpha;

    % Frequency-dependent steering vectors (beam squint is modeled)
    f_start = fc - (Nc - 1) / 2 * fdelta;
    fm_vec = f_start + (0:Nc - 1) * fdelta;
    n_tx = ((1:Nt)' - (Nt + 1) / 2);
    n_rx = ((1:Nr)' - (Nr + 1) / 2);
    A_tx = exp(1j * ((2 * pi / c0) * d_tx * sin(theta_tx)) * (n_tx * fm_vec));
    A_rx = exp(1j * ((2 * pi / c0) * d_rx * sin(aoa_lcs_rad)) * (n_rx * fm_vec));

    range_phase = exp(-1j * 2 * pi * (r_tx + r_rx) / c0 * fm_vec);
    doppler_phase = exp((-1j * 2 * pi * fc * (doppler_vel / c0) * Ts) * (0:Ns - 1)');
    phase = doppler_phase * range_phase;

    H_spatial = pagemtimes(reshape(A_rx, [Nr, 1, Nc]), pagetranspose(reshape(A_tx, [Nt, 1, Nc])));
    H_echo = common_gain * reshape(H_spatial, [Nr, Nt, 1, Nc]) .* reshape(phase, [1, 1, Ns, Nc]);
end
