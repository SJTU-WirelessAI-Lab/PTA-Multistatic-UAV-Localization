function [Y1, Y2, N1, N2] = simulate_element_signals(target_pos, target_vel, Ptx, p)
% Noiseless element-level echoes at Rx1/Rx2 and the corresponding element noise.
% All receiver architectures are built from the same (Y, N), so they see identical samples.

    H1 = generate_ISAC_Hecho_channel_vec(p.Nt, p.Nr, p.Nc, p.Ns, p.fc, p.lambda, p.d, p.d, p.Ts, ...
        target_pos, target_vel, p.rcs, p.fdelta, p.pos_rx1, 1, 1, p.orientation_rx(1), Ptx);
    H2 = generate_ISAC_Hecho_channel_vec(p.Nt, p.Nr, p.Nc, p.Ns, p.fc, p.lambda, p.d, p.d, p.Ts, ...
        target_pos, target_vel, p.rcs, p.fdelta, p.pos_rx2, 1, 1, p.orientation_rx(2), Ptx);
    Y1 = generate_element_signal(H1, p.s_data, p.v_tx, p.Nc, p.Ns, p.Nr, p.Nt);
    Y2 = generate_element_signal(H2, p.s_data, p.v_tx, p.Nc, p.Ns, p.Nr, p.Nt);
    N1 = p.noise_std * (randn(size(Y1)) + 1j * randn(size(Y1)));
    N2 = p.noise_std * (randn(size(Y2)) + 1j * randn(size(Y2)));
end
