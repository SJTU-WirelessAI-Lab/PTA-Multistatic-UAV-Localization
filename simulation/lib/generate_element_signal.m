function Y_elements = generate_element_signal(H_echo, s_data, v_tx, Nc, Ns, Nr, Nt)
% Noiseless per-element received signal (Nr x Nc x Ns) for Tx beamformer v_tx and symbols s_data.

    H = reshape(permute(H_echo, [1, 2, 4, 3]), [Nr, Nt, Nc * Ns]);
    H_v = reshape(pagemtimes(H, v_tx), [Nr, Nc, Ns]);
    Y_elements = H_v .* reshape(s_data, [1, Nc, Ns]);
end
