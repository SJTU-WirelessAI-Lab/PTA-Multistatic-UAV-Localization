function Y = rx_observation(arch, Y_elements, N_elements, p)
% Receiver output for one Rx from element-level signal and noise (Nr x Nc x Ns).
%   'FD' : all Nr element signals (Nr x Nc x Ns)
%   'PTA': single RF chain with the rainbow-beam PS/TTD combiner (Nc x Ns)
%   'PA' : single RF chain with a frequency-flat wide-beam combiner (Nc x Ns)

    Y_total = Y_elements + N_elements;
    switch arch
        case 'FD'
            Y = Y_total;
        case 'PTA'
            % PS phases and TTD delays that steer subcarrier 0 to theta_start and
            % the last subcarrier to theta_end
            W = (p.Nc - 1) * p.fdelta;
            f0 = p.fc - W / 2;
            n_rx = ((1:p.Nr)' - (p.Nr + 1) / 2);
            tilde_fm = (0:p.Nc - 1) * p.fdelta;
            phi = - (f0 * n_rx * p.d * sin(deg2rad(p.theta_start_deg))) / p.c0;
            t = - (phi / W) - (((f0 + W) * n_rx * p.d * sin(deg2rad(p.theta_end_deg))) / (W * p.c0));
            W_pta = (1 / sqrt(p.Nr)) * exp(-1j * 2 * pi * phi) .* exp(-1j * 2 * pi * t * tilde_fm);
            Y = shiftdim(sum(conj(W_pta) .* Y_total, 1), 1);
        case 'PA'
            w = p.w_pa_rx(:);
            w = w / max(norm(w), eps);
            Y = shiftdim(sum(conj(w) .* Y_total, 1), 1);
        otherwise
            error('Unknown receiver architecture: %s', arch);
    end
end
