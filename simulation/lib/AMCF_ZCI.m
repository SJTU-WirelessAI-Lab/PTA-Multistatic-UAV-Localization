function v = AMCF_ZCI(N, Omega0, B, Q, M)
% Constant-modulus wide beam covering normalized spatial frequencies [Omega0, Omega0 + B],
% designed by alternating minimization (AMCF) from a Zadoff-Chu-like initialization.
%   N: number of antennas, Q: angle grid size, M: number of iterations.

    v = zeros(N, 1);
    for n_init = 1:N
        if mod(N, 2) == 0
            phase = pi * (B * n_init ^ 2 / (2 * N) + n_init * Omega0);
        else
            phase = pi * (B * n_init * (n_init + 1) / (2 * N) + n_init * Omega0);
        end
        v(n_init) = (1 / sqrt(N)) * exp(1j * phase);
    end

    % Steering matrix on the angle grid and the target (flat-top) gain
    Omega = linspace(-1, 1, Q);
    A = zeros(N, Q);
    g = zeros(Q, 1);
    for q_idx = 1:Q
        A(:, q_idx) = exp(1j * pi * (0:N - 1)' * Omega(q_idx)) / sqrt(N);
        if Omega(q_idx) >= Omega0 && Omega(q_idx) <= Omega0 + B
            g(q_idx) = sqrt(2 / B);
        else
            g(q_idx) = 0;
        end
    end

    for m = 1:M
        r = g .* exp(1j * angle(A' * v));
        p = A * r;
        for n = 1:N
            v(n) = p(n) / (sqrt(N) * abs(p(n)));
        end
    end
end
