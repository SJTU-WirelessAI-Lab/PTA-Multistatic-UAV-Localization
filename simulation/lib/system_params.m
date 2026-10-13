function p = system_params()
% System parameters shared by all scripts (Table I of the paper).

    p.fc = 2.6e9;
    p.c0 = 3e8;
    p.lambda = p.c0 / p.fc;
    p.Nt = 32;
    p.Nr = 32;
    p.d = p.lambda / 2;
    p.Nc = 3276;
    p.Ns = 1;
    p.fdelta = 30e3;
    p.Ts = 35.6e-6;
    p.rcs = 0.1;

    % Thermal noise per subcarrier, split into real/imag parts
    noise_power = 1.38e-23 * 290 * (p.Nc * p.fdelta);
    p.noise_std = sqrt(noise_power / p.Nc / 2);

    % Geometry: Tx at the origin, two Rx on an equilateral triangle with ISD L
    p.L = 200;
    p.pos_tx = [0, 0];
    p.pos_rx1 = [sqrt(3) * p.L / 2, p.L / 2];
    p.pos_rx2 = [sqrt(3) * p.L / 2, -p.L / 2];
    p.orientation_rx = [240, 120];
    p.area = struct('shape', 'hexagon', 'vertices', [p.pos_tx; p.pos_rx1; p.pos_rx2]);
    p.velocity_range = struct('vx', [0, 0], 'vy', [0, 0]);

    % Wide Tx beam over [-60, 60] deg; the PA receiver uses the same combiner
    p.v_tx = AMCF_ZCI(p.Nt, -sqrt(3) / 2, sqrt(3), 512, 1000);
    p.w_pa_rx = AMCF_ZCI(p.Nr, -sqrt(3) / 2, sqrt(3), 512, 1000);
    p.s_data = ones(p.Nc, p.Ns) / sqrt(p.Nc);

    % PTA rainbow beam coverage (local coordinates)
    p.theta_start_deg = -60;
    p.theta_end_deg = 60;

    % Estimators
    p.range_window = [p.L, 2 * p.L];
    p.range_fft_n = 16384;
    p.pta_subband_half_width = 50;
    p.fd_angle_fft_n = 512;

    % Error std of the PTA estimator per link, [sigma_d1, sigma_d2] (m) and
    % [sigma_theta1, sigma_theta2] (deg), calibrated at each Ptx (calibrate_sigma.m)
    p.sigma_ptx = [42, 44, 46, 48, 50, 52];
    p.sigma_d = [26.7105043399031, 25.8468257538736;
                 18.5479011520564, 18.2660233205375;
                 11.1793679979854, 10.8479231022012;
                 5.35177231101393, 5.58103280370885;
                 2.17827010309565, 2.4019845870298;
                 0.9181648955, 0.9152479166];
    p.sigma_theta_deg = [13.9975664328558, 13.3831564191466;
                         9.48823499168497, 9.0138531304009;
                         5.70169724671074, 5.10053839743319;
                         2.71335858069841, 2.46869413033915;
                         1.32857533173049, 0.896431757534122;
                         0.8335711454, 0.4813528494];
end
