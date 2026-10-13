function pos_est = fuse_gdop_wls(r1, r2, pos_tx, pos_rx1, pos_rx2)
% GDOP-WLS (proposed): GDOP-weighted initialization, then Levenberg-Marquardt on the
% sigma-normalized range and angle residuals of both links.
% r1, r2: structs with d_l, aoa_gcs_deg, sigma_d, sigma_angle_deg.

    pos_init = gdop_init(r1, r2, pos_tx, pos_rx1, pos_rx2);

    persistent opts
    if isempty(opts)
        opts = optimoptions('lsqnonlin', 'Algorithm', 'levenberg-marquardt', 'Display', 'off', ...
            'FunctionTolerance', 1e-6, 'StepTolerance', 1e-6);
    end
    meas = [r1.d_l, deg2rad(r1.aoa_gcs_deg), r2.d_l, deg2rad(r2.aoa_gcs_deg)];
    sigma = [r1.sigma_d, deg2rad(r1.sigma_angle_deg), r2.sigma_d, deg2rad(r2.sigma_angle_deg)];
    pos_est = lsqnonlin(@(x) residuals(x, meas, sigma, pos_tx, pos_rx1, pos_rx2), pos_init, [], [], opts);
end

function res = residuals(pos, meas, sigma, pos_tx, pos_rx1, pos_rx2)
    x = pos(1);
    y = pos(2);
    d_tx = sqrt((x - pos_tx(1)) ^ 2 + (y - pos_tx(2)) ^ 2);
    d1 = d_tx + sqrt((x - pos_rx1(1)) ^ 2 + (y - pos_rx1(2)) ^ 2);
    d2 = d_tx + sqrt((x - pos_rx2(1)) ^ 2 + (y - pos_rx2(2)) ^ 2);
    theta1 = atan2(y - pos_rx1(2), x - pos_rx1(1));
    theta2 = atan2(y - pos_rx2(2), x - pos_rx2(1));
    res = [(meas(1) - d1) / sigma(1);
           (mod(meas(2) - theta1 + pi, 2 * pi) - pi) / sigma(2);
           (meas(3) - d2) / sigma(3);
           (mod(meas(4) - theta2 + pi, 2 * pi) - pi) / sigma(4)];
end
