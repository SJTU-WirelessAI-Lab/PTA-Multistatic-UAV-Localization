function [pos_init, gdop] = gdop_init(r1, r2, pos_tx, pos_rx1, pos_rx2)
% GDOP-weighted initialization: each Rx gives a single-link position from (d, AoA);
% the two are averaged with weights 1/GDOP_i.
% r1, r2: structs with d_l, aoa_gcs_deg, sigma_d, sigma_angle_deg.

    [gdop1, pos1] = link_gdop(r1, pos_tx, pos_rx1);
    [gdop2, pos2] = link_gdop(r2, pos_tx, pos_rx2);
    if gdop1 < 1e-6, gdop1 = 1e-6; end
    if gdop2 < 1e-6, gdop2 = 1e-6; end
    w1 = 1 / gdop1;
    w2 = 1 / gdop2;
    w_sum = w1 + w2;
    pos_init = (w1 / w_sum) * pos1 + (w2 / w_sum) * pos2;
    gdop = [gdop1, gdop2];
end

function [gdop, pos] = link_gdop(r, pos_tx, pos_rx)
    sigma_theta = deg2rad(r.sigma_angle_deg);
    pos = position_from_link(pos_tx, pos_rx, r.d_l, deg2rad(r.aoa_gcs_deg));

    % Jacobian of (bistatic range, AoA) w.r.t. position, weighted by the measurement errors
    x = pos(1); y = pos(2);
    R1 = max(sqrt((x - pos_tx(1)) ^ 2 + (y - pos_tx(2)) ^ 2), 1e-3);
    R2 = max(sqrt((x - pos_rx(1)) ^ 2 + (y - pos_rx(2)) ^ 2), 1e-3);
    H = [(x - pos_tx(1)) / R1 + (x - pos_rx(1)) / R2, (y - pos_tx(2)) / R1 + (y - pos_rx(2)) / R2;
         -(y - pos_rx(2)) / R2 ^ 2, (x - pos_rx(1)) / R2 ^ 2];
    J = H' * (diag([r.sigma_d ^ 2, sigma_theta ^ 2]) \ H);

    % GDOP = sqrt(trace(J^-1))
    det_J = J(1, 1) * J(2, 2) - J(1, 2) * J(2, 1);
    val = (J(1, 1) + J(2, 2)) / det_J;
    if abs(det_J) < 1e-15 || isnan(det_J) || val < 0
        gdop = 1e9;
    else
        gdop = sqrt(val);
    end
end

function pos = position_from_link(pos_tx, pos_rx, d_l, theta)
    % Intersection of the AoA ray from Rx with the bistatic range ellipse
    u = [cos(theta), sin(theta)];
    P = pos_rx - pos_tx;
    L_sq = sum(P .^ 2);
    if d_l < sqrt(L_sq) + 0.1
        d_l = sqrt(L_sq) + 0.1;
    end
    denominator = 2 * (d_l + sum(P .* u));
    if abs(denominator) < 1e-5
        R_rx = 0;
    else
        R_rx = (d_l ^ 2 - L_sq) / denominator;
    end
    pos = pos_rx + R_rx * u;
end
