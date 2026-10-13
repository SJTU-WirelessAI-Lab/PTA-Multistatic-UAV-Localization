function pos_est = fuse_pl_weighted(r1, r2, pos_tx, pos_rx1, pos_rx2, init)
% Path-loss-weighted fusion baseline of Liu et al. (absolute range and angle residuals
% weighted by path-loss terms), solved by quasi-Newton from one of two initializations:
%   init = 'ray'  -> GI-PL   (intersection of the two AoA rays)
%   init = 'gdop' -> GDOP-PL (GDOP-weighted initialization, as in GDOP-WLS)

    switch init
        case 'ray'
            pos_init = ray_intersection(r1, r2, pos_tx, pos_rx1, pos_rx2);
        case 'gdop'
            pos_init = gdop_init(r1, r2, pos_tx, pos_rx1, pos_rx2);
        otherwise
            error('Unknown initialization: %s', init);
    end

    meas = [r1.d_l, deg2rad(r1.aoa_gcs_deg), r2.d_l, deg2rad(r2.aoa_gcs_deg)];
    opts = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', 'none');
    pos_est = fminunc(@(x) pl_loss(x, meas, pos_tx, pos_rx1, pos_rx2), pos_init, opts);
end

function loss = pl_loss(p, meas, pos_tx, pos_rx1, pos_rx2)
    p = p(:)';
    d_T = norm(p - pos_tx);
    d_R1 = norm(p - pos_rx1);
    d_R2 = norm(p - pos_rx2);
    if d_T < 1e-6 || d_R1 < 1e-6 || d_R2 < 1e-6
        loss = inf;
        return;
    end
    phi1 = atan2(p(2) - pos_rx1(2), p(1) - pos_rx1(1));
    phi2 = atan2(p(2) - pos_rx2(2), p(1) - pos_rx2(1));

    alpha1 = (d_T ^ -2) * (d_R1 ^ -2);
    alpha2 = (d_T ^ -2) * (d_R2 ^ -2);
    gamma1 = alpha1 * d_R1;
    gamma2 = alpha2 * d_R2;
    dist_err1 = abs(meas(1) - (d_T + d_R1));
    dist_err2 = abs(meas(3) - (d_T + d_R2));
    angle_err1 = abs(wrap_to_pi(phi1 - meas(2)));
    angle_err2 = abs(wrap_to_pi(phi2 - meas(4)));

    loss = (alpha1 * dist_err1 + alpha2 * dist_err2) / (alpha1 + alpha2) ...
        + (gamma1 * angle_err1 + gamma2 * angle_err2) / (gamma1 + gamma2);
end

function a = wrap_to_pi(a)
    if abs(a) > pi
        a = mod(a + pi, 2 * pi) - pi;
    end
end

function pos_init = ray_intersection(r1, r2, pos_tx, pos_rx1, pos_rx2)
    t1 = deg2rad(r1.aoa_gcs_deg);
    t2 = deg2rad(r2.aoa_gcs_deg);
    x1 = pos_rx1(1); y1 = pos_rx1(2);
    x2 = pos_rx2(1); y2 = pos_rx2(2);
    x_init = (y2 - y1 + tan(t1) * x1 - tan(t2) * x2) / (tan(t1) - tan(t2));
    y_init = (x2 - x1 + cot(t1) * y1 - cot(t2) * y2) / (cot(t1) - cot(t2));
    pos_init = [x_init, y_init];

    % Nudge the start point off the nodes to keep the path-loss weights finite
    if norm(pos_init - pos_tx) < 1e-3 || norm(pos_init - pos_rx1) < 1e-3 || norm(pos_init - pos_rx2) < 1e-3
        pos_init = pos_init + [0.01, 0.01];
    end
end
