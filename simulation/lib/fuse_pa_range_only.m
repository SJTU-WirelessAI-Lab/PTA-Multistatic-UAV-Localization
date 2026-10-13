function pos_est = fuse_pa_range_only(d1, d2, p)
% Range-only LS for the PA receiver (no AoA): coarse grid search over the ROI,
% then Nelder-Mead refinement of the two bistatic range residuals.

    if isnan(d1) || isnan(d2)
        pos_est = [NaN, NaN];
        return;
    end
    hex = hexagon_vertices(p.area);
    grid_points = hexagonal_grid(hex, p.L, 45, 35);

    score = zeros(size(grid_points, 1), 1);
    for k = 1:size(grid_points, 1)
        score(k) = sum(([d1, d2] - bistatic_ranges(grid_points(k, :), p)) .^ 2);
    end
    [~, best] = min(score);

    cost = @(x) sum(([d1, d2] - bistatic_ranges(x, p)) .^ 2) + outside_penalty(x, hex, p.area);
    opts = optimset('Display', 'off', 'TolX', 1e-6, 'TolFun', 1e-6, 'MaxIter', 300);
    pos_est = fminsearch(cost, grid_points(best, :), opts);
end

function r = bistatic_ranges(pos, p)
    d_tx = norm(pos - p.pos_tx);
    r = [d_tx + norm(pos - p.pos_rx1), d_tx + norm(pos - p.pos_rx2)];
end

function penalty = outside_penalty(pos, hex, area)
    if inpolygon(pos(1), pos(2), hex(:, 1), hex(:, 2))
        penalty = 0;
    else
        penalty = 1e6 * norm(pos - mean(area.vertices, 1)) ^ 2;
    end
end

function hex = hexagon_vertices(area)
    base = area.vertices;
    c = mean(base, 1);
    hex = [base; c + (c - base(1, :)); c + (c - base(2, :)); c + (c - base(3, :))];
    center = mean(hex, 1);
    [~, order] = sort(atan2(hex(:, 2) - center(2), hex(:, 1) - center(1)));
    hex = hex(order, :);
end

function grid_points = hexagonal_grid(hex, L, nx, ny)
    [X, Y] = meshgrid(linspace(0, 2 * sqrt(3) * L / 3, nx), linspace(-L / 2, L / 2, ny));
    in_hex = inpolygon(X(:), Y(:), hex(:, 1), hex(:, 2));
    grid_points = [X(in_hex), Y(in_hex)];
end
