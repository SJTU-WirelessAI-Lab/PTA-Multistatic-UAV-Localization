function [target_pos, target_vel] = generate_random_target(area, vel_range)
% Draw one target uniformly inside the hexagonal ROI and a velocity uniformly in vel_range.
% area.vertices holds the Tx and the two Rx; the hexagon is rebuilt by point reflection.

    if size(area.vertices, 1) == 3
        base = area.vertices;
        c = mean(base, 1);
        all_vertices = [base; c + (c - base(1, :)); c + (c - base(2, :)); c + (c - base(3, :))];
    else
        all_vertices = area.vertices;
    end

    center = mean(all_vertices, 1);
    angles = atan2(all_vertices(:, 2) - center(2), all_vertices(:, 1) - center(1));
    [~, sort_idx] = sort(angles);
    sorted_vertices = all_vertices(sort_idx, :);
    num_vertices = size(sorted_vertices, 1);

    % Pick one of the equal-area sub-triangles, then sample it uniformly
    k = randi(num_vertices);
    V1 = center;
    V2 = sorted_vertices(k, :);
    V3 = sorted_vertices(mod(k, num_vertices) + 1, :);
    r1 = rand();
    r2 = rand();
    if r1 + r2 > 1
        r1 = 1 - r1;
        r2 = 1 - r2;
    end
    target_pos = V1 + r1 * (V2 - V1) + r2 * (V3 - V1);

    vx = vel_range.vx(1) + (vel_range.vx(2) - vel_range.vx(1)) * rand();
    vy = vel_range.vy(1) + (vel_range.vy(2) - vel_range.vy(1)) * rand();
    target_vel = [vx, vy];
end
