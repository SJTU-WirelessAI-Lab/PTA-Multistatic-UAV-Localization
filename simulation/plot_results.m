% Figures 3-5 of the paper from a run_simulation.m result file.
%   Fig. 3: spatial distribution of the position error (highest Ptx)
%   Fig. 4: CDF of the position error (highest Ptx)
%   Fig. 5: position RMSE versus Ptx

clear; close all;
root = fileparts(mfilename('fullpath'));
result_file = '';              % empty: latest results/sim_*.mat
font_name = 'DejaVu Sans';

if isempty(result_file)
    f = dir(fullfile(root, 'results', 'sim_*.mat'));
    assert(~isempty(f), 'No result file in results/. Run run_simulation.m first.');
    [~, k] = max([f.datenum]);
    result_file = fullfile(f(k).folder, f(k).name);
end
fprintf('Loading %s\n', result_file);
res = load(result_file);
p = res.geometry;
fig_dir = fullfile(root, 'figures');
if ~isfolder(fig_dir), mkdir(fig_dir); end

ip = numel(res.ptx_list);
schemes = {'GI_PL', 'GDOP_PL', 'GDOP_WLS', 'PF_MLP', 'SF_CNN'};
labels = {'GI-PL', 'GDOP-PL', 'GDOP-WLS', 'PF-MLP', 'SF-CNN'};
colors = [0.29 0.42 0.16; 0.20 0.25 0.55; 0.85 0.10 0.10; 0.85 0.15 0.85; 1.00 0.45 0.00];
errors = cell(1, numel(schemes));
for k = 1:numel(schemes)
    errors{k} = vecnorm(res.est.(schemes{k})(:, :, ip) - res.true_pos(:, :, ip), 2, 2);
end

plot_heatmap(res.true_pos(:, :, ip), errors, labels, p, font_name, fullfile(fig_dir, 'ErrorHeatmap.png'));
plot_cdf(errors, labels, colors, font_name, fullfile(fig_dir, 'CDF_PosError.png'));
plot_rmse_vs_ptx(res, font_name, fullfile(fig_dir, 'RMSE_vs_Ptx.png'));

%% Local functions
function plot_heatmap(true_pos, errors, labels, p, font_name, out_file)
    % Mean error at each target position, interpolated on a grid and clipped to the hexagon
    c = mean([p.pos_tx; p.pos_rx1; p.pos_rx2], 1);
    verts = [p.pos_tx; p.pos_rx1; p.pos_rx2; c + (c - p.pos_tx); c + (c - p.pos_rx1); c + (c - p.pos_rx2)];
    [~, order] = sort(atan2(verts(:, 2) - c(2), verts(:, 1) - c(1)));
    verts = verts(order, :);
    boundary = [verts; verts(1, :)];
    x_lim = [min(verts(:, 1)), max(verts(:, 1))];
    y_lim = [min(verts(:, 2)), max(verts(:, 2))];
    [Xq, Yq] = meshgrid(linspace(x_lim(1), x_lim(2), 400), linspace(y_lim(1), y_lim(2), 400));
    in_hex = inpolygon(Xq, Yq, verts(:, 1), verts(:, 2));
    span = max(diff(x_lim), diff(y_lim));
    half_span = 0.5 * span * 1.08;
    x_plot = mean(x_lim) + [-half_span, half_span];
    y_plot = mean(y_lim) + [-half_span, half_span];
    c_lim = [0.01, 200];

    fig = figure('Position', [80 80 1400 900], 'Visible', 'off');
    n = numel(errors);
    ax = gobjects(n, 1);
    for k = 1:n
        valid = isfinite(errors{k});
        Vq = nan(size(Xq));     % blank panel if the scheme was not simulated
        if any(valid)
            [tp, ~, ic] = unique(true_pos(valid, :), 'rows');
            F = scatteredInterpolant(tp(:, 1), tp(:, 2), accumarray(ic, errors{k}(valid), [], @mean), 'natural', 'none');
            Vq = F(Xq, Yq);
            Vq(~in_hex) = NaN;
        end

        ax(k) = subplot(2, 3, k);
        hold(ax(k), 'on');
        surf(ax(k), Xq, Yq, Vq, 'EdgeColor', 'none');
        view(ax(k), 2);
        shading(ax(k), 'interp');
        colormap(ax(k), 'jet');
        set(ax(k), 'ColorScale', 'log');
        caxis(ax(k), c_lim);
        z_top = max([Vq(:); 0], [], 'omitnan') + 200;
        plot3(ax(k), boundary(:, 1), boundary(:, 2), (z_top - 100) * ones(size(boundary, 1), 1), '-w', 'LineWidth', 1.5);
        h = [draw_node(ax(k), p.pos_tx, 180, [1, 0.5, 0], 'Tx', span, z_top), ...
             draw_node(ax(k), p.pos_rx1, 60, 'g', 'Rx1', span, z_top), ...
             draw_node(ax(k), p.pos_rx2, 60, 'm', 'Rx2', span, z_top)];
        hold(ax(k), 'off');
        text(ax(k), 0.05, 0.95, labels{k}, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
            'FontSize', 16, 'FontWeight', 'bold', 'FontName', font_name);
        box(ax(k), 'on');
        set(ax(k), 'FontSize', 16, 'FontName', font_name);
        axis(ax(k), 'equal');
        xlim(ax(k), x_plot);
        ylim(ax(k), y_plot);
        if k == 1
            set(ax(k), 'XTickLabel', [], 'YTickLabel', []);
        elseif k < n
            set(ax(k), 'YTickLabel', []);
            xlabel(ax(k), 'x (m)', 'FontSize', 16, 'FontName', font_name);
        else
            set(ax(k), 'YAxisLocation', 'right');
            xlabel(ax(k), 'x (m)', 'FontSize', 16, 'FontName', font_name);
            ylabel(ax(k), 'y (m)', 'FontSize', 16, 'FontName', font_name);
        end
        if k == 3
            lgd = legend(ax(k), h, 'Location', 'southwest', 'FontSize', 14, 'FontName', font_name);
            lgd.BoxFace.ColorType = 'truecoloralpha';
            lgd.BoxFace.ColorData = uint8([255; 255; 255; 128]);
        end
    end

    % Layout: the two baselines stacked on the left, three large panels, shared color bar
    ratio = 1400 / 900;
    gap = 0.02; cb_w = 0.015; inner = 0.005; left = 0.06;
    d_w = (1 - left - 0.11 - cb_w - gap - 2 * gap - 0.01 - 3 * inner) / 7;
    box_w = 2 * d_w + inner;
    d_h = d_w * ratio;
    box_h = box_w * ratio;
    y0 = 0.5 - box_h / 2 + 0.03;
    x1 = left + cb_w + gap;
    x2 = x1 + d_w + gap;
    x3 = x2 + box_w + gap;
    x4 = x3 + box_w + 0.01;
    pos = {[x1, y0 + d_h + inner * ratio, d_w, d_h], [x1, y0, d_w, d_h], ...
           [x2, y0, box_w, box_h], [x3, y0, box_w, box_h], [x4, y0, box_w, box_h]};
    for k = 1:n
        set(ax(k), 'Position', pos{k});
    end
    cb_ax = axes(fig, 'Position', [left, y0, cb_w, box_h], 'Visible', 'off');
    cb = colorbar(cb_ax, 'Position', [left, y0, cb_w, box_h]);
    colormap(cb_ax, 'jet');
    set(cb_ax, 'ColorScale', 'log');
    caxis(cb_ax, c_lim);
    cb.Label.String = 'Positioning Error (m)';
    cb.Label.FontSize = 16;
    cb.Label.FontName = font_name;
    cb.FontSize = 16;
    cb.FontName = font_name;

    exportgraphics(fig, out_file, 'Resolution', 600);
    close(fig);
    fprintf('Saved %s\n', out_file);
end

function h = draw_node(ax, pos, angle_deg, color, name, span, z)
    a = deg2rad(angle_deg + [0, 120, 240]);
    s = 0.02 * span;
    fill3(ax, pos(1) + s * cos(a), pos(2) + s * sin(a), z * [1 1 1], color, ...
        'EdgeColor', 'k', 'LineWidth', 1, 'HandleVisibility', 'off');
    h = plot3(ax, NaN, NaN, NaN, '^', 'MarkerSize', 10, 'MarkerFaceColor', color, ...
        'MarkerEdgeColor', 'k', 'DisplayName', name);
end

function plot_cdf(errors, labels, colors, font_name, out_file)
    % Legend ordered by CDF at 5 m (best first); dashed drop lines mark the 90th percentile
    errs = cellfun(@(e) sort(e(isfinite(e))), errors, 'UniformOutput', false);
    [~, order] = sort(cellfun(@(e) mean(e <= 5), errs), 'descend', 'MissingPlacement', 'last');
    order = order(~cellfun(@isempty, errs(order)));     % skip schemes that were not simulated

    fig = figure('Position', [100 100 640 460], 'Visible', 'off');
    hold on; grid on; box on;
    for k = order
        plot(errs{k}, (1:numel(errs{k}))' / numel(errs{k}), '-', 'Color', colors(k, :), ...
            'LineWidth', 1.8, 'DisplayName', labels{k});
    end
    yline(0.9, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.0, 'HandleVisibility', 'off');
    for k = order
        p90 = percentile(errs{k}, 90);
        plot([p90 p90], [0 0.9], '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.0, 'HandleVisibility', 'off');
    end
    xlabel('Position Error (m)', 'FontSize', 14, 'FontName', font_name);
    ylabel('CDF', 'FontSize', 14, 'FontName', font_name);
    lgd = legend('Location', 'southeast', 'FontSize', 13, 'FontName', font_name);
    lgd.BoxFace.ColorType = 'truecoloralpha';
    lgd.BoxFace.ColorData = uint8([255; 255; 255; 128]);
    xlim([0, 10]);
    xticks(0:10);
    ylim([0, 1]);
    set(gca, 'FontSize', 13, 'FontName', font_name);
    hold off;

    % Inset: full error range on a log axis
    axes('Position', [0.55 0.45 0.3 0.3]);
    hold on; box on; grid on;
    for k = order
        plot(errs{k}, (1:numel(errs{k}))' / numel(errs{k}), '-', 'Color', colors(k, :), 'LineWidth', 1.2);
    end
    set(gca, 'XScale', 'log', 'FontSize', 13, 'FontName', font_name);
    xlim([1e-2, 1e2]);
    ylim([0, 1]);
    title('Full Scale (Log)', 'FontSize', 14, 'FontWeight', 'normal', 'FontName', font_name);
    hold off;

    exportgraphics(fig, out_file, 'Resolution', 600);
    close(fig);
    fprintf('Saved %s\n', out_file);
end

function plot_rmse_vs_ptx(res, font_name, out_file)
    % Power-matched networks: solid lines; generalized (trained at one Ptx): dashed
    schemes = {'GDOP_WLS', 'GI_PL', 'GDOP_PL', 'PF_MLP', 'PF_MLP_matched', 'SF_CNN', 'SF_CNN_matched'};
    labels = {'GDOP-WLS', 'GI-PL', 'GDOP-PL', 'PF-MLP', 'PF-MLP (Power-Matched)', 'SF-CNN', 'SF-CNN (Power-Matched)'};
    markers = {'o-', 's-', '^-', 'd--', 'd-', 'v--', 'v-'};
    colors = [0.85 0.10 0.10; 0.29 0.42 0.16; 0.20 0.25 0.55; 0.85 0.15 0.85; 0.85 0.15 0.85; 1.00 0.45 0.00; 1.00 0.45 0.00];
    np = numel(res.ptx_list);
    rmse = nan(numel(schemes), np);
    for k = 1:numel(schemes)
        for i = 1:np
            e = vecnorm(res.est.(schemes{k})(:, :, i) - res.true_pos(:, :, i), 2, 2);
            e = e(isfinite(e));
            if ~isempty(e), rmse(k, i) = sqrt(mean(e .^ 2)); end
        end
    end
    % Legend ordered by RMSE at the second highest power (worst on top)
    [~, order] = sort(rmse(:, max(np - 1, 1)), 'descend', 'MissingPlacement', 'last');

    fig = figure('Position', [100 100 640 460], 'Visible', 'off');
    hold on; grid on; box on;
    for k = order'
        valid = isfinite(rmse(k, :));
        if any(valid)
            plot(res.ptx_list(valid), rmse(k, valid), markers{k}, 'Color', colors(k, :), ...
                'LineWidth', 1.8, 'MarkerSize', 7, 'DisplayName', labels{k});
        end
    end
    xlabel('Transmit Power P_{tx} (dBm)', 'FontSize', 14, 'FontName', font_name);
    ylabel('Position RMSE (m)', 'FontSize', 14, 'FontName', font_name);
    lgd = legend('Location', 'northeast', 'FontSize', 13, 'FontName', font_name);
    lgd.BoxFace.ColorType = 'truecoloralpha';
    lgd.BoxFace.ColorData = uint8([255; 255; 255; 128]);
    xticks(res.ptx_list);
    ylim([1e-1, 60]);
    set(gca, 'FontSize', 13, 'FontName', font_name);
    hold off;

    exportgraphics(fig, out_file, 'Resolution', 600);
    close(fig);
    fprintf('Saved %s\n', out_file);
end

function v = percentile(e, pct)
    e = sort(e(isfinite(e)));
    n = numel(e);
    if n == 0
        v = NaN;
        return;
    end
    v = interp1([0; ((1:n)' - 0.5) / n; 1] * 100, [e(1); e; e(end)], pct);
end
