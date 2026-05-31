%% =====================================================================
%  cTDBG 能带绘图 — 单图版,自动保存 PNG / EPS / PDF
% =====================================================================
clear; clc; close all;

%% ---------- 1. 定位 & 加载数据 ----------
search_dir   = 'F:\cTDBG能带计算';
file_pattern = 'cTDBG_t1_260_D0_000_both_one_*.mat';   % timestamp 用 * 通配

data_file = '';
if exist(search_dir, 'dir')
    d = dir(fullfile(search_dir, '**', file_pattern));
    if ~isempty(d)
        [~, mi]  = max([d.datenum]);   % 同模式多个文件时取最新
        data_file = fullfile(d(mi).folder, d(mi).name);
        fprintf('[自动匹配] %s\n', data_file);
    end
end

if isempty(data_file)
    fprintf('未自动找到文件,请手动选择...\n');
    start_dir = '';
    if exist(search_dir, 'dir'); start_dir = search_dir; end
    [fn, pn] = uigetfile({'*.mat','MAT-files'}, ...
                         '选择 cTDBG 数据文件', start_dir);
    if isequal(fn, 0); error('未选择文件,程序终止.'); end
    data_file = fullfile(pn, fn);
end

S = load(data_file);

%% ---------- 2. 计算横坐标 & 过滤能带 ----------
dk = diff(S.kpath, 1, 1);
xx = [0; cumsum(sqrt(sum(dk.^2, 2)))];
tick_pos = xx(double(S.xticks)+1);

Emin = -60;  Emax = 40;
mask_K  = any(S.bands_K  > Emin-30 & S.bands_K  < Emax+30, 1);
mask_Kp = any(S.bands_Kp > Emin-30 & S.bands_Kp < Emax+30, 1);
fprintf('保留 K  谷:%d / %d 条带\n', nnz(mask_K),  size(S.bands_K, 2));
fprintf('保留 K'' 谷:%d / %d 条带\n', nnz(mask_Kp), size(S.bands_Kp, 2));

%% ---------- 3. 绘图 ----------
fig = figure('Color','w','Units','pixels','Position',[100 100 620 520]);
ax  = axes('Parent', fig);
hold(ax,'on'); box(ax,'on');

p1 = plot(ax, xx, S.bands_K (:, mask_K ), 'k-' , 'LineWidth',1.2);
p2 = plot(ax, xx, S.bands_Kp(:, mask_Kp), 'r--', 'LineWidth',1.2);

axis(ax, [xx(1) xx(end) Emin Emax]);
set(ax, 'XTick', tick_pos, ...
        'XTickLabel', {'K','\Gamma','M','K^{\prime}'}, ...
        'YTick', Emin:20:Emax, ...
        'FontName','Arial', 'FontSize',14, ...
        'LineWidth',1.0, 'TickDir','out', ...
        'TickLength',[0.015 0.015]);
ylabel(ax, 'E[meV]', 'FontSize',16);

% 左上"U = X meV"方框标签 — 根据 param_D 自动生成
u_label = sprintf('\\itU\\rm = %g meV', S.param_D);
text(ax, 0.05, 0.93, u_label, ...
     'Units','normalized', ...
     'FontName','Arial', 'FontSize',13, ...
     'EdgeColor','k', 'LineWidth',0.6, ...
     'BackgroundColor','w', 'Margin',5, ...
     'VerticalAlignment','top', 'HorizontalAlignment','left');

% 右上 K / K' 方框图例
lg = legend(ax, [p1(1) p2(1)], {'K','K^{\prime}'}, ...
            'Location','northeast', 'Box','on', ...
            'FontName','Arial', 'FontSize',12);
lg.EdgeColor = [0 0 0];
lg.LineWidth = 0.6;

% 微调坐标轴位置(留够 ylabel 空间)
set(ax, 'Units','normalized', 'Position',[0.14 0.13 0.82 0.83]);

%% ---------- 4. 导出图片 ----------
% 输出目录:与数据文件同目录(也可以改成自己想要的路径)
[out_dir, basename, ~] = fileparts(data_file);
out_prefix = fullfile(out_dir, [basename '_bands']);

% painters 渲染器是矢量输出的关键(否则 EPS/PDF 会被光栅化)
set(fig, 'Renderer', 'painters');

% 优先用 exportgraphics (MATLAB R2020a+),不可用时回退到 print
try
    exportgraphics(fig, [out_prefix '.png'], 'Resolution', 300);
    exportgraphics(fig, [out_prefix '.eps'], 'ContentType','vector', ...
                   'BackgroundColor','white');
    exportgraphics(fig, [out_prefix '.pdf'], 'ContentType','vector', ...
                   'BackgroundColor','white');
catch
    warning('exportgraphics 不可用(需 R2020a+),已改用 print 命令');
    print(fig, [out_prefix '.png'], '-dpng',   '-r300');
    print(fig, [out_prefix '.eps'], '-depsc2', '-painters');
    print(fig, [out_prefix '.pdf'], '-dpdf',   '-painters', '-bestfit');
end

fprintf('\n已保存:\n  %s.png\n  %s.eps\n  %s.pdf\n', ...
        out_prefix, out_prefix, out_prefix);