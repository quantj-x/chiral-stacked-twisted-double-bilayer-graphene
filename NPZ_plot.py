"""
cTDBG 能带绘图 — Nature 期刊风格
基于参考图样式:左上角方框 U 标签 / 右上角方框 K-K' 图例 / 黑实红虚双色能带
支持 .npz 与 .mat 两种数据格式,自动嵌入字体,矢量导出 PDF/EPS/PNG。

Author: Jiao Xie, 2026.
Email: xiejiao@smail.nju.edu.cn
Version: 1.0 (2026-05-20)
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

# ==========================================
# 1. Nature 期刊绘图风格
# ==========================================
mpl.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset':  'custom',
    'mathtext.rm':       'Arial',
    'mathtext.it':       'Arial:italic',
    'mathtext.bf':       'Arial:bold',
    'axes.linewidth':    1.0,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'xtick.top':         True,
    'ytick.right':       True,
    'xtick.major.size':  4,
    'ytick.major.size':  4,
    'xtick.minor.size':  2,
    'ytick.minor.size':  2,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'font.size':         8,
    'axes.labelsize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
    # —— 关键:嵌入 TrueType 字体,EPS/PDF 在 Illustrator/Inkscape 中可编辑 ——
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
})

# ==========================================
# 2. 用户可调参数
# ==========================================
file_path = "cTDBG_t1.260_D0.000_both_one_20260531_035150.npz"
# 同时支持 .mat;若指定路径找不到,会自动在常见目录搜索同名/同前缀文件

E_min, E_max = -60, 40         # 能量窗口 (meV)
E_tick_step  = 20              # y 轴主刻度间隔
E_pad        = 10              # 筛选能带时窗口外的余量,避免边缘曲线被裁掉

color_K  = 'black'
color_Kp = 'red'
line_w   = 1.3

# 左上方框边框颜色 —— 'black' 完全对齐参考图;想换深蓝改成 '#27408B'
ANNOT_BOX_EDGE   = 'black'
U_LABEL_OVERRIDE = None        # 留 None 则自动从数据读 param_D 生成

out_prefix = 'band_structure_nature_new_data1'

# ==========================================
# 3. 文件查找 & 数据读取(支持 .npz / .mat)
# ==========================================
def find_data_file(path):
    """按下面顺序找文件:① 原路径;② 同前缀但换扩展名;③ 常见目录递归搜索."""
    if os.path.isfile(path):
        return path
    stem, ext = os.path.splitext(os.path.basename(path))
    # 备选文件名:同前缀,可能换扩展名,可能下划线/小数点互换
    alt_stems = {stem,
                 stem.replace('.', '_'),     # cTDBG_t1.260_... → cTDBG_t1_260_...
                 stem.replace('_', '.')}     # 反过来
    candidates = []
    for s in alt_stems:
        for e in ('.npz', '.mat'):
            candidates.append(s + e)
    search_roots = ['.', os.path.dirname(os.path.abspath(path)) or '.',
                    os.path.expanduser('~'),
                    r'F:\cTDBG能带计算',
                    '/mnt/user-data/uploads']
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for cand in candidates:
            hits = glob.glob(os.path.join(root, '**', cand), recursive=True)
            if hits:
                return hits[0]
    raise FileNotFoundError(
        f"找不到 {path}。请把数据文件放到脚本同目录,或修改 file_path 变量。")

def load_data(path):
    """统一返回 dict:bands_K/bands_Kp/xticks/labels/param_D/param_use_D_as_field"""
    real_path = find_data_file(path)
    print(f"加载文件: {real_path}")

    if real_path.lower().endswith('.npz'):
        d = np.load(real_path, allow_pickle=True)
        get = lambda k, dflt=None: d[k].item() if k in d.files else dflt
        return {
            'bands_K':              d['bands_K'],
            'bands_Kp':             d['bands_Kp'],
            'xticks':               np.asarray(d['xticks']).astype(int).flatten(),
            'labels':               [str(s) for s in np.asarray(d['labels']).flatten()],
            'param_D':              get('param_D'),
            'param_use_D_as_field': get('param_use_D_as_field', 0),
        }

    if real_path.lower().endswith('.mat'):
        from scipy.io import loadmat
        d = loadmat(real_path)
        return {
            'bands_K':              d['bands_K'],
            'bands_Kp':             d['bands_Kp'],
            'xticks':               d['xticks'].astype(int).flatten(),
            'labels':               ['K', r'$\Gamma$', 'M', r"K$^{\prime}$"],
            'param_D':              float(d['param_D'][0, 0])
                                    if 'param_D' in d else None,
            'param_use_D_as_field': int(d['param_use_D_as_field'][0, 0])
                                    if 'param_use_D_as_field' in d else 0,
        }

    raise ValueError(f"不支持的扩展名: {real_path}")

data = load_data(file_path)

bK     = data['bands_K']
bKp    = data['bands_Kp']
xticks = data['xticks']
labels = data['labels']
x      = np.arange(bK.shape[0])   # 由于三段路径按 |KΓ|:|ΓM|:|MK'|=2:√3:1 比例采样,
                                  # 直接用索引做横坐标,段间长度自动正确

# ---- 自动生成 U / D 标签 ----
if U_LABEL_OVERRIDE is not None:
    field_label = U_LABEL_OVERRIDE
elif data['param_D'] is None:
    field_label = None
else:
    D_raw = float(data['param_D'])
    if data['param_use_D_as_field']:
        field_label = rf'$D$ = {D_raw:g} V/nm'
    else:
        field_label = rf'$U$ = {D_raw:g} meV'

# ==========================================
# 4. 绘图
# ==========================================
fig, ax = plt.subplots(figsize=(3.3, 3.3), dpi=300)

# 带余量地筛能带,避免在窗口边缘被截
inside_K  = np.any((bK  > E_min - E_pad) & (bK  < E_max + E_pad), axis=0)
inside_Kp = np.any((bKp > E_min - E_pad) & (bKp < E_max + E_pad), axis=0)
ax.plot(x, bK [:, inside_K ], color=color_K,  linewidth=line_w)
ax.plot(x, bKp[:, inside_Kp], color=color_Kp, linewidth=line_w, linestyle='--')
print(f"绘制 K  谷:{inside_K.sum()} / {bK.shape[1]} 条带")
print(f"绘制 K' 谷:{inside_Kp.sum()} / {bKp.shape[1]} 条带")

# 如需高对称点竖直辅助线,取消下面两行注释(参考图未画)
# for tk in xticks[1:-1]:
#     ax.axvline(tk, color='0.6', linewidth=0.6, alpha=0.6)

# ==========================================
# 5. 坐标轴 / 图例 / 标注
# ==========================================
ax.set_xlim(xticks[0], xticks[-1])
ax.set_ylim(E_min, E_max)
ax.set_yticks(np.arange(E_min, E_max + 1, E_tick_step))
ax.set_xticks(xticks)
ax.set_xticklabels(labels)
ax.set_ylabel(r'$E$ [meV]')

# 仅 y 轴加次刻度;x 轴是高对称点路径,不加
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# —— 右上 K / K' 图例(黑色细边框)——
custom_lines = [
    Line2D([0], [0], color=color_K,  lw=1.5),
    Line2D([0], [0], color=color_Kp, lw=1.5, linestyle='--'),
]
leg = ax.legend(custom_lines, ['K', "K'"], loc='upper right',
                handlelength=1.8, frameon=True, framealpha=1.0,
                edgecolor='black', fancybox=False)
leg.get_frame().set_linewidth(0.8)

# —— 左上 U / D 标签(白底细边框)——
if field_label:
    ax.text(0.04, 0.95, field_label, transform=ax.transAxes,
            ha='left', va='top', fontsize=8,
            bbox=dict(boxstyle='square,pad=0.35', facecolor='white',
                      edgecolor=ANNOT_BOX_EDGE, linewidth=0.8))

plt.tight_layout()

# ==========================================
# 6. 导出 PDF / PNG / EPS
# ==========================================
plt.savefig(out_prefix + '.pdf', bbox_inches='tight')
plt.savefig(out_prefix + '.png', bbox_inches='tight', dpi=300)
plt.savefig(out_prefix + '.eps', bbox_inches='tight')
print(f"\n已保存:\n  {out_prefix}.pdf\n  {out_prefix}.png\n  {out_prefix}.eps")
plt.show()