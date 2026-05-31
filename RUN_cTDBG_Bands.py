#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
RUN_cTDBG_Bands.py
================================================================================
Compute band structure for ABBA-stacked cTDBG (no WSe2 proximity) and save
publication-quality figures (PNG + EPS) and data files in NPZ + MAT formats.

Valley convention (from cTDBG.py):
    xi = -1  ->  K   (solid line)
    xi = +1  ->  K'  (dashed line)

Energy units:
    cTDBG.py returns eV; this script converts to meV for all output.

Usage:
    python RUN_cTDBG_Bands.py --theta 0.9 --D 0.5 --valley K
    python RUN_cTDBG_Bands.py --theta 0.9 --D 0.5 --valley both --res 600
    python RUN_cTDBG_Bands.py --theta 0.9 --D 0.0 --valley both --cut 6
    python RUN_cTDBG_Bands.py --theta 1.26 --save_mat 1 --save_npz 1

Author: Jiao Xie, 2026.
Email: xiejiao@smail.nju.edu.cn
Version: 1.0 (2026-05-20)
================================================================================
"""

import os
import sys
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# scipy is required for .mat output; fall back gracefully if missing
try:
    import scipy.io as sio
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ─── Model import ───
_MODEL_IMPORTED = False
_MODEL_SOURCE = None
TDBGModel = None

try:
    from cTDBG import TDBGModel
    _MODEL_IMPORTED = True
    _MODEL_SOURCE = "cTDBG (module)"
except ImportError:
    pass

if not _MODEL_IMPORTED:
    try:
        import importlib.util
        fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cTDBG.py")
        if os.path.exists(fpath):
            spec = importlib.util.spec_from_file_location("cTDBG_Model", fpath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            TDBGModel = module.TDBGModel
            _MODEL_IMPORTED = True
            _MODEL_SOURCE = "cTDBG.py (local)"
    except Exception:
        pass

if not _MODEL_IMPORTED:
    print("[ERROR] Cannot import TDBGModel from cTDBG.py")
    sys.exit(1)


# ─── Utilities ───

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def format_ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def parse_valley(valley):
    s = str(valley).strip().lower()
    if s in ("both", "all"): return [-1, +1]
    if s in ("k", "solid", "-", "-1"): return [-1]
    if s in ("kp", "k'", "kprime", "dashed", "+", "+1"): return [+1]
    raise ValueError(f"Cannot parse valley='{valley}'")

def valley_label(xi):
    return "K" if int(xi) == -1 else "Kp"

def path_ticks_labels(path, res, nk_total):
    p = str(path).strip().lower()
    if p in ("one", "1"):
        l1, l2 = int(res), int(np.sqrt(3)*res/2)
        return [0, l1, l1+l2, nk_total-1], ["K", r"$\Gamma$", "M", r"$K'$"]
    if p in ("two", "2"):
        l1, l2 = int(res/2), int(np.sqrt(3)*res/2)
        return [0, l1, l1+l2, nk_total-1], [r"$K_1$", "M", r"$\Gamma$", r"$K_2$"]
    if p in ("three", "3"):
        l1, l2 = int(res), int(np.sqrt(3)*res/2)
        return [0, l1, l1+l2, nk_total-1], [r"$K_s$", r"$\Gamma_s$", r"$M_s$", r"$K_s$"]
    raise ValueError(f"Unknown path: {path}")


# ─── Core computation ───

def compute_bands(model, res=300, path="one", valley="both"):
    p = str(path).strip().lower()
    dispatch = {"one": model.solve_along_path_one,
                "1": model.solve_along_path_one,
                "two": model.solve_along_path_two,
                "2": model.solve_along_path_two,
                "three": model.solve_along_path_three,
                "3": model.solve_along_path_three}
    func = dispatch.get(p)
    if func is None:
        raise ValueError(f"Unknown path: {path}")
    evals_m, evals_p, kpath = func(res=res, plot_it=False, valley=valley)

    bands_mev = {}
    if evals_m is not None and getattr(evals_m, "size", 0) > 0:
        bands_mev[-1] = np.asarray(evals_m, dtype=float) * 1000.0
    if evals_p is not None and getattr(evals_p, "size", 0) > 0:
        bands_mev[+1] = np.asarray(evals_p, dtype=float) * 1000.0

    kpath = np.asarray(kpath, dtype=float)
    xticks, labels = path_ticks_labels(path, res, kpath.shape[0])
    return bands_mev, kpath, xticks, labels


def compute_cnp(bands_mev):
    """Estimate CNP Fermi level (meV) from half-filling."""
    all_E = []
    for E in bands_mev.values():
        all_E.append(E)
    if not all_E:
        return 0.0, 0.0, 0.0
    E_all = np.concatenate(all_E, axis=1)
    n_occ = E_all.shape[1] // 2
    E_sorted = np.sort(E_all, axis=1)
    vbm = np.max(E_sorted[:, n_occ - 1])
    cbm = np.min(E_sorted[:, n_occ])
    return 0.5 * (vbm + cbm), vbm, cbm


# ─── Plot ───

def plot_bands(bands_mev, kpath, xticks, labels, params,
               energy_ylim=(-60, 60), show_cnp=True,
               out_png=None, out_eps=None, dpi=300):
    Nk = kpath.shape[0]
    x = np.arange(Nk, dtype=float)

    fig, ax = plt.subplots(figsize=(6.0, 4.5))

    colors = {-1: '#2E75B6', +1: '#E24B4A'}
    lstyles = {-1: '-', +1: '--'}
    vlabels_map = {-1: '$K$', +1: "$K'$"}

    for xi in sorted(bands_mev.keys()):
        E = bands_mev[xi]
        for n in range(E.shape[1]):
            lbl = vlabels_map[xi] if n == 0 else None
            ax.plot(x, E[:, n], ls=lstyles[xi], lw=0.7, color=colors[xi],
                    alpha=0.85, label=lbl)

    if show_cnp and bands_mev:
        try:
            E_cnp, vbm, cbm = compute_cnp(bands_mev)
            ax.axhline(E_cnp, color='red', ls='-', lw=1.2, alpha=0.7,
                       label=f'CNP: {E_cnp:.2f} meV')
            gap = cbm - vbm
            if gap > 0.01:
                ax.axhspan(vbm, cbm, color='yellow', alpha=0.12, zorder=0)
        except Exception:
            pass

    ax.set_xlim(0, Nk - 1)
    ax.set_ylim(*energy_ylim)
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Energy (meV)", fontsize=12)
    for t in xticks[1:-1]:
        ax.axvline(t, lw=0.5, color='gray', alpha=0.4)
    ax.axhline(0.0, lw=0.6, ls=':', color='gray', alpha=0.5)

    theta_deg = params.get('theta', 0)
    D_val = params.get('D', 0)
    ax.set_title(f"ABBA cTDBG  $\\theta$ = {theta_deg:.3f}$^\\circ$  "
                 f"D = {D_val:.3f} V/nm", fontsize=11)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)

    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
        print(f"  [PNG] {out_png}")
    if out_eps:
        fig.savefig(out_eps, format='eps', bbox_inches='tight')
        print(f"  [EPS] {out_eps}")
    plt.close(fig)
    return fig


# ─── NPZ save ───

def save_npz(filepath, bands_mev, kpath, xticks, labels, params):
    data = {'kpath': kpath,
            'xticks': np.array(xticks, dtype=int),
            'labels': np.array(labels, dtype=str)}
    for xi, E in bands_mev.items():
        data[f'bands_{valley_label(xi)}'] = E
    for k, v in params.items():
        try:
            data[f'param_{k}'] = np.array(v)
        except Exception:
            data[f'param_{k}'] = np.array(str(v))
    data['model_source'] = np.array(_MODEL_SOURCE or "unknown")
    data['timestamp'] = np.array(format_ts())
    data['version'] = np.array("2.0")
    np.savez_compressed(filepath, **data)
    print(f"  [NPZ] {filepath}")


# ─── MAT save (MATLAB v5) ───

def save_mat(filepath, bands_mev, kpath, xticks, labels, params):
    """
    Save band data in MATLAB .mat (v5) format.

    Variables in the resulting file (MATLAB-side):
        kpath           Nk x 2 double             k-points
        xticks          4 x 1 int64               0-based tick positions
        labels          4 x 1 cell of strings     tick labels (LaTeX-ready)
        bands_K         Nk x Nbands double        valley xi = -1 (if computed)
        bands_Kp        Nk x Nbands double        valley xi = +1 (if computed)
        param_<name>    scalar / string           every run parameter
        model_source    string
        timestamp       string                    "YYYYmmdd_HHMMSS"
        version         string

    MATLAB usage:
        >> S = load('<file>.mat');
        >> figure; hold on
        >> plot(S.bands_K,  'b-');
        >> plot(S.bands_Kp, 'r--');
        >> set(gca, 'XTick', double(S.xticks)+1, 'XTickLabel', S.labels);
        >> ylabel('Energy (meV)');
    """
    if not _HAS_SCIPY:
        print("  [WARN] scipy not installed; skipping .mat output.")
        print("         install with:  pip install scipy")
        return

    # Ensure correct extension
    if not filepath.endswith('.mat'):
        filepath += '.mat'

    data = {
        'kpath':  kpath,
        'xticks': np.array(xticks, dtype=int),
        # object array -> MATLAB cell array of strings
        'labels': np.array(labels, dtype=object),
    }

    # Per-valley band data
    for xi, E in bands_mev.items():
        data[f'bands_{valley_label(xi)}'] = E

    # Parameter scalars / strings as top-level variables (param_<name>)
    for k, v in params.items():
        try:
            data[f'param_{k}'] = np.array(v)
        except Exception:
            data[f'param_{k}'] = np.array(str(v))

    # Metadata
    data['model_source'] = _MODEL_SOURCE or "unknown"
    data['timestamp']    = format_ts()
    data['version']      = "2.0"

    sio.savemat(filepath, data, do_compression=True, oned_as='column')
    print(f"  [MAT] {filepath}")


# ─── CLI ───

def build_argparser():
    p = argparse.ArgumentParser(
        description="ABBA cTDBG band structure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("Geometry")
    g.add_argument("--theta", type=float, default=1.33)
    g.add_argument("--phi", type=float, default=0)
    g.add_argument("--epsilon", type=float, default=5e-4)
    g.add_argument("--D", type=float, default=20.0)
    g.add_argument("--use_D_as_field", type=int, default=0)

    c = p.add_argument_group("Continuum")
    c.add_argument("--a", type=float, default=0.246)
    c.add_argument("--beta", type=float, default=3.14)
    c.add_argument("--delta", type=float, default=0.16)
    c.add_argument("--vf", type=float, default=1.0)
    c.add_argument("--u", type=float, default=79.7, help="meV")
    c.add_argument("--up", type=float, default=97.5, help="meV")
    c.add_argument("--cut", type=float, default=4)
    c.add_argument("--dt_nm", type=float, default=1.005)
    c.add_argument("--eps_s", type=float, default=3.0)

    r = p.add_argument_group("Run")
    r.add_argument("--path", type=str, default="one", choices=["one", "two", "three"])
    r.add_argument("--res", type=int, default=200)
    r.add_argument("--valley", type=str, default="both")
    r.add_argument("--emin", type=float, default=-60.0)
    r.add_argument("--emax", type=float, default=60.0)

    o = p.add_argument_group("Output")
    o.add_argument("--out_dir", type=str, default="./cTDBG_bands(theta=1.33deg)")
    o.add_argument("--out_prefix", type=str, default="")
    o.add_argument("--dpi", type=int, default=600)
    o.add_argument("--save_eps", type=int, default=1, help="Save EPS (1/0)")
    o.add_argument("--save_npz", type=int, default=1, help="Save NPZ (1/0)")
    o.add_argument("--save_mat", type=int, default=1, help="Save MAT (1/0)")   # NEW
    o.add_argument("--show_cnp", type=int, default=1, help="Annotate CNP (1/0)")
    return p


def main():
    args = build_argparser().parse_args()
    ensure_dir(args.out_dir)

    # 1. Model
    model = TDBGModel(
        theta=args.theta, phi=args.phi, epsilon=args.epsilon,
        D=args.D if args.use_D_as_field else args.D/1000.0,
        a=args.a, beta=args.beta, delta=args.delta,
        vf=args.vf, u=args.u/1000.0, up=args.up/1000.0, cut=args.cut,
        dt_nm=args.dt_nm, eps_s=args.eps_s,
        use_D_as_field=bool(args.use_D_as_field))

    # 2. Parameter dictionary
    params = dict(theta=args.theta, phi=args.phi, epsilon=args.epsilon,
                  D=args.D, use_D_as_field=args.use_D_as_field,
                  a=args.a, vf=args.vf, u_meV=args.u, up_meV=args.up,
                  cut=args.cut, dt_nm=args.dt_nm, eps_s=args.eps_s,
                  res=args.res, path=args.path, valley=args.valley,
                  Nq=model.Nq)

    # 3. Output filename
    ts = format_ts()
    prefix = args.out_prefix.strip() or \
        f"cTDBG_t{args.theta:.3f}_D{args.D:.3f}_{args.valley}_{args.path}"
    base = os.path.join(args.out_dir, f"{prefix}_{ts}")

    # 4. Banner
    print("=" * 65)
    print(f"  ABBA cTDBG Band Structure (v2.0)")
    print(f"  theta={args.theta} deg, D={args.D} V/nm, valley={args.valley}")
    print(f"  Nq={model.Nq}, cut={args.cut}, res={args.res}")
    print("=" * 65)

    # 5. Compute bands
    t0 = time.time()
    bands_mev, kpath, xticks, labels = compute_bands(
        model, res=args.res, path=args.path, valley=args.valley)
    print(f"  Computed in {time.time()-t0:.1f} s")

    for xi, E in bands_mev.items():
        print(f"  {valley_label(xi)}: {E.shape[1]} bands, "
              f"E in [{E.min():.2f}, {E.max():.2f}] meV")

    # 6. Plot
    plot_bands(bands_mev, kpath, xticks, labels, params,
               energy_ylim=(args.emin, args.emax),
               show_cnp=bool(args.show_cnp),
               out_png=base + ".png",
               out_eps=base + ".eps" if args.save_eps else None,
               dpi=args.dpi)

    # 7. Data files
    if args.save_npz:
        save_npz(base + ".npz", bands_mev, kpath, xticks, labels, params)
    if args.save_mat:
        save_mat(base + ".mat", bands_mev, kpath, xticks, labels, params)

    print(f"\n  Done. Output: {args.out_dir}/")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())