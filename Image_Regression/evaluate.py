"""
Plot how PSNR evolves on the TRAIN and TEST sets DURING training.

Input: the epoch-level traces written by TrainingApp
    checkpoints/<model_name>_psnr_curve.csv
with columns
    grade, epoch, global_epoch, E_T_train, E_T_test,
    mse_train, mse_test, psnr_train_dB, psnr_test_dB, lr.

Outputs (in OUTPUT_DIR):
  * figure_psnr_curves.png      -- one panel per run: PSNR_train and
                                   PSNR_test vs global epoch, grade
                                   boundaries marked;
  * figure_energy_curves.png    -- one panel per run: O_Omega(F_k) vs
                                   global epoch (log scale), when the
                                   trace carries the O_Fk column;
  * figure_psnr_by_kappa.png    -- per (image, N): top row PSNR_test vs
                                   global epoch, one line per kappa
                                   (dashed = its PSNR_train); bottom row
                                   O_Omega(F_k) by kappa when available;
  * psnr_final_summary.csv      -- final PSNR_train / PSNR_test / gap /
                                   O_Omega(F_k) per run, printed as a
                                   table.

Additional outputs for the CURRENT protocol:
  * kappa_search_report.csv / figure_kappa_search.png -- for every
    per-grade kappa-search run (_KSRCH_ in the name, driven by
    KappaSearchApp): all trials (kappa, lambda, PSNR_val, energy) per
    grade with the chosen one marked, plus a figure of chosen kappa_k
    vs grade and the trial clouds;
  * nested_test_report.csv / figure_nested_test.png -- the
    resolution-nested test errors E^(m), m = N, ..., fN, per grade
    (history keys E_T_res<m>), the discrete-testing-error convergence.

Run metadata is parsed from the model name written by main_run.py:
    IMG_<image>_N[..](_f[..])_w[..]_L[..]_K[..]_ep[..]_lr[..]_H[..]
    _act[..](_kappa[..] | _KSRCH_kmin[..]_kmax[..]_tau[..])_seed[..]
Files whose names don't match still plot, keyed by file name.  Runs
are grouped per (image, N, H, factor) cell, so H1 and H2 never mix.
"""

import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CHECKPOINT_DIR = "./checkpoints"
OUTPUT_DIR = "./eval_psnr"

MAX_GRADE = 4      # present only grades <= MAX_GRADE everywhere
                   # (curves, tables, search report, prediction
                   # panels); None -> show all grades.  The
                   # cascade saturates at grade 4, so grades 5-6
                   # of the K=6 runs add nothing to the figures.

# --- Pair comparison (no-reg vs best-reg) config ---------------------------
PAIR_IMAGE = None      # e.g. "fox"; None -> the (image, N) cell with most runs
NOREG_KAPPA = 0.0      # the unregularized baseline
BEST_KAPPA = None      # e.g. 0.01; None -> auto-select by final test PSNR
                       # among runs with kappa > 0


# =============================================================================
# LOADING
# =============================================================================

def parse_run_metadata(name):
    """Parse metadata from a model name; missing fields become NaN/None."""
    def grab(pattern, cast, default):
        m = re.search(pattern, name)
        return cast(m.group(1)) if m else default

    return {
        "image": grab(r"IMG_([A-Za-z0-9]+)_", str, None),
        "N": grab(r"_N\[(\d+)\]", int, np.nan),
        "factor": grab(r"_f\[(\d+)\]", int, 2),
        "noise": grab(r"_ns\[([\d\.eE+-]+)\]", float, 0.0),
        "rs": grab(r"_rs\[(\d+)\]", int, 0),
        "width": grab(r"_w\[(\d+)\]", int, np.nan),
        "layers": grab(r"_L\[(\d+)\]", int, np.nan),
        "num_grades": grab(r"_K\[(\d+)\]", int, np.nan),
        "H": grab(r"_H\[(\d+)\]", int, 1),
        "has_H_tag": re.search(r"_H\[\d+\]", name) is not None,
        "act": grab(r"_act\[([A-Za-z0-9]+)\]", str, None),
        "kappa": grab(r"_kappa\[([\d\.eE+-]+)\]", float, np.nan),
        "ksearch": re.search(r"_KSRCH", name) is not None,
        "tau": grab(r"_tau\[([\d\.eE+-]+)\]", float, np.nan),
        "kmin": grab(r"_kmin\[([\d\.eE+-]+)\]", float, np.nan),
        "kmax": grab(r"_kmax\[([\d\.eE+-]+)\]", float, np.nan),
        "eta": grab(r"_eta\[([\d\.eE+-]+)\]", float, 1.0),
        "epochs": grab(r"_ep\[(\d+)\]", int, np.nan),
        "lr": grab(r"_lr\[([\d\.eE+-]+)\]", float, np.nan),
        "seed": grab(r"_seed\[(\d+)\]", int, np.nan),
    }


def load_search(name):
    """The per-grade kappa-search log written by KappaSearchApp, or
    None: <name>_kappa_search.json with, per grade, every trial's
    (kappa, lambda, psnr_val_dB, energy) and the chosen kappa_k."""
    import json
    path = os.path.join(CHECKPOINT_DIR, f"{name}_kappa_search.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def load_curves(checkpoint_dir):
    """-> list of (run_name, meta dict, curve DataFrame)."""
    runs = []
    for path in sorted(glob.glob(
            os.path.join(checkpoint_dir, "*_psnr_curve.csv"))):
        name = os.path.basename(path)[:-len("_psnr_curve.csv")]
        df = pd.read_csv(path)
        need = {"global_epoch", "psnr_train_dB", "psnr_test_dB", "grade"}
        if df.empty or not need.issubset(df.columns):
            print(f"  (skipping {name}: missing columns)")
            continue
        if MAX_GRADE is not None:
            df = df[df["grade"] <= MAX_GRADE]
            if df.empty:
                continue
        runs.append((name, parse_run_metadata(name), df))
    return runs


def grade_boundaries(df):
    """Global epochs at which a new grade starts (grade > 1)."""
    starts = df.groupby("grade")["global_epoch"].min().sort_index()
    return [int(e) for g, e in starts.items() if g > starts.index.min()]


def has_energy(df):
    return ("O_Fk" in df.columns
            and pd.to_numeric(df["O_Fk"], errors="coerce").notna().any())


def run_title(name, meta):
    parts = []
    if meta["image"]:
        parts.append(meta["image"])
    if pd.notna(meta["N"]):
        parts.append(f"N={int(meta['N'])}")
    if int(meta.get("factor", 2)) != 2:
        parts.append(f"f={int(meta['factor'])}")
    if meta.get("has_H_tag"):
        parts.append(f"H{int(meta['H'])}")
    if meta.get("ksearch"):
        t = (rf" $\tau$={meta['tau']:g}"
             if pd.notna(meta.get("tau", np.nan)) else "")
        parts.append(r"$\kappa$-search" + t)
    elif pd.notna(meta["kappa"]):
        parts.append(rf"$\kappa$={meta['kappa']:g}")
    if pd.notna(meta.get("epochs", np.nan)):
        parts.append(f"ep={int(meta['epochs'])}")
    return " | ".join(parts) if parts else name



def penalty_weight(meta):
    """The penalty weight per unit kappa (lambda / kappa).  Runs with an
    _H[..] tag use TrainingApp's HYPERCUBE constants (d = 2):
        H1: (d/12) h^2 = h^2/6,   H2: ((25 d^2 + 2d)/720) h^4.
    Legacy names without the tag used C_Q^2 h^2 = 2 h^2."""
    N = meta.get("N", np.nan)
    if N is None or (isinstance(N, float) and np.isnan(N)):
        return np.nan
    h = 2.0 / float(N)
    if meta.get("has_H_tag"):
        return (h * h / 6.0 if int(meta.get("H", 1)) == 1
                else (104.0 / 720.0) * h ** 4)
    return 2.0 * h * h


def _energy_col(meta):
    """The energy the run's penalty controls: O_Fk for H1, O2_Fk for H2."""
    return "O_Fk" if int(meta.get("H", 1)) == 1 else "O2_Fk"


def augment_run(meta, df, name=None):
    """Add the note's quantities to an epoch-level trace:
      E_Omega    = E_T_test  (Lemma 1: the fine-grid test error IS
                   ||Pi_{fN}(F_k - I)||^2_{L2(Omega)});
      objective  = E_T_train + lambda * O^(m)(F_k).  For fixed-kappa
                   runs lambda = penalty_weight * kappa; for kappa-search
                   runs lambda is the per-grade chosen lambda from
                   <name>_kappa_search.json.  H2 runs need the O2_Fk
                   trace column (trace_hess_mc > 0), else NaN.
    """
    df = df.copy()
    if "E_T_test" in df.columns:
        df["E_Omega"] = df["E_T_test"]
    ecol = _energy_col(meta)
    have_e = (ecol in df.columns
              and pd.to_numeric(df[ecol], errors="coerce").notna().any())
    df["objective"] = np.nan
    if have_e and "E_T_train" in df.columns:
        if meta.get("ksearch") and name is not None:
            s = load_search(name)
            if s:
                lam = df["grade"].map({g["grade"]: g.get("chosen_lambda",
                                                         np.nan)
                                       for g in s["grades"]})
                df["objective"] = df["E_T_train"] + lam * df[ecol]
        else:
            w, kappa = penalty_weight(meta), meta.get("kappa", np.nan)
            if pd.notna(w) and pd.notna(kappa):
                df["objective"] = (df["E_T_train"]
                                   + w * float(kappa) * df[ecol])
    return df


# =============================================================================
# FIGURE: one panel per run, O_Omega(F_k) across training
# =============================================================================

def plot_energy_curves(runs, output_dir):
    runs = [(n, m, d) for n, m, d in runs if has_energy(d)]
    if not runs:
        print("  (no O_Fk column in any trace -- skipping the energy "
              "figure; re-train with trace_energy_mc > 0)")
        return
    n = len(runs)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.6 * ncols, 4.3 * nrows),
                             squeeze=False)

    for idx, (name, meta, df) in enumerate(runs):
        ax = axes[idx // ncols][idx % ncols]
        df = df.sort_values("global_epoch")
        ax.plot(df["global_epoch"], df["O_Fk"],
                color="#1baf7a", lw=2, label=r"$\mathcal{O}_\Omega(F_k)$")
        for e in grade_boundaries(df):
            ax.axvline(e, color="gray", lw=0.8, ls=":", alpha=0.7)
        last = df.iloc[-1]
        ax.annotate(f"{last['O_Fk']:.3g}",
                    (last["global_epoch"], last["O_Fk"]),
                    textcoords="offset points", xytext=(4, 3),
                    fontsize=8, color="#1baf7a")
        ax.set_yscale("log")
        ax.set_title(run_title(name, meta), fontsize=10, fontweight="bold")
        ax.set_xlabel("global epoch")
        ax.set_ylabel(r"$\mathcal{O}_\Omega(F_k)$")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.legend(fontsize=8, frameon=True, loc="lower right")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(r"Sobolev energy $\mathcal{O}_\Omega(F_k)$ during training "
                 "(dotted lines = grade boundaries)",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(output_dir, "figure_energy_curves.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n{path}")


# =============================================================================
# FIGURE 1: one panel per run, PSNR train vs test across training
# =============================================================================

def plot_psnr_curves(runs, output_dir):
    n = len(runs)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.6 * ncols, 4.3 * nrows),
                             squeeze=False)

    for idx, (name, meta, df) in enumerate(runs):
        ax = axes[idx // ncols][idx % ncols]
        df = df.sort_values("global_epoch")

        ax.plot(df["global_epoch"], df["psnr_train_dB"],
                color="#2a78d6", lw=2, label="PSNR train")
        ax.plot(df["global_epoch"], df["psnr_test_dB"],
                color="#eb6834", lw=2, label="PSNR test")

        for e in grade_boundaries(df):
            ax.axvline(e, color="gray", lw=0.8, ls=":", alpha=0.7)

        last = df.iloc[-1]
        ax.annotate(f"{last['psnr_train_dB']:.1f}",
                    (last["global_epoch"], last["psnr_train_dB"]),
                    textcoords="offset points", xytext=(4, 3),
                    fontsize=8, color="#2a78d6")
        ax.annotate(f"{last['psnr_test_dB']:.1f}",
                    (last["global_epoch"], last["psnr_test_dB"]),
                    textcoords="offset points", xytext=(4, -10),
                    fontsize=8, color="#eb6834")

        parts = []
        if meta["image"]:
            parts.append(meta["image"])
        if pd.notna(meta["N"]):
            parts.append(f"N={int(meta['N'])}")
        if pd.notna(meta["kappa"]):
            parts.append(rf"$\kappa$={meta['kappa']:g}")
        ax.set_title(" | ".join(parts) if parts else name,
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("global epoch", fontsize=10)
        ax.set_ylabel("PSNR (dB)")
        # ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=8, frameon=True, loc="lower right")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("PSNR on train and test during training "
                 "(dotted lines = grade boundaries)",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(output_dir, "figure_psnr_curves.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n{path}")


# =============================================================================
# FIGURE 2: kappa comparison per (image, N) cell
# =============================================================================

def plot_psnr_by_kappa(runs, output_dir):
    cells = {}
    for name, meta, df in runs:
        key = (meta["image"] or name, meta["N"], meta.get("H", 1),
               meta.get("factor", 2), meta.get("noise", 0.0),
               meta.get("rs", 0), meta.get("epochs", np.nan))
        cells.setdefault(key, []).append((meta, df))
    cells = {k: v for k, v in cells.items() if len(v) >= 2}
    if not cells:
        print("  (only one run per (image, N) cell -- skipping the "
              "kappa-comparison figure)")
        return

    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    n = len(cells)
    with_energy = any(has_energy(df) for entries in cells.values()
                      for _, df in entries)
    nrows = 2 if with_energy else 1
    fig, axes = plt.subplots(nrows, n,
                             figsize=(6.0 * n, 4.6 * nrows), squeeze=False)

    for col, (key, entries) in enumerate(sorted(cells.items(),
                                                key=lambda t: str(t[0]))):
        entries = sorted(entries,
                         key=lambda t: (np.isnan(t[0]["kappa"]),
                                        t[0]["kappa"]))
        img, N, H, factor, noise, rs, ep = key
        title = f"{img}" + (f"  (N={int(N)})" if pd.notna(N) else "")
        title += f"  H{int(H)}"
        if int(factor) != 2:
            title += f"  f={int(factor)}"
        if float(noise) > 0:
            title += f"  ns={noise:g}"
        if int(rs):
            title += "  rs"
        if pd.notna(ep):
            title += f"  ep={int(ep)}"

        # Row 1: PSNR
        ax = axes[0][col]
        for i, (meta, df) in enumerate(entries):
            df = df.sort_values("global_epoch")
            c = colors[i % len(colors)]
            lab = _label(meta)
            ax.plot(df["global_epoch"], df["psnr_test_dB"],
                    color=c, lw=2.2, label=lab + " | test")
            ax.plot(df["global_epoch"], df["psnr_train_dB"],
                    color=c, lw=1.4, ls="--", alpha=0.65,
                    label=lab + " | train")
        for e in grade_boundaries(entries[0][1]):
            ax.axvline(e, color="gray", lw=0.8, ls=":", alpha=0.7)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("global epoch")
        ax.set_ylabel("PSNR (dB)")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=7.5, frameon=True, ncol=1, loc="lower right")

        # Row 2: O_Omega(F_k) by kappa
        if with_energy:
            ax = axes[1][col]
            for i, (meta, df) in enumerate(entries):
                if not has_energy(df):
                    continue
                df = df.sort_values("global_epoch")
                c = colors[i % len(colors)]
                ax.plot(df["global_epoch"], df["O_Fk"],
                        color=c, lw=2.2, label=_label(meta))
            for e in grade_boundaries(entries[0][1]):
                ax.axvline(e, color="gray", lw=0.8, ls=":", alpha=0.7)
            ax.set_yscale("log")
            ax.set_xlabel("global epoch")
            ax.set_ylabel(r"$\mathcal{O}_\Omega(F_k)$")
            ax.grid(True, which="both", linestyle="--", alpha=0.3)
            ax.legend(fontsize=7.5, frameon=True, loc="lower right")

    fig.suptitle("PSNR (top; solid = test, dashed = train) and "
                 r"$\mathcal{O}_\Omega(F_k)$ (bottom) during training, "
                 "by $\\kappa$",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "figure_psnr_by_kappa.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n{path}")


# =============================================================================
# FINAL SUMMARY TABLE
# =============================================================================

def _last_hist_val(name, key):
    """The last grade's value of `key` from the run's history (the
    epoch trace does not carry the unseen metric -- it is evaluated
    once per grade by finalize_grade)."""
    h = load_history(name)
    if h is None:
        return np.nan
    d = h[1]
    if key not in d.columns or d.empty:
        return np.nan
    return d.sort_values("grade").iloc[-1][key]


def print_final_summary(runs, output_dir):
    rows = []
    for name, meta, df in runs:
        last = df.sort_values("global_epoch").iloc[-1]
        if meta.get("ksearch"):
            s = load_search(name)
            kappa_str = ("search: " + ",".join(
                f"{g['chosen_kappa']:.3g}" for g in s["grades"])
                if s else "search")
        else:
            kappa_str = (f"{meta['kappa']:g}"
                         if pd.notna(meta["kappa"]) else "")
        rows.append({
            "run": name,
            "image": meta["image"], "N": meta["N"],
            "f": meta.get("factor", 2), "ns": meta.get("noise", 0.0),
            "rs": meta.get("rs", 0), "H": meta.get("H", 1),
            "act": meta.get("act"), "tau": meta.get("tau", np.nan),
            "kappa": meta["kappa"], "kappa_used": kappa_str,
            "seed": meta["seed"],
            "final_psnr_train_dB": last["psnr_train_dB"],
            "final_psnr_val_dB": last.get("psnr_val_dB", np.nan),
            "final_psnr_unseen_dB": _last_hist_val(name,
                                                   "psnr_unseen_dB"),
            "final_psnr_test_dB": last["psnr_test_dB"],
            "train_test_gap_dB": (last["psnr_train_dB"]
                                  - last["psnr_test_dB"]),
            "best_psnr_test_dB": df["psnr_test_dB"].max(),
            "best_test_at_epoch": int(
                df.loc[df["psnr_test_dB"].idxmax(), "global_epoch"]),
            "final_O_Fk": (last["O_Fk"] if has_energy(df) else np.nan),
            "final_E_Omega": last.get("E_T_test", np.nan),
            "final_objective": augment_run(meta, df, name).sort_values(
                "global_epoch").iloc[-1]["objective"],
        })
    out = pd.DataFrame(rows).sort_values(
        ["image", "N", "H", "kappa", "tau"], na_position="last")
    print("\n" + "=" * 90)
    print(" FINAL PSNR SUMMARY (train / test / gap, best test and when)")
    print("=" * 90)
    print(out.to_string(index=False,
                        float_format=lambda v: f"{v:.2f}"))
    path = os.path.join(output_dir, "psnr_final_summary.csv")
    out.to_csv(path, index=False)
    print(f"\nSaved summary to:\n{path}")


# =============================================================================
# PAIR COMPARISON: no-reg (kappa=0) vs best-reg (best kappa)
# =============================================================================

C_NOREG, C_BEST = "#2a78d6", "#eb6834"          # blue / orange


def _slug(title):
    """Filesystem-safe tag from a cell title, for per-cell filenames."""
    return re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")


def select_pair(runs):
    """EVERY (image, N, H, f, ns, rs, ep) cell with a usable pair.
    Returns a list of (title, noreg_run, best_run) -- one entry per
    cell, so runs with different epoch budgets (etc.) each get their
    own pair figures/tables instead of only the most-populated cell."""
    cells = {}
    for r in runs:
        key = (r[1]["image"] or r[0], r[1]["N"], r[1].get("H", 1),
               r[1].get("factor", 2), r[1].get("noise", 0.0),
               r[1].get("rs", 0), r[1].get("epochs", np.nan))
        cells.setdefault(key, []).append(r)
    if PAIR_IMAGE is not None:
        cells = {k: v for k, v in cells.items()
                 if str(k[0]) == str(PAIR_IMAGE)}
        if not cells:
            print(f"  (pair: no runs for image '{PAIR_IMAGE}')")
            return []

    def final_test(r):
        return r[2].sort_values("global_epoch").iloc[-1]["psnr_test_dB"]

    out = []
    for key in sorted(cells, key=str):
        entries = cells[key]
        noreg = [r for r in entries
                 if pd.notna(r[1]["kappa"])
                 and r[1]["kappa"] == NOREG_KAPPA]
        if not noreg:
            cand = [r for r in entries if pd.notna(r[1]["kappa"])]
            if not cand:
                continue
            noreg = sorted(cand, key=lambda r: r[1]["kappa"])[:1]
        noreg = noreg[0]

        # Regularized candidates: fixed kappa > 0 AND per-grade
        # kappa-search runs.
        reg = [r for r in entries
               if r[1].get("ksearch")
               or (pd.notna(r[1]["kappa"])
                   and r[1]["kappa"] > NOREG_KAPPA)]
        if BEST_KAPPA is not None:
            reg = [r for r in reg if pd.notna(r[1]["kappa"])
                   and np.isclose(r[1]["kappa"], BEST_KAPPA)]
        if not reg:
            continue
        best = max(reg, key=final_test)

        img, N, H, factor, noise, rs, ep = key
        title = f"{img}" + (f" (N={int(N)})" if pd.notna(N) else "")
        title += f" H{int(H)}" + (f" f={int(factor)}"
                                  if int(factor) != 2 else "")
        if float(noise) > 0:
            title += f" ns={noise:g}"
        if int(rs):
            title += " rs"
        if pd.notna(ep):
            title += f" ep={int(ep)}"
        print(f"Pair comparison [{title}]: no-reg vs "
              f"{_label(best[1]).replace('$', '')}")
        out.append((title, noreg, best))
    if not out:
        print("  (pair: no cell with a baseline AND a regularized run)")
    return out


def load_history(name):
    """Per-grade history DataFrame from <name>_history.json (or the last
    checkpoint as fallback).  Returns (args_dict, DataFrame) or None."""
    import json
    path = os.path.join(CHECKPOINT_DIR, f"{name}_history.json")
    if os.path.exists(path):
        with open(path) as fh:
            blob = json.load(fh)
        d = pd.DataFrame(blob["history"])
        if MAX_GRADE is not None and "grade" in d.columns:
            d = d[d["grade"] <= MAX_GRADE]
        return blob["args"], d
    ckpts = [p for p in glob.glob(
                 os.path.join(glob.escape(CHECKPOINT_DIR),
                              glob.escape(name) + "_g*.pt"))
             if re.search(r"_g(\d+)\.pt$", p)]   # skip _trial[..] files
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: int(re.search(r"_g(\d+)\.pt$", p).group(1)))
    import torch
    ck = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    d = pd.DataFrame(ck["history"])
    if MAX_GRADE is not None and "grade" in d.columns:
        d = d[d["grade"] <= MAX_GRADE]
    return ck["args"], d


def _label(meta, kappa_only=True):
    if meta.get("ksearch"):
        lab = r"per-grade $\kappa$"
        if pd.notna(meta.get("tau", np.nan)):
            lab += rf" ($\tau$={meta['tau']:g})"
        return lab
    k = meta["kappa"]
    lab = ("no reg" if pd.notna(k) and k == 0
           else (rf"reg with $\kappa$={k:g}" if pd.notna(k) else "reg"))
    eta = meta.get("eta", 1.0)
    if pd.notna(eta) and eta != 1.0:
        lab += rf", $\eta$={eta:g}"
    return lab


def fig_pair_predictions(title, noreg, best, output_dir, tag=""):
    """Fig I-1: per-grade prediction images for the two runs, plus target."""
    try:
        import torch
        from network import FixedDepthMGDL2D
        from dataset import make_train_test
    except ImportError as e:
        print(f"  (pair predictions skipped: {e})")
        return
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_final(name):
        ckpts = glob.glob(os.path.join(glob.escape(CHECKPOINT_DIR),
                                       glob.escape(name) + "_g*.pt"))
        # keep only the per-grade checkpoints _g<k>.pt -- the glob also
        # matches the kappa-search trial files _g<k>_trial[<kappa>].pt
        # written with save_trials=1
        grades = {}
        for p in ckpts:
            m = re.search(r"_g(\d+)\.pt$", p)
            if m:
                grades[int(m.group(1))] = p
        if not grades:
            return None
        return torch.load(grades[max(grades)], map_location=device,
                          weights_only=False)

    ck0, ck1 = load_final(noreg[0]), load_final(best[0])
    if ck0 is None or ck1 is None:
        print("  (pair predictions skipped: checkpoints not found)")
        return
    args = ck0["args"]
    N = int(args["N"])
    factor = int(args.get("train_factor", 2))
    try:
        data = make_train_test(args["image"], N, device, factor=factor)
    except TypeError:              # older dataset.py without `factor`
        data = make_train_test(args["image"], N, device)
        factor = 2
    x_te, I = data["x_test"], data["I"]
    side = factor * N

    def build(ck):
        a = ck["args"]
        m = FixedDepthMGDL2D(width=a["width"], max_grades=a["num_grades"],
                             n_layers=a["layers_per_grade"], in_dim=2,
                             out_dim=3,
                             activation=a.get("activation", "relu"),
                             w0=a.get("siren_w0", 30.0),
                             softplus_beta=a.get("softplus_beta", 10.0),
                             ).to(device)
        # strict=False: checkpoints from before the eta feature carry no
        # grade_scales buffer; the default (all ones) is exactly right
        # for those eta=1 runs.
        m.load_state_dict(ck["state_dict"], strict=False)
        m.eval()
        return m, int(ck["grade"])

    def predict(model, grade):
        outs = []
        with torch.no_grad():
            for i in range(0, x_te.size(0), 65536):
                outs.append(model(x_te[i:i + 65536], current_grade=grade))
        return torch.cat(outs).view(side, side, 3).clamp(0, 1).cpu().numpy()

    m0, K0 = build(ck0)
    m1, K1 = build(ck1)
    K = min(K0, K1)
    if MAX_GRADE is not None:
        K = min(K, MAX_GRADE)
    h0 = load_history(noreg[0])
    h1 = load_history(best[0])

    fig = plt.figure(figsize=(2.3 * (K + 1.35), 2.55 * 2))
    gs = fig.add_gridspec(2, K + 2, wspace=0.05, hspace=0.14,
                          width_ratios=[1.0, 0.28] + [1.0] * K)

    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(I.clamp(0, 1).cpu().numpy())
    ax.set_title("target $\\mathbf{I}$", fontsize=10)
    ax.set_xticks([]), ax.set_yticks([])

    for row, (model, hist, meta) in enumerate(
            [(m0, h0, noreg[1]), (m1, h1, best[1])]):
        for k in range(1, K + 1):
            ax = fig.add_subplot(gs[row, k + 1])
            ax.imshow(predict(model, k))
            t = f"$F_{{{k}}}$"
            if hist is not None:
                g = hist[1][hist[1]["grade"] == k]
                if not g.empty and "psnr_test_dB" in g.columns:
                    t += f"  {g['psnr_test_dB'].iloc[0]:.2f} dB"
            ax.set_title(t, fontsize=9)
            ax.set_xticks([]), ax.set_yticks([])
            if k == 1:
                ax.set_ylabel(_label(meta), fontsize=10)

    fig.suptitle(f"{title}: per-grade reconstructions "
                 f"(top: {_label(noreg[1])}, bottom: {_label(best[1])}; "
                 "titles show test PSNR)", fontsize=12, y=1.0)
    path = os.path.join(output_dir, f"figure_pair_predictions{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n{path}")


def fig_pair_psnr(title, noreg, best, output_dir, tag=""):
    """Fig I-2: train and test PSNR during training, both runs overlaid.
    No title; legend labels are 'no reg' / 'reg with kappa=...'."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for (name, meta, df), c in [(noreg, C_NOREG), (best, C_BEST)]:
        df = df.sort_values("global_epoch")
        lab = _label(meta)
        ax.plot(df["global_epoch"], df["psnr_test_dB"], color=c, lw=2.2,
                label=f"{lab} | test")
        ax.plot(df["global_epoch"], df["psnr_train_dB"], color=c, lw=1.4,
                ls="--", alpha=0.7, label=f"{lab} | train")
    for e in grade_boundaries(noreg[2]):
        ax.axvline(e, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel("global epoch", fontsize=17)
    ax.set_ylabel("PSNR (dB)", fontsize=17)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=14, frameon=True, loc="lower right")
    fig.tight_layout()
    path = os.path.join(output_dir, f"figure_pair_psnr{tag}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n{path}")



def fig_pair_objective(title, noreg, best, output_dir, tag=""):
    """Fig I-4: E_Omega(F_k) (solid) vs the training objective (12)
    (dashed) during training, both runs.  For kappa = 0 the objective
    equals E_T_train, so the dashed no-reg curve doubles as the train
    error; the vertical distance solid-vs-dashed is the generalization
    gap the bound controls."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for (name, meta, df), c in [(noreg, C_NOREG), (best, C_BEST)]:
        d = augment_run(meta, df, name).sort_values("global_epoch")
        if "E_Omega" in d.columns:
            ax.plot(d["global_epoch"], d["E_Omega"], color=c, lw=2.2,
                    label=_label(meta) + r" | $E_\Omega(F_k)$")
        if d["objective"].notna().any():
            ax.plot(d["global_epoch"], d["objective"], color=c, lw=1.4,
                    ls="--", alpha=0.75,
                    label=_label(meta) + " | objective (12)")
    for e in grade_boundaries(noreg[2]):
        ax.axvline(e, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("global epoch")
    ax.set_ylabel("value (integral form)")
    ax.set_title(f"{title}: continuous error vs training objective (12)",
                 fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(fontsize=8.5, frameon=True, loc="upper right")
    fig.tight_layout()
    path = os.path.join(output_dir, f"figure_pair_objective{tag}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n{path}")


def _target_energies(noreg):
    """O(I) at train and test resolution, as reference lines."""
    try:
        import torch
        from dataset import load_image
        h = load_history(noreg[0])
        if h is None:
            return None
        args = h[0]
        N = int(args["N"])
        f = int(args.get("train_factor", 2))
        try:
            I = load_image(args["image"], N, factor=f)
        except TypeError:          # older dataset.py without `factor`
            I = load_image(args["image"], N)
            f = 2
        delta = (2.0 / N) / f

        def energy(img, sp):
            gx = (img[1:] - img[:-1]) / sp
            gy = (img[:, 1:] - img[:, :-1]) / sp
            return ((gx ** 2).sum() + (gy ** 2).sum()).item() * sp ** 2

        return {"O_I_test": energy(I, delta),
                "O_I_train": energy(I[0::f, 0::f], 2.0 / N)}
    except Exception as e:
        print(f"  (target-energy reference skipped: {e})")
        return None


def fig_pair_energy(title, noreg, best, output_dir, tag=""):
    """Fig I-3: O_Omega(F_k) during training, both runs.  No title, no
    O(I) reference lines; legend labels 'no reg' / 'reg with kappa=...'."""
    if not (has_energy(noreg[2]) and has_energy(best[2])):
        print("  (pair energy skipped: no O_Fk column)")
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for (name, meta, df), c in [(noreg, C_NOREG), (best, C_BEST)]:
        df = df.sort_values("global_epoch")
        ax.plot(df["global_epoch"], df["O_Fk"], color=c, lw=2.2,
                label=_label(meta))
    for e in grade_boundaries(noreg[2]):
        ax.axvline(e, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("global epoch", fontsize=17)
    ax.set_ylabel(r"$\mathcal{O}_\Omega(F_k)$", fontsize=17)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(fontsize=14, frameon=True, loc="lower right")
    fig.tight_layout()
    path = os.path.join(output_dir, f"figure_pair_energy{tag}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n{path}")


def tables_pair_per_grade(title, noreg, best, output_dir, tag=""):
    """Table 1: E_T / E_Omega / O(F_k) per grade;  Table 2: train / test
    PSNR per grade -- both for the two runs, printed and saved as CSV and
    LaTeX.  E_Omega here is the fine-grid test error, which by Lemma 1
    equals ||Pi_{2N}(F_k - I)||^2_{L2(Omega)}."""
    h0, h1 = load_history(noreg[0]), load_history(best[0])
    if h0 is None or h1 is None:
        print("  (pair tables skipped: history not found)")
        return
    l0, l1 = _label(noreg[1]), _label(best[1]).replace("$", "")

    refs = _target_energies(noreg)

    def enrich(hist, meta):
        """Per-grade table with the objective and the bound RHS
        3E_T + 3 lambda_k O^(m)(F_k) + (h^2/4) O(f), using the
        a-posteriori kappa_hat and O(I)@test as the stand-in for O(f).
        lambda_k is taken from the history itself (lambda_T /
        lambda_T2 per grade), so fixed-kappa AND per-grade-search runs
        are both handled; the energy and kappa_hat match the run's
        reg order (O_Fk / kappa_hat at H1, O2_Fk / kappa2_hat at H2)."""
        args, d = hist
        d = d.copy()
        N = int(args.get("N", meta.get("N")))
        H = int(args.get("reg_order", meta.get("H", 1)))
        h2 = (2.0 / N) ** 2
        ecol = "O_Fk" if H == 1 else "O2_Fk"
        kcol = "kappa_hat" if H == 1 else "kappa2_hat"
        if ecol not in d.columns:
            ecol = "O_Fk"
        # per-grade lambda from the history (works for search runs too)
        if "lambda_T" in d.columns or "lambda_T2" in d.columns:
            lam = (d.get("lambda_T", 0.0).fillna(0.0)
                   + d.get("lambda_T2", 0.0).fillna(0.0))
        else:
            lam = penalty_weight(meta) * float(args.get("kappa", 0.0))
        d["E_Omega"] = d["E_T_test"]
        d["objective"] = d["E_T_train"] + lam * d[ecol]

        # ---- the RHS, term by term ----------------------------------
        # term1 = 3 E_T_train
        # term2 = 3 lambda_k * O^(m)(F_k)     (the run's own penalty)
        # term3 = (h^2/4) * O(f)              (O(f) ~ O(I)@test)
        d["term1_E_T"] = 3.0 * d["E_T_train"]
        d["term2_pen"] = 3.0 * lam * d[ecol]
        if refs:
            d["term3_Of"] = (h2 / 4.0) * refs["O_I_test"]
            d["RHS"] = d["term1_E_T"] + d["term2_pen"] + d["term3_Of"]
            d["RHS/E_Omega"] = d["RHS"] / d["E_Omega"]

        # bound with the a-posteriori kappa_hat: lambda_hat =
        # (lambda per unit kappa) * kappa_hat
        if refs and kcol in d.columns:
            w = penalty_weight(meta)
            d["bound_RHS"] = (3.0 * d["E_T_train"]
                              + 3.0 * w * d[kcol] * d[ecol]
                              + (h2 / 4.0) * refs["O_I_test"])
            d["bound/E_Omega"] = d["bound_RHS"] / d["E_Omega"]
        return d

    d0, d1 = enrich(h0, noreg[1]), enrich(h1, best[1])

    def col(df, key):
        return (df.set_index("grade")[key] if key in df.columns
                else pd.Series(dtype=float))

    t1 = pd.DataFrame({
        (f"E_T", l0): col(d0, "E_T_train"),
        (f"E_T", l1): col(d1, "E_T_train"),
        (f"E_Omega", l0): col(d0, "E_Omega"),
        (f"E_Omega", l1): col(d1, "E_Omega"),
        (f"objective(12)", l0): col(d0, "objective"),
        (f"objective(12)", l1): col(d1, "objective"),
        (f"O(F_k)", l0): col(d0, "O_Fk"),
        (f"O(F_k)", l1): col(d1, "O_Fk"),
        (f"bound_RHS", l0): col(d0, "bound_RHS"),
        (f"bound_RHS", l1): col(d1, "bound_RHS"),
        (f"bound/E_Omega", l0): col(d0, "bound/E_Omega"),
        (f"bound/E_Omega", l1): col(d1, "bound/E_Omega"),
    })
    t2 = pd.DataFrame({
        ("PSNR_train", l0): col(d0, "psnr_train_dB"),
        ("PSNR_train", l1): col(d1, "psnr_train_dB"),
        ("PSNR_val", l0): col(d0, "psnr_val_dB"),
        ("PSNR_val", l1): col(d1, "psnr_val_dB"),
        ("PSNR_unseen", l0): col(d0, "psnr_unseen_dB"),
        ("PSNR_unseen", l1): col(d1, "psnr_unseen_dB"),
        ("PSNR_test", l0): col(d0, "psnr_test_dB"),
        ("PSNR_test", l1): col(d1, "psnr_test_dB"),
    })

    t3 = pd.DataFrame({
        ("term1 3E_T", l0): col(d0, "term1_E_T"),
        ("term1 3E_T", l1): col(d1, "term1_E_T"),
        ("term2 3*lambda_k*O", l0): col(d0, "term2_pen"),
        ("term2 3*lambda_k*O", l1): col(d1, "term2_pen"),
        ("term3 h2/4*O(f)", l0): col(d0, "term3_Of"),
        ("term3 h2/4*O(f)", l1): col(d1, "term3_Of"),
        ("RHS", l0): col(d0, "RHS"),
        ("RHS", l1): col(d1, "RHS"),
        ("E_Omega", l0): col(d0, "E_Omega"),
        ("E_Omega", l1): col(d1, "E_Omega"),
        ("RHS/E_Omega", l0): col(d0, "RHS/E_Omega"),
        ("RHS/E_Omega", l1): col(d1, "RHS/E_Omega"),
    })

    for t_, tbl, fmt in [("errors_energy", t1, "{:.4e}"),
                         ("bound_terms", t3, "{:.4e}"),
                         ("psnr", t2, "{:.2f}")]:
        print(f"\n===== PER-GRADE TABLE [{t_}] {title} =====")
        print(tbl.to_string(float_format=fmt.format))
        tbl.to_csv(os.path.join(output_dir, f"table_pair_{t_}{tag}.csv"))
        with open(os.path.join(output_dir, f"table_pair_{t_}{tag}.tex"),
                  "w") as fh:
            fh.write(tbl.to_latex(float_format=fmt.format))
    print(f"\nSaved per-grade tables (csv + tex) to {output_dir}")


# =============================================================================
# KAPPA-SEARCH REPORT: per-grade trials and chosen kappa_k
# =============================================================================

def search_report(runs, output_dir):
    """For every per-grade kappa-search run: print + save all trials
    (kappa, lambda, PSNR_val, energy) per grade with the chosen one
    marked, and a figure: chosen kappa_k vs grade (all search runs
    overlaid) plus, per run, the trial cloud kappa vs PSNR_val."""
    srs = [(n, m, load_search(n)) for n, m, _ in runs if m.get("ksearch")]
    srs = [(n, m, s) for n, m, s in srs if s]
    if not srs:
        print("  (no kappa-search runs -- skipping the search report)")
        return

    if MAX_GRADE is not None:
        srs = [(n, m, {**s, "grades": [g for g in s["grades"]
                                       if g["grade"] <= MAX_GRADE]})
               for n, m, s in srs]
    rows = []
    for n, m, s in srs:
        for g in s["grades"]:
            for t in g["trials"]:
                rows.append({
                    "run": n, "H": m.get("H", 1),
                    "tau": s.get("tau_dB", np.nan), "grade": g["grade"],
                    "kappa": t["kappa"], "lambda": t.get("lambda", np.nan),
                    "psnr_val_dB": t["psnr_val_dB"], "energy": t["energy"],
                    "chosen": bool(np.isclose(t["kappa"],
                                              g["chosen_kappa"])),
                })
    tbl = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print(" KAPPA-SEARCH TRIALS (per grade; * = chosen by rules (i)-(ii))")
    print("=" * 90)
    for n, _, _ in srs:
        sub = tbl[tbl["run"] == n]
        print(f"\n--- {n} ---")
        for g, gg in sub.groupby("grade"):
            for _, r in gg.sort_values("kappa").iterrows():
                mark = "  <== chosen" if r["chosen"] else ""
                print(f"  g{g}  kappa={r['kappa']:<10.4g} "
                      f"lambda={r['lambda']:.4e}  "
                      f"PSNR_val={r['psnr_val_dB']:7.3f} dB  "
                      f"energy={r['energy']:.4e}{mark}")
    tbl.to_csv(os.path.join(output_dir, "kappa_search_report.csv"),
               index=False)

    n_runs = len(srs)
    fig, axes = plt.subplots(1, 1 + n_runs,
                             figsize=(5.6 * (1 + n_runs), 4.6),
                             squeeze=False)
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#4a3aa7"]
    ax = axes[0][0]
    for i, (n, m, s) in enumerate(srs):
        gs = sorted(s["grades"], key=lambda g: g["grade"])
        ax.plot([g["grade"] for g in gs],
                [g["chosen_kappa"] for g in gs],
                marker="o", lw=2, color=colors[i % len(colors)],
                label=f"H{m.get('H', 1)}, " + _label(m))
    ax.set_yscale("log")
    ax.set_xlabel("grade $k$")
    ax.set_ylabel(r"chosen $\kappa_k$")
    ax.set_title("selected $\\kappa$ per grade", fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(fontsize=8, frameon=True)

    for i, (n, m, s) in enumerate(srs):
        ax = axes[0][1 + i]
        gs = sorted(s["grades"], key=lambda g: g["grade"])
        cmap = plt.get_cmap("viridis")
        for g in gs:
            cg = cmap((g["grade"] - 1) / max(1, len(gs) - 1))
            tr = sorted(g["trials"], key=lambda t: t["kappa"])
            ax.plot([t["kappa"] for t in tr],
                    [t["psnr_val_dB"] for t in tr],
                    marker=".", lw=1.0, color=cg,
                    label=f"g{g['grade']}")
            ax.plot([g["chosen_kappa"]],
                    [next(t["psnr_val_dB"] for t in tr if np.isclose(
                        t["kappa"], g["chosen_kappa"]))],
                    marker="*", ms=13, color=cg, mec="black", mew=0.5)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\kappa$ (trials)")
        ax.set_ylabel("PSNR val (dB)")
        ax.set_title(f"H{m.get('H', 1)}  " + _label(m),
                     fontsize=10)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.legend(fontsize=7, frameon=True, ncol=2)

    fig.tight_layout()
    path = os.path.join(output_dir, "figure_kappa_search.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n{path}")


# =============================================================================
# NESTED-RESOLUTION TEST REPORT: E^(m), m = N, ..., fN, per grade
# =============================================================================

def _nested_analysis(E):
    """Convergence analysis of one grade's nested sequence E^(m),
    m doubling per level (coarsest = training phase, finest = full
    grid).  Returns a dict:
      inflation  = E^(fN)/E^(N) -- how much the training-phase grid
                   UNDERestimates the fine-grid error (off-grid
                   overfitting reads directly as inflation >> 1);
      order_p    = Richardson order log2(d1/d2) of the increments
                   d_j = E^(2m) - E^(m)  (needs >= 3 levels, i.e.
                   f >= 4, and increments of one sign);
      E_ext      = Richardson-extrapolated m -> infinity limit
                   E^(fN) + d_last/(2^p - 1) -- the estimate of the
                   CONTINUOUS error ||F - I||^2_{L2(Omega)} that the
                   bound actually controls;
      psnr_ext   = the PSNR of E_ext (E = 12 * 10^(-PSNR/10));
      ext_over_fine = E_ext / E^(fN) -- how much error the finest
                   grid still misses (1.0 = nothing).
    """
    out = {"inflation": (E[-1] / E[0] if E[0] > 0 else np.nan),
           "order_p": np.nan, "E_ext": np.nan, "psnr_ext": np.nan,
           "ext_over_fine": np.nan}
    if len(E) >= 3:
        d1, d2 = E[-2] - E[-3], E[-1] - E[-2]
        if d1 * d2 > 0 and abs(d2) < abs(d1):
            p = np.log2(abs(d1) / abs(d2))
            E_ext = E[-1] + d2 / (2.0 ** p - 1.0)
            out.update(order_p=p, E_ext=E_ext,
                       psnr_ext=10.0 * np.log10(12.0 / E_ext)
                       if E_ext > 0 else np.nan,
                       ext_over_fine=E_ext / E[-1])
    return out


def nested_report(runs, output_dir):
    """Resolution-nested test errors from the per-grade history
    (keys E_T_res<m>): the discrete-testing-error convergence sequence
    E^(m) = ||Pi_m(F - I)||^2, m = N, 2N, ..., fN (Lemma 1 holds at
    every level).  Besides the raw table this ANALYZES the nesting per
    grade: the off-grid inflation E^(fN)/E^(N), the Richardson
    convergence order p of the increments, and the extrapolated
    continuous error E^(inf) with its PSNR (f >= 4 runs only -- two
    levels cannot estimate an order).  Figures: per-run E^(m) vs m
    log-log with the extrapolated limit dashed, plus a final-grade
    comparison panel across runs of the same cell."""
    found = []
    for n, m, _ in runs:
        h = load_history(n)
        if h is None:
            continue
        args, d = h
        res_cols = sorted(
            [c for c in d.columns if re.fullmatch(r"E_T_res\d+", c)],
            key=lambda c: int(c[len("E_T_res"):]))
        if len(res_cols) >= 2:
            found.append((n, m, d, res_cols))
    if not found:
        print("  (no nested-resolution columns in any history -- "
              "skipping the nested-test report)")
        return

    rows = []
    print("\n" + "=" * 90)
    print(" NESTED TEST ERROR E^(m) per grade (discrete -> continuous)")
    print("   inflation = E^(fN)/E^(N)   p = Richardson order   "
          "E_ext = extrapolated continuous error")
    print("=" * 90)
    for n, m, d, res_cols in found:
        ms = [int(c[len("E_T_res"):]) for c in res_cols]
        print(f"\n--- {n} ---")
        print("  grade  " + "  ".join(f"m={mm:<10d}" for mm in ms)
              + "  inflation   p      E_ext        PSNR_ext")
        for _, r in d.iterrows():
            E = [float(r[c]) for c in res_cols]
            an = _nested_analysis(E)
            psnr = [10.0 * np.log10(12.0 / e) if e > 0 else np.nan
                    for e in E]
            print(f"  {int(r['grade']):>4}   "
                  + "  ".join(f"{e:<12.5e}" for e in E)
                  + f"{an['inflation']:>8.3f}  "
                  + (f"{an['order_p']:>5.2f}  " if pd.notna(an["order_p"])
                     else "   --  ")
                  + (f"{an['E_ext']:.5e}  {an['psnr_ext']:6.2f} dB"
                     if pd.notna(an["E_ext"]) else "     --"))
            print("         "
                  + "  ".join(f"{p:<12.2f}" for p in psnr)
                  + "  (PSNR^(m), dB)")
            row = {"run": n, "grade": int(r["grade"])}
            row.update({f"E_res{mm}": e for mm, e in zip(ms, E)})
            row.update({f"psnr_res{mm}_dB": p
                        for mm, p in zip(ms, psnr)})
            row.update(an)
            rows.append(row)
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, "nested_test_report.csv"), index=False)

    nplots = len(found) + 1                 # +1: final-grade comparison
    fig, axes = plt.subplots(2, nplots,
                             figsize=(5.6 * nplots, 4.6 * 2),
                             squeeze=False)

    def _psnr(E):
        return [10.0 * np.log10(12.0 / e) if e > 0 else np.nan
                for e in E]

    for i, (n, m, d, res_cols) in enumerate(found):
        axE, axP = axes[0][i], axes[1][i]
        ms = [int(c[len("E_T_res"):]) for c in res_cols]
        cmap = plt.get_cmap("viridis")
        Kmax = int(d["grade"].max())
        for _, r in d.iterrows():
            g = int(r["grade"])
            E = [float(r[c]) for c in res_cols]
            cg = cmap((g - 1) / max(1, Kmax - 1))
            axE.plot(ms, E, marker="o", lw=1.8, color=cg,
                     label=f"$F_{{{g}}}$")
            axP.plot(ms, _psnr(E), marker="o", lw=1.8, color=cg,
                     label=f"$F_{{{g}}}$")
            an = _nested_analysis(E)
            if pd.notna(an["E_ext"]):       # extrapolated limit, dashed
                axE.axhline(an["E_ext"], color=cg, lw=1.0, ls="--",
                            alpha=0.6)
                axP.axhline(an["psnr_ext"], color=cg, lw=1.0, ls="--",
                            alpha=0.6)
        for ax, ylab, logy in [(axE, r"$E^{(m)}$", True),
                               (axP, r"PSNR$^{(m)}$ (dB)", False)]:
            ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            ax.set_xticks(ms)
            ax.set_xticklabels([str(mm) for mm in ms])
            ax.xaxis.set_minor_formatter(
                matplotlib.ticker.NullFormatter())
            ax.set_xlabel("test resolution $m$")
            ax.set_ylabel(ylab)
            ax.set_title(run_title(n, m), fontsize=10)
            ax.grid(True, which="both", linestyle="--", alpha=0.3)
            ax.legend(fontsize=7.5, frameon=True)

    # Comparison panels: FINAL grade's E^(m) / PSNR^(m) of every run
    # overlaid -- how the regularizers change the whole nested
    # sequence, not just the finest level.
    axE, axP = axes[0][len(found)], axes[1][len(found)]
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#4a3aa7", "#e34948", "#008300"]
    for i, (n, m, d, res_cols) in enumerate(found):
        ms = [int(c[len("E_T_res"):]) for c in res_cols]
        r = d.sort_values("grade").iloc[-1]
        E = [float(r[c]) for c in res_cols]
        c = colors[i % len(colors)]
        lab = run_title(n, m) + f" ($F_{{{int(r['grade'])}}}$)"
        axE.plot(ms, E, marker="o", lw=2.0, color=c, label=lab)
        axP.plot(ms, _psnr(E), marker="o", lw=2.0, color=c, label=lab)
        an = _nested_analysis(E)
        if pd.notna(an["E_ext"]):
            axE.axhline(an["E_ext"], color=c, lw=1.0, ls="--",
                        alpha=0.6)
            axP.axhline(an["psnr_ext"], color=c, lw=1.0, ls="--",
                        alpha=0.6)
    for ax, ylab, logy in [(axE, r"$E^{(m)}$", True),
                           (axP, r"PSNR$^{(m)}$ (dB)", False)]:
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("test resolution $m$")
        ax.set_ylabel(ylab)
        ax.set_title("final grade, all runs (dashed = extrapolated "
                     "limit)", fontsize=10)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.legend(fontsize=7, frameon=True)

    fig.tight_layout()
    path = os.path.join(output_dir, "figure_nested_test.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n{path}")


def run_pair_comparison(runs, output_dir):
    for title, noreg, best in select_pair(runs):
        tag = "_" + _slug(title)
        fig_pair_predictions(title, noreg, best, output_dir, tag)
        fig_pair_psnr(title, noreg, best, output_dir, tag)
        fig_pair_energy(title, noreg, best, output_dir, tag)
        fig_pair_objective(title, noreg, best, output_dir, tag)
        tables_pair_per_grade(title, noreg, best, output_dir, tag)



# =============================================================================
# TRIO COMPARISON: no-reg vs best H1 vs best H2 (fixed kappa OR search)
# =============================================================================

TRIO_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # no-reg / H1 / H2


def _trio_label(meta):
    """Legend label for the trio figures/tables: "no reg", "$H^1$ reg",
    "$H^2$ reg".  The per-grade kappa search and its tau are run
    settings, not part of the legend -- they are reported in the
    kappa-search table instead."""
    if pd.notna(meta["kappa"]) and meta["kappa"] == 0:
        return "no reg"
    base = f"$H^{int(meta.get('H', 1))}$ reg"
    if meta.get("ksearch"):
        return base
    return base + rf" ($\kappa$={meta['kappa']:g})"


def select_trio(runs):
    """Within one (image, N, factor) cell: the kappa = 0 baseline plus
    the BEST H1 run and the BEST H2 run by final test PSNR, where the
    candidates include both fixed-kappa (kappa > 0) and per-grade
    kappa-search runs.  Returns (title, [runs]) or None."""
    cells = {}
    for r in runs:
        key = (r[1]["image"] or r[0], r[1]["N"], r[1].get("factor", 2),
               r[1].get("noise", 0.0), r[1].get("rs", 0),
               r[1].get("epochs", np.nan))
        cells.setdefault(key, []).append(r)
    if PAIR_IMAGE is not None:
        cells = {k: v for k, v in cells.items()
                 if str(k[0]) == str(PAIR_IMAGE)}
        if not cells:
            print(f"  (trio: no runs for image '{PAIR_IMAGE}')")
            return []

    def final_test(r):
        return r[2].sort_values("global_epoch").iloc[-1]["psnr_test_dB"]

    out = []
    for key in sorted(cells, key=str):
        entries = cells[key]
        noreg = [r for r in entries
                 if pd.notna(r[1]["kappa"]) and r[1]["kappa"] == 0]
        if not noreg:
            continue
        trio = [noreg[0]]
        for H in (1, 2):
            cand = [r for r in entries if int(r[1].get("H", 1)) == H
                    and (r[1].get("ksearch")
                         or (pd.notna(r[1]["kappa"])
                             and r[1]["kappa"] > 0))]
            if cand:
                trio.append(max(cand, key=final_test))
        if len(trio) < 2:
            continue

        img, N, factor, noise, rs, ep = key
        title = f"{img}" + (f" (N={int(N)})" if pd.notna(N) else "")
        if int(factor) != 2:
            title += f" f={int(factor)}"
        if float(noise) > 0:
            title += f" ns={noise:g}"
        if int(rs):
            title += " rs"
        if pd.notna(ep):
            title += f" ep={int(ep)}"
        print(f"Trio comparison [{title}]: "
              + "  vs  ".join(_trio_label(r[1]).replace("$", "")
                              for r in trio))
        out.append((title, trio))
    if not out:
        print("  (trio: no cell with a kappa=0 baseline and a "
              "regularized run)")
    return out


def fig_trio_psnr(title, trio, output_dir, tag=""):
    """Train (dashed) / test (solid) PSNR during training, all runs."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for (name, meta, df), c in zip(trio, TRIO_COLORS):
        d = df.sort_values("global_epoch")
        lab = _trio_label(meta)
        ax.plot(d["global_epoch"], d["psnr_test_dB"], color=c, lw=2.2,
                label=f"{lab} | test")
        ax.plot(d["global_epoch"], d["psnr_train_dB"], color=c, lw=1.2,
                ls="--", alpha=0.6, label=f"{lab} | train")
    for e in grade_boundaries(trio[0][2]):
        ax.axvline(e, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel("global epoch", fontsize=17)
    ax.set_ylabel("PSNR (dB)", fontsize=17)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=11, frameon=True, loc="lower right")
    fig.tight_layout()
    path = os.path.join(output_dir, f"figure_trio_psnr{tag}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n{path}")


def fig_trio_energies(title, trio, output_dir, tag=""):
    """Per-grade energies from the history: O(F_k) (left panel) and
    O^(2)(F_k) (right panel) vs grade, one line per run."""
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    for (name, meta, df), c in zip(trio, TRIO_COLORS):
        h = load_history(name)
        if h is None:
            continue
        d = h[1].sort_values("grade")
        if "O_Fk" in d.columns and d["O_Fk"].notna().any():
            axes[0].plot(d["grade"], d["O_Fk"], marker="o", lw=2,
                         color=c, label=_trio_label(meta))
        if "O2_Fk" in d.columns and d["O2_Fk"].notna().any():
            axes[1].plot(d["grade"], d["O2_Fk"], marker="o", lw=2,
                         color=c, label=_trio_label(meta))
    for ax, ylab in [(axes[0], r"$\mathcal{O}_\Omega(F_k)$"),
                     (axes[1], r"$\mathcal{O}^{(2)}_\Omega(F_k)$")]:
        ax.set_yscale("log")
        ax.set_xlabel("grade $k$", fontsize=15)
        ax.set_ylabel(ylab, fontsize=15)
        ax.tick_params(labelsize=12)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.legend(fontsize=10, frameon=True)
    fig.tight_layout()
    path = os.path.join(output_dir, f"figure_trio_energies{tag}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n{path}")


def _trio_kappa_lambda(name, meta, d):
    """Per-grade (kappa_k, lambda_k) of one trio run, aligned to the
    grades of its history d.  kappa-search runs read chosen_kappa /
    chosen_lambda from <name>_kappa_search.json; fixed-kappa runs
    repeat the single kappa (lambda = penalty_weight * kappa); the
    kappa = 0 baseline is all zeros."""
    grades = [int(g) for g in d["grade"]]
    if pd.notna(meta.get("kappa", np.nan)) and meta["kappa"] == 0:
        kk = {g: 0.0 for g in grades}
        ll = {g: 0.0 for g in grades}
    elif meta.get("ksearch"):
        s = load_search(name)
        gs = (s or {}).get("grades", [])
        kk = {int(g["grade"]): g.get("chosen_kappa", np.nan) for g in gs}
        ll = {int(g["grade"]): g.get("chosen_lambda", np.nan) for g in gs}
    else:
        w, k = penalty_weight(meta), meta.get("kappa", np.nan)
        kk = {g: k for g in grades}
        lam = w * k if pd.notna(w) and pd.notna(k) else np.nan
        ll = {g: lam for g in grades}
    sk = pd.Series({g: kk.get(g, np.nan) for g in grades})
    sl = pd.Series({g: ll.get(g, np.nan) for g in grades})
    sk.index.name = sl.index.name = "grade"
    return sk, sl


def tables_trio(title, trio, output_dir, tag=""):
    """Per-grade tables for the three runs:
      table_trio_psnr   -- PSNR on train / val / unseen / test;
      table_trio_energy -- E_T, E_Omega, O(F_k), O^(2)(F_k);
      table_trio_kappa  -- the chosen kappa_k and lambda_k per grade
                           (search runs: from <name>_kappa_search.json).
    All values from the per-grade history."""
    hists = [load_history(r[0]) for r in trio]
    if any(h is None for h in hists):
        print("  (trio tables skipped: history not found)")
        return
    labels = [_trio_label(r[1]).replace("$", "") for r in trio]
    ds = []
    for h in hists:
        d = h[1].copy()
        d["E_Omega"] = d["E_T_test"]
        ds.append(d)

    def col(d, key):
        return (d.set_index("grade")[key] if key in d.columns
                else pd.Series(dtype=float))

    def block(metric_keys):
        out = {}
        for mk, key in metric_keys:
            for lab, d in zip(labels, ds):
                out[(mk, lab)] = col(d, key)
        return pd.DataFrame(out)

    t_psnr = block([("PSNR_train", "psnr_train_dB"),
                    ("PSNR_val", "psnr_val_dB"),
                    ("PSNR_unseen", "psnr_unseen_dB"),
                    ("PSNR_test", "psnr_test_dB")])
    t_energy = block([("E_T", "E_T_train"),
                      ("E_Omega", "E_T_test"),
                      ("O(F_k)", "O_Fk"),
                      ("O2(F_k)", "O2_Fk")])

    kls = [_trio_kappa_lambda(r[0], r[1], d) for r, d in zip(trio, ds)]
    t_kappa = pd.DataFrame(
        {**{("kappa_k", lab): kl[0] for lab, kl in zip(labels, kls)},
         **{("lambda_k", lab): kl[1] for lab, kl in zip(labels, kls)}})
    t_kappa.index.name = "grade"

    for t_, tbl, fmt in [("psnr", t_psnr, "{:.2f}"),
                         ("energy", t_energy, "{:.4e}"),
                         ("kappa", t_kappa, "{:.4g}")]:
        print(f"\n===== TRIO TABLE [{t_}] {title} =====")
        print(tbl.to_string(float_format=fmt.format))
        tbl.to_csv(os.path.join(output_dir, f"table_trio_{t_}{tag}.csv"))
        with open(os.path.join(output_dir, f"table_trio_{t_}{tag}.tex"),
                  "w") as fh:
            fh.write(tbl.to_latex(float_format=fmt.format))
    print(f"\nSaved trio tables (csv + tex) to {output_dir}")


def run_trio_comparison(runs, output_dir):
    for title, trio in select_trio(runs):
        tag = "_" + _slug(title)
        fig_trio_psnr(title, trio, output_dir, tag)
        fig_trio_energies(title, trio, output_dir, tag)
        tables_trio(title, trio, output_dir, tag)


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    runs = load_curves(CHECKPOINT_DIR)
    if not runs:
        print(f"No *_psnr_curve.csv found in {CHECKPOINT_DIR}.\n"
              "Train with TrainingApp (eval_psnr_every > 0) first.")
        return
    print(f"Found {len(runs)} run(s).")
    plot_psnr_curves(runs, OUTPUT_DIR)
    plot_energy_curves(runs, OUTPUT_DIR)
    plot_psnr_by_kappa(runs, OUTPUT_DIR)
    print_final_summary(runs, OUTPUT_DIR)
    search_report(runs, OUTPUT_DIR)
    nested_report(runs, OUTPUT_DIR)
    run_pair_comparison(runs, OUTPUT_DIR)
    run_trio_comparison(runs, OUTPUT_DIR)


if __name__ == "__main__":
    main()