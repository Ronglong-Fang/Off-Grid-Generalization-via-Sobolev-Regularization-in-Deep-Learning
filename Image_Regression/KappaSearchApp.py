"""
Per-grade selection of kappa.

With F_{k-1} frozen, grade k is trained once for every trial value of
kappa.  Every trial starts from the same snapshot (F_{k-1} and the
initial weights of grade k) and the same random stream, so that the
trials differ in kappa only.  The trial values are produced by a
golden-section search in log kappa over [kappa_min, kappa_max], with at
most max_trials trials; the search maximizes the PSNR on the validation
pixels, and the trial with the highest validation PSNR is retained.
Only the validation pixels are read by the selection; the test metrics
are computed by finalize_grade for the retained trial.

Usage: the CLI of TrainingApp plus the search options, e.g.

  python KappaSearchApp.py \
      --model_name=IMG_santorini_H1 --image=images/santorini.png \
      --N=642 --width=256 --layers_per_grade=3 --num_grades=4 \
      --grade_epochs=2000 --reg_order=1 \
      --kappa_min=2.0833e-5 --kappa_max=2.0833e-2 --max_trials=8

--kappa is ignored; the search sets it per grade.  Outputs are the usual
TrainingApp files for the retained trials, plus
<model_name>_kappa_search.json with every trial of every grade
(kappa, lambda, psnr_val_dB) and the chosen kappa_k.
"""

import argparse
import json
import logging
import math
import os
import sys

from TrainingApp import TrainingApp
from dataset import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s:%(lineno)03d %(message)s")
log = logging.getLogger(__name__)

INV_PHI = (math.sqrt(5.0) - 1.0) / 2.0        # 0.618...


def get_search_args(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--kappa_min", type=float, required=True,
                   help="Lower end of the search interval")
    p.add_argument("--kappa_max", type=float, required=True,
                   help="Upper end of the search interval")
    p.add_argument("--max_trials", type=int, default=8,
                   help="Maximum number of trials (grade trainings) per grade")
    p.add_argument("--bracket_tol", type=float, default=1.5,
                   help="Stop early when the bracket kappa_hi/kappa_lo "
                        "falls below this factor")
    return p.parse_known_args(argv)


class KappaSearchApp:
    def __init__(self, sys_argv=None):
        argv = list(sys_argv if sys_argv is not None else sys.argv[1:])
        self.s, rest = get_search_args(argv)
        self.app = TrainingApp(rest)
        if self.s.kappa_min <= 0 or self.s.kappa_max <= self.s.kappa_min:
            raise SystemExit("need 0 < kappa_min < kappa_max")

    # ---- model state snapshot / restore -----------------------------
    def _snapshot(self):
        return {k: v.detach().clone()
                for k, v in self.app.model.state_dict().items()}

    def _restore(self, snap):
        self.app.model.load_state_dict(snap)

    # ---- one trial: train grade k at one kappa from the snapshot ----
    def _trial(self, grade, n_epoch, kappa, data, snap):
        app = self.app
        self._restore(snap)
        app.set_kappa(kappa)
        # Same random stream for every trial of this grade (batch order
        # and penalty samples), so the trials differ in kappa only.
        set_seed(int(app.args.seed) + 100003 * grade)
        app.train_grade(grade, n_epoch, data)

        E_va = app._grid_error(data["x_val"], data["y_val"], grade)
        mse_va = E_va / app.vol / app.channels
        psnr_va = -10.0 * math.log10(max(mse_va, 1e-20))

        if app.args.reg_type == "wd":
            lam = app.wd                     # the AdamW coefficient
        elif int(app.args.reg_order) == 2:
            lam = app.lam2
        else:
            lam = app.lam1
        t = {"kappa": float(kappa), "lambda": float(lam),
             "psnr_val_dB": float(psnr_va), "state": self._snapshot()}
        log.info(f"[search g{grade}] kappa={kappa:.4g} "
                 f"(lambda={lam:.4e}) -> PSNR_val={psnr_va:.3f} dB")
        return t

    # ---- golden-section search in log kappa, maximizing PSNR_val ----
    def _search_grade(self, grade, n_epoch, data, snap):
        s = self.s
        lo, hi = math.log(s.kappa_min), math.log(s.kappa_max)
        trials = []

        def trial_at(x):
            t = self._trial(grade, n_epoch, math.exp(x), data, snap)
            trials.append(t)
            return t

        x1 = hi - INV_PHI * (hi - lo)
        x2 = lo + INV_PHI * (hi - lo)
        t1, t2 = trial_at(x1), trial_at(x2)
        while (len(trials) < s.max_trials
               and (hi - lo) > math.log(s.bracket_tol)):
            if t1["psnr_val_dB"] >= t2["psnr_val_dB"]:
                hi, x2, t2 = x2, x1, t1
                x1 = hi - INV_PHI * (hi - lo)
                t1 = trial_at(x1)
            else:
                lo, x1, t1 = x1, x2, t2
                x2 = lo + INV_PHI * (hi - lo)
                t2 = trial_at(x2)

        chosen = max(trials, key=lambda t: t["psnr_val_dB"])
        log.info(f"[search g{grade}] {len(trials)} trials; chosen "
                 f"kappa_{grade}={chosen['kappa']:.4g} "
                 f"(PSNR_val={chosen['psnr_val_dB']:.3f} dB)")
        return chosen, trials

    # ---- main -------------------------------------------------------
    def main(self):
        app, s = self.app, self.s
        a = app.args
        data = app.load_data()
        ep = app.grade_epoch_list()
        os.makedirs(a.checkpoint_dir, exist_ok=True)

        search_log = {"kappa_min": s.kappa_min, "kappa_max": s.kappa_max,
                      "max_trials": s.max_trials,
                      "bracket_tol": s.bracket_tol,
                      "reg_type": a.reg_type,
                      "penalty_points": a.penalty_points,
                      "grades": []}
        for grade in range(1, a.num_grades + 1):
            snap = self._snapshot()      # F_{grade-1} + grade k's init
            chosen, trials = self._search_grade(grade, ep[grade - 1],
                                                data, snap)
            # Adopt the retained trial: weights and kappa.
            self._restore(chosen["state"])
            app.set_kappa(chosen["kappa"])
            app.finalize_grade(grade, data)

            search_log["grades"].append({
                "grade": grade,
                "chosen_kappa": chosen["kappa"],
                "chosen_lambda": chosen["lambda"],
                "trials": [{k: t[k] for k in
                            ("kappa", "lambda", "psnr_val_dB")}
                           for t in sorted(trials,
                                           key=lambda t: t["kappa"])],
            })
            log.info(f"[search g{grade}] trials (kappa, lambda, PSNR_val):")
            for t in sorted(trials, key=lambda t: t["kappa"]):
                log.info(f"    {t['kappa']:>10.4g}  {t['lambda']:.4e}  "
                         f"{t['psnr_val_dB']:7.3f} dB"
                         + ("   <== chosen" if t is chosen else ""))
            with open(os.path.join(
                    a.checkpoint_dir,
                    f"{a.model_name}_kappa_search.json"), "w") as fh:
                json.dump(search_log, fh, indent=2)

        app.save_history()
        ks = [g["chosen_kappa"] for g in search_log["grades"]]
        log.info("kappa per grade: " + ", ".join(f"{k:.4g}" for k in ks))


if __name__ == "__main__":
    KappaSearchApp().main()
