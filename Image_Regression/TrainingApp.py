"""
Training engine for off-grid regularized SGDL / MGDL on image regression.

Objective for grade k:
    E_X(F_k) + lambda_r * S_M^(r)(F_k),   r = --reg_order in {1, 2},
where E_X is the grid error on the training pixels and S_M^(r) is the
Monte Carlo estimator of the Sobolev energy, evaluated at M fresh
uniform points of Omega = [-1,1]^2 at every step.  lambda_r is the
regularization weight lambda_{X,r}(kappa) of the paper; kappa is the only
free parameter.

Data: the image has fN x fN pixels (f = --train_factor).  Train = one
pixel per coarse cell (N x N), val = the diagonally opposite pixel
(used only to select kappa), test = all pixels.  For f = 4 the nested
test grids N, 2N, 4N are also evaluated.

Baselines: --reg_type wd (AdamW weight decay, no penalty)
and --penalty_points train (the same estimator with sample points drawn
from the training pixels).
"""

import argparse
import json
import logging
import math
import os
import sys
import time

import torch
import torch.nn as nn

from network import FixedDepthMGDL2D
from dataset import make_train_test, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s:%(lineno)03d %(message)s")
log = logging.getLogger(__name__)


# =====================================================================
# CLI
# =====================================================================

def get_args(sys_argv=None):
    p = argparse.ArgumentParser(description="Off-grid regularized MGDL, "
                                            "image regression")

    # Model
    p.add_argument("--model_name", type=str, default="mgdl_img")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--num_grades", type=int, default=4)
    p.add_argument("--layers_per_grade", type=int, default=3)

    # Optimization
    p.add_argument("--grade_epochs", type=str, default="2000",
                   help="Comma-separated epochs per grade, or a single value "
                        "used for every grade")
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--lr_decay", type=float, default=0.9)
    p.add_argument("--lr_decays_per_grade", type=int, default=7)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)

    # Data: image and grid
    p.add_argument("--image", type=str, required=True,
                   help="path of the target image")
    p.add_argument("--N", type=int, default=64,
                   help="Coarse grid N x N; the image is fN x fN")
    p.add_argument("--train_factor", type=int, default=2,
                   help="f: the image is fN x fN and TRAIN is every f-th "
                        "pixel per axis.  f=2 standard; f=4 also evaluates "
                        "the nested test grids N, 2N, 4N")

    # Penalty
    p.add_argument("--reg_order", type=int, default=1,
                   help="order r of the penalty S_M^(r): "
                        "1 = H^1, 2 = H^2")
    p.add_argument("--kappa", type=float, default=0.0,
                   help="penalty weight kappa; 0 = no penalty "
                        "(the unregularized model)")
    p.add_argument("--mc_samples", type=int, default=4096,
                   help="M: sample points of the penalty, redrawn at "
                        "every step")

    # Baselines: weight decay, penalty at the training points
    p.add_argument("--reg_type", type=str, default="sobolev",
                   choices=["sobolev", "wd"],
                   help="sobolev (default): the H^r penalty of --reg_order "
                        "with weight kappa.  wd: no penalty; decoupled "
                        "weight decay (AdamW) with coefficient kappa on the "
                        "trainable parameters of the current grade")
    p.add_argument("--penalty_points", type=str, default="offgrid",
                   choices=["offgrid", "train"],
                   help="offgrid (default): penalty samples x ~ U(Omega).  "
                        "train: penalty samples drawn uniformly from the "
                        "training pixels, same estimator and same |Omega| "
                        "factor (at --reg_order 1: Jacobian regularization)")
    p.add_argument("--eval_grid_energy", type=int, default=32768,
                   help="training pixels (fixed subset) for the energies "
                        "|Omega| mean_{x in D_train} ||grad^r F||^2, r = 1, 2, "
                        "reported next to O_Omega (0 disables)")

    # Evaluation (never touches training)
    p.add_argument("--eval_energy_mc", type=int, default=131072,
                   help="MC points for O^(1)_Omega(F_k)")
    p.add_argument("--eval_hess_mc", type=int, default=32768,
                   help="MC points for O^(2)_Omega(F_k) (0 disables)")
    p.add_argument("--eval_seed", type=int, default=12345)

    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")

    return p.parse_args(sys_argv if sys_argv is not None else [])


# =====================================================================
# Training application
# =====================================================================

class TrainingApp:
    def __init__(self, sys_argv=None):
        self.args = get_args(sys_argv or sys.argv[1:])
        a = self.args

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        set_seed(a.seed)

        self.d = 2
        self.channels = 3
        self.lo, self.hi = -1.0, 1.0
        self.vol = (self.hi - self.lo) ** self.d          # |Omega| = 4
        self.N = int(a.N)
        self.h = (self.hi - self.lo) / self.N             # h = 2/N
        self.factor = int(a.train_factor)                 # image = fN x fN

        # Regularization weights lambda_{X,r}(kappa) of the paper for square
        # cells of side h, whose
        # diameter is hbar = sqrt(d) h:
        #   lambda_1 = 4 hbar^2 kappa = 4 d h^2 kappa,   lambda_2 = d^2 h^4 kappa.
        d = self.d
        self.c1 = 4.0 * d
        self.c2 = float(d * d)

        self._x_train = None    # training pixels, used when --penalty_points train
        self.set_kappa(a.kappa)

        self.model = FixedDepthMGDL2D(
            width=a.width, max_grades=a.num_grades,
            n_layers=a.layers_per_grade, in_dim=self.d,
            out_dim=self.channels,
        ).to(self.device)

        self.history = []

        log.info(
            f"image={a.image} N={self.N} factor={self.factor} "
            f"(train {self.N}x{self.N}, test "
            f"{self.factor * self.N}x{self.factor * self.N}) "
            f"h={self.h:.6e} order={a.reg_order} kappa={a.kappa:.4g} -> "
            f"lambda_1={self.lam1:.6e} lambda_2={self.lam2:.6e}"
        )

    # -----------------------------------------------------------------
    def set_kappa(self, kappa):
        """Set kappa and recompute lambda_1, lambda_2 (or the weight-decay
        coefficient).  Called at construction and before every kappa trial."""
        a = self.args
        a.kappa = float(kappa)
        self.wd = 0.0
        if a.reg_type == "wd":          # weight-decay baseline: no penalty
            self.lam1, self.lam2 = 0.0, 0.0
            self.wd = a.kappa           # AdamW coefficient
            return
        if int(a.reg_order) == 2:
            self.lam1, self.lam2 = 0.0, self.c2 * self.h ** 4 * a.kappa
        else:
            self.lam1, self.lam2 = self.c1 * self.h ** 2 * a.kappa, 0.0

    # -----------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------
    def _eval_gen(self):
        g = torch.Generator(device=self.device)
        g.manual_seed(int(self.args.eval_seed))
        return g

    def _uniform(self, n, gen=None):
        return (torch.rand(n, self.d, device=self.device, generator=gen)
                * (self.hi - self.lo) + self.lo)

    def _mc_points(self, n, gen=None):
        """Sample points of the penalty: uniform on Omega, or drawn from
        the training pixels for --penalty_points train."""
        if self.args.penalty_points == "train":
            idx = torch.randint(0, self._x_train.size(0), (n,),
                                device=self._x_train.device, generator=gen)
            return self._x_train[idx], self.vol
        return self._uniform(n, gen), self.vol

    def _unit_dirs(self, n, gen=None):
        v = torch.randn(n, self.d, device=self.device, generator=gen)
        return v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)

    @staticmethod
    def _sq(res):
        """Channel-summed squared norm per sample: (n, C) -> (n,)."""
        return res.pow(2).sum(dim=1)

    # -----------------------------------------------------------------
    # S_M^(1): first-order penalty
    # -----------------------------------------------------------------
    def _S_hat(self, grade, M, create_graph=True):
        """S_M^(1) = |Omega| mean_m sum_c ||grad F_c(x_m)||^2, the Monte
        Carlo estimator of O^(1)_Omega."""
        x, vol = self._mc_points(M)
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            y = self.model(x, current_grade=grade)
            total = torch.zeros(x.size(0), device=x.device)
            for c in range(y.size(1)):
                g, = torch.autograd.grad(y[:, c].sum(), x,
                                         create_graph=create_graph,
                                         retain_graph=True)
                total = total + g.pow(2).sum(dim=1)
        return vol * total.mean()

    # -----------------------------------------------------------------
    # S_M^(2): second-order penalty
    # -----------------------------------------------------------------
    def _hvp_sq(self, x, v, grade, create_graph):
        """sum_c ||H_{F_c}(x) v||^2 per sample, by two autograd calls per
        channel.  create_graph=True keeps the parameters in the graph so
        the penalty can be back-propagated."""
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            y = self.model(x, current_grade=grade)
            total = torch.zeros(x.size(0), device=x.device)
            for c in range(y.size(1)):
                g, = torch.autograd.grad(y[:, c].sum(), x,
                                         create_graph=True, retain_graph=True)
                Hv, = torch.autograd.grad((g * v).sum(), x,
                                          create_graph=create_graph,
                                          retain_graph=True)
                total = total + Hv.pow(2).sum(dim=1)
        return total

    def _S2_hat(self, grade, M, create_graph=True):
        """S_M^(2) = d |Omega| mean_m sum_c ||H_{F_c}(x_m) v_m||^2 with
        x_m ~ U(Omega), v_m ~ U(S^{d-1}); an unbiased estimator of
        O^(2)_Omega."""
        x, vol = self._mc_points(M)
        v = self._unit_dirs(M)
        return self.d * vol * self._hvp_sq(x, v, grade, create_graph).mean()

    def _penalty(self, grade, M):
        """Returns (S1, S2); the inactive one is a zero scalar."""
        z = torch.zeros((), device=self.device)
        s1 = self._S_hat(grade, M) if self.lam1 > 0 else z
        s2 = self._S2_hat(grade, M) if self.lam2 > 0 else z
        return s1, s2

    # -----------------------------------------------------------------
    # One epoch of the objective on the training pixels
    # -----------------------------------------------------------------
    def train_epoch(self, x_all, y_all, grade):
        a = self.args
        self.model.train()

        n = x_all.size(0)
        bs = min(a.batch_size, n)
        nb = n // bs
        self._x_train = x_all

        perm = torch.randperm(n, device=self.device)
        # loss = E_X(F_k) + lambda_r S_M^(r)(F_k); both terms carry |Omega|.
        tot_fid = torch.zeros((), device=self.device)
        tot_s1 = torch.zeros((), device=self.device)
        tot_s2 = torch.zeros((), device=self.device)

        for i in range(nb):
            idx = perm[i * bs:(i + 1) * bs]
            self.optimizer.zero_grad(set_to_none=True)

            pred = self.model(x_all[idx], current_grade=grade)
            mse = self._sq(pred - y_all[idx]).mean()
            fid = self.vol * mse                          # = E_X

            s1, s2 = self._penalty(grade, a.mc_samples)
            loss = fid + self.lam1 * s1 + self.lam2 * s2
            loss.backward()
            self.optimizer.step()

            tot_fid += fid.detach()
            tot_s1 += s1.detach()
            tot_s2 += s2.detach()

        return ((tot_fid / nb).item(), (tot_s1 / nb).item(),
                (tot_s2 / nb).item())

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------
    @torch.no_grad()
    def _predict(self, x, grade):
        """Batched forward pass; x (n, 2) -> (n, 3)."""
        self.model.eval()
        bs = self.args.batch_size
        out = []
        for i in range(0, x.size(0), bs):
            out.append(self.model(x[i:i + bs], current_grade=grade))
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def _grid_error(self, x, y, grade):
        """Grid error E_X(F_k) = (|Omega|/n) sum ||y - F_k(x)||^2 on the
        n pixels x of a uniform grid."""
        pred = self._predict(x, grade)
        return (self.vol * self._sq(pred - y).sum().double()
                / x.size(0)).item()

    def eval_test_error(self, x_te, y_te, grade):
        """Grid error on the full fN x fN image, with the per-pixel MSE
        and the PSNR (dB, [0,1] scale)."""
        E = self._grid_error(x_te, y_te, grade)
        mse = E / self.vol / self.channels     # per pixel and channel
        psnr = -10.0 * math.log10(max(mse, 1e-20))
        return E, mse, psnr

    # -----------------------------------------------------------------
    # Pointwise derivative energies (evaluation only)
    # -----------------------------------------------------------------
    def _grad_sq(self, x, grade):
        """||grad F(x)||_F^2 by autograd, one backward per channel."""
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            y = self.model(x, current_grade=grade)
            total = torch.zeros(x.size(0), device=x.device)
            for c in range(y.size(1)):
                g, = torch.autograd.grad(y[:, c].sum(), x,
                                         retain_graph=(c < y.size(1) - 1))
                total = total + g.pow(2).sum(dim=1)
        return total.detach()

    def _hess_sq(self, x, grade):
        """sum_c ||H_{F_c}(x)||_F^2 by autograd double backward.
        Evaluation only."""
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            y = self.model(x, current_grade=grade)
            total = torch.zeros(x.size(0), device=x.device)
            C = y.size(1)
            for c in range(C):
                g, = torch.autograd.grad(y[:, c].sum(), x,
                                         create_graph=True, retain_graph=True)
                for i in range(self.d):
                    hi, = torch.autograd.grad(g[:, i].sum(), x,
                                              retain_graph=True)
                    total = total + hi.pow(2).sum(dim=1)
        return total.detach()

    # -----------------------------------------------------------------
    # Energies on Omega and on the training pixels
    # -----------------------------------------------------------------
    def _mc_energy(self, grade, n_points, fn, chunk=None):
        """|Omega| * mean over U(Omega) of the pointwise energy fn."""
        was_training = self.model.training
        self.model.eval()
        gen = self._eval_gen()
        bs = chunk or self.args.batch_size
        acc = torch.zeros((), device=self.device, dtype=torch.float64)
        done = 0
        while done < n_points:
            m = min(bs, n_points - done)
            acc += fn(self._uniform(m, gen), grade).sum().double()
            done += m
        if was_training:
            self.model.train()
        return (self.vol * acc / done).item()

    def eval_energy(self, grade, n_points=None):
        """O^(1)_Omega(F_k) = int ||grad F||_F^2 by MC."""
        n_points = n_points or self.args.eval_energy_mc
        return self._mc_energy(grade, n_points, self._grad_sq)

    def eval_hess_energy(self, grade, n_points=None):
        """O^(2)_Omega(F_k) = int ||H_F||_F^2 by MC with the exact Hessian.
        Chunked more finely than eval_energy: the double-backward graph
        holds d*C times the activations."""
        n_points = n_points or self.args.eval_hess_mc
        if n_points <= 0:
            return float("nan")
        chunk = max(256, self.args.batch_size // (2 * self.d * self.channels))
        return self._mc_energy(grade, n_points, self._hess_sq, chunk=chunk)

    def eval_grid_energies(self, grade, x_tr):
        """Energies |Omega| mean_{x in S} ||grad F||^2 and
        |Omega| mean_{x in S} ||H_F||_F^2 over a FIXED subset S of the
        training pixels (--eval_grid_energy pixels, the same subset for
        every run).  Compared with O_Omega they show how much of the
        energy lies between the training pixels."""
        n_max = int(self.args.eval_grid_energy)
        if n_max <= 0:
            return float("nan"), float("nan")
        was_training = self.model.training
        self.model.eval()
        x = x_tr
        if x.size(0) > n_max:
            idx = torch.randperm(x.size(0), generator=self._eval_gen(),
                                 device=self.device)[:n_max]
            x = x[idx]
        out = []
        for fn, chunk in (
                (self._grad_sq, self.args.batch_size),
                (self._hess_sq, max(256, self.args.batch_size
                                    // (2 * self.d * self.channels)))):
            acc = torch.zeros((), device=self.device, dtype=torch.float64)
            for i in range(0, x.size(0), chunk):
                acc += fn(x[i:i + chunk], grade).sum().double()
            out.append((self.vol * acc / x.size(0)).item())
        if was_training:
            self.model.train()
        return out[0], out[1]

    # -----------------------------------------------------------------
    # Data and training schedule
    # -----------------------------------------------------------------
    def load_data(self):
        """Build the train / val / unseen / test tensors once."""
        a = self.args
        data = make_train_test(a.image, self.N, self.device,
                               factor=self.factor)
        log.info(
            f"factor: {self.factor} (train = 1/{self.factor ** 2} of "
            f"the pixels)   "
            f"train: {tuple(data['x_train'].shape)} -> "
            f"{tuple(data['y_train'].shape)}   "
            f"val: {tuple(data['x_val'].shape)} (selection only)   "
            f"unseen: {tuple(data['x_unseen'].shape)}   "
            f"test: {tuple(data['x_test'].shape)} -> "
            f"{tuple(data['y_test'].shape)}   "
            f"nested resolutions: "
            f"{[t['m'] for t in data['test_nested']]}"
        )
        return data

    def grade_epoch_list(self):
        a = self.args
        ep = [int(v) for v in a.grade_epochs.split(",")]
        if len(ep) == 1:
            ep = ep * a.num_grades
        assert len(ep) == a.num_grades
        return ep

    def train_grade(self, grade, n_epoch, data):
        """Train grade `grade` for n_epoch epochs from the current model
        state (grades 1..grade-1 frozen) with the current kappa.  Used by
        main() and, trial by trial from the same snapshot, by the kappa
        search."""
        a = self.args
        x_tr, y_tr = data["x_train"], data["y_train"]

        self.model.set_trainable_grade(grade)

        # eps_k = RMS training residual of F_{k-1} (grade 1: RMS of y_train),
        # fixed for the whole grade and for every kappa trial of the grade.
        with torch.no_grad():
            if grade == 1:
                res = y_tr
            else:
                res = y_tr - self._predict(x_tr, grade - 1)
            eps_k = float(res.pow(2).mean().sqrt().clamp_min(1e-8))
        self.model.set_grade_scale(grade, eps_k)
        log.info(f"[grade {grade}] eps_{grade} = {eps_k:.4e}")

        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.wd > 0:                # weight-decay baseline
            self.optimizer = torch.optim.AdamW(
                params, lr=a.learning_rate, weight_decay=self.wd)
        else:
            self.optimizer = torch.optim.Adam(params, lr=a.learning_rate)

        step = (max(1, n_epoch // a.lr_decays_per_grade)
                if a.lr_decays_per_grade > 0 else n_epoch + 1)
        sched = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step, gamma=a.lr_decay
        )

        log.info(f"--- grade {grade}/{a.num_grades}, {n_epoch} epochs, "
                 f"kappa={a.kappa:g} ({a.reg_type}, "
                 f"penalty points: {a.penalty_points}) ---")

        for epoch in range(1, n_epoch + 1):
            t0 = time.time()
            fid, s1, s2 = self.train_epoch(x_tr, y_tr, grade)
            sched.step()
            if epoch % 50 == 0 or epoch == n_epoch:
                log.info(
                    f"G{grade} ep {epoch}/{n_epoch} | E_X {fid:.6e} | "
                    f"S1 {s1:.6e} S2 {s2:.6e} | lr "
                    f"{self.optimizer.param_groups[0]['lr']:.3e} | "
                    f"{time.time() - t0:.1f}s"
                )

    def finalize_grade(self, grade, data):
        """End-of-grade evaluation: errors, PSNR, energies, history row,
        checkpoint.  Returns the history record."""
        a = self.args
        x_tr, y_tr = data["x_train"], data["y_train"]
        x_va, y_va = data["x_val"], data["y_val"]
        x_un, y_un = data["x_unseen"], data["y_unseen"]
        x_te, y_te = data["x_test"], data["y_test"]

        E_tr = self._grid_error(x_tr, y_tr, grade)
        E_va = self._grid_error(x_va, y_va, grade)
        E_un = self._grid_error(x_un, y_un, grade)
        E_te, mse_te, psnr_te = self.eval_test_error(x_te, y_te, grade)
        mse_va = E_va / self.vol / self.channels
        psnr_va = -10.0 * math.log10(max(mse_va, 1e-20))
        mse_un = E_un / self.vol / self.channels
        psnr_un = -10.0 * math.log10(max(mse_un, 1e-20))
        # Nested test grids m = N, 2N, ..., fN: the grid error
        # on each sublattice; the last level is E_te above.
        nested = {}
        for t in data.get("test_nested", []):
            E_m = self._grid_error(t["x"], t["y"], grade)
            mse_m = E_m / self.vol / self.channels
            nested[int(t["m"])] = (E_m,
                                   -10.0 * math.log10(max(mse_m, 1e-20)))

        O_Fk = self.eval_energy(grade)
        O2_Fk = self.eval_hess_energy(grade)
        O_grid, O2_grid = self.eval_grid_energies(grade, x_tr)
        mse_tr = E_tr / self.vol / self.channels
        psnr_tr = -10.0 * math.log10(max(mse_tr, 1e-20))

        rec = {
            "grade": grade,
            "E_T_train": E_tr,
            "E_T_val": E_va,
            "E_T_unseen": E_un,
            "E_T_test": E_te,
            "mse_train": mse_tr,
            "mse_val": mse_va,
            "mse_test": mse_te,
            "psnr_train_dB": psnr_tr,
            "psnr_val_dB": psnr_va,
            "psnr_unseen_dB": psnr_un,
            "psnr_test_dB": psnr_te,
            "O_Fk": O_Fk,
            "O2_Fk": O2_Fk,
            "O_grid_Fk": O_grid,
            "O2_grid_Fk": O2_grid,
            "reg_type": a.reg_type,
            "penalty_points": a.penalty_points,
            "weight_decay": self.wd,
            "kappa": a.kappa,
            "lambda_1": self.lam1, "lambda_2": self.lam2,
            "eps_k": float(self.model.grade_scales[grade - 1].item()),
        }
        for m, (E_m, psnr_m) in sorted(nested.items()):
            rec[f"E_T_res{m}"] = E_m
            rec[f"psnr_res{m}_dB"] = psnr_m
        self.history.append(rec)
        if nested:
            log.info("[grade %d] nested test error E^(m): %s", grade,
                     "  ".join(f"m={m}: {E:.6e} ({p:.2f} dB)"
                               for m, (E, p) in sorted(nested.items())))
        log.info(
            f"[grade {grade}] E_train={E_tr:.6e} E_val={E_va:.6e} "
            f"E_test={E_te:.6e} | PSNR train={psnr_tr:.2f} val={psnr_va:.2f} "
            f"unseen={psnr_un:.2f} test={psnr_te:.2f} dB | "
            f"O1={O_Fk:.4e} O2={O2_Fk:.4e} | "
            f"on D_train: O1={O_grid:.4e} O2={O2_grid:.4e}"
        )

        torch.save(
            {"state_dict": self.model.state_dict(),
             "grade": grade, "history": self.history,
             "args": vars(a)},
            os.path.join(a.checkpoint_dir, f"{a.model_name}_g{grade}.pt"),
        )
        return rec

    def save_history(self):
        a = self.args
        with open(os.path.join(a.checkpoint_dir,
                               f"{a.model_name}_history.json"), "w") as fh:
            json.dump({"args": vars(a), "history": self.history}, fh, indent=2)

    # -----------------------------------------------------------------
    def main(self):
        a = self.args
        data = self.load_data()
        ep = self.grade_epoch_list()
        os.makedirs(a.checkpoint_dir, exist_ok=True)

        for grade in range(1, a.num_grades + 1):
            self.train_grade(grade, ep[grade - 1], data)
            self.finalize_grade(grade, data)
        self.save_history()


if __name__ == "__main__":
    TrainingApp().main()
