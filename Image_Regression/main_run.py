"""
Driver for the image-regression experiments of the paper.

One call runs one image with one penalty:
  (1) the unregularized model (kappa = 0), unless baseline=0, and
  (2) the regularized model, with kappa selected at every grade by
      KappaSearchApp.

Command-line tokens:
  img=santorini|tokyo|sst|cathedral4|sst4
                     required; the "4" entries are the 4N x 4N images
                     evaluated on the nested grids N, 2N, 4N (f = 4)
  arch=mgdl|sgdl     model, default mgdl
  H=1|2              order of the penalty, default 1
  reg=sobolev|wd     wd: weight-decay baseline
  pts=offgrid|train  train: penalty at the training points (baseline)
  baseline=0|1       also run the kappa = 0 model, default 1

Examples:
  python main_run.py img=santorini H=1              # MGDL: no-reg and H1
  python main_run.py img=santorini H=2 baseline=0   # MGDL: H2
  python main_run.py img=santorini arch=sgdl H=1    # SGDL: no-reg and H1
  python main_run.py img=cathedral4 H=1             # nested test grids
  python main_run.py img=santorini reg=wd           # weight decay
  python main_run.py img=santorini H=1 pts=train    # H1 at the training pixels

Outputs (in ./checkpoints): <run>_g<k>.pt and <run>_history.json for every
run, and <run>_kappa_search.json for the regularized run, where <run> is
<img>_<arch>_<method>, e.g. santorini_mgdl_noreg, santorini_mgdl_H1,
santorini_mgdl_H1_train, santorini_mgdl_WD.
"""

import sys

from TrainingApp import TrainingApp
from KappaSearchApp import KappaSearchApp


# ------------------------------------------------------------------
# Command line
# ------------------------------------------------------------------
opts = {"img": None, "arch": "mgdl", "H": "1", "reg": "sobolev",
        "pts": "offgrid", "baseline": "1"}
for tok in sys.argv[1:]:
    key, sep, val = tok.partition("=")
    if not sep or key not in opts:
        raise SystemExit(f"unknown argument '{tok}'; use "
                         + " ".join(f"{k}=" for k in opts))
    opts[key] = val


# ------------------------------------------------------------------
# Targets.  N = (image side) / f, so the fN x fN test
# grid is the original image, without resampling.  The last two
# entries are the settings of the MGDL runs on that image.
#   name: (path, N, f, MGDL grades, MGDL epochs per grade)
# ------------------------------------------------------------------
IMAGES = {
    "santorini":  ("images/santorini.png",      642, 2, 4, 2000),
    "tokyo":      ("images/tokyo.png",          678, 2, 4, 2000),
    "sst":        ("images/sst1284_turbo.tif",  642, 2, 5, 2000),
    "cathedral4": ("images/cathedral.jpg",     1344, 4, 4, 2000),  # nested grids
    "sst4":       ("images/sst1284_turbo.tif",  321, 4, 4, 5000),  # nested grids
}
img = opts["img"]
if img not in IMAGES:
    raise SystemExit(f"img= is required, one of {list(IMAGES)}")
image, N, factor, mgdl_grades, mgdl_epochs = IMAGES[img]


# ------------------------------------------------------------------
# Models, as in the paper: width 256, GELU, minibatch 4096, M = 4096,
# learning rate reduced by the factor 0.9 seven times (per grade for
# MGDL).
# ------------------------------------------------------------------
if opts["arch"] == "mgdl":
    layers, grades, epochs, lr = 3, mgdl_grades, mgdl_epochs, 5e-4
elif opts["arch"] == "sgdl":
    if img.startswith("sst"):
        layers, grades, epochs, lr = 18, 1, 12000, 1e-4
    else:
        layers, grades, epochs, lr = 12, 1, 8000, 1e-4
else:
    raise SystemExit("arch= must be mgdl or sgdl")


# ------------------------------------------------------------------
# Penalty and search interval of kappa, as in the paper, in the units of
# the regularization weights lambda_{X,r}(kappa) of the paper:
# (kappa_min, kappa_max, max_trials).
# ------------------------------------------------------------------
H = int(opts["H"])
reg, pts = opts["reg"], opts["pts"]
if H not in (1, 2):
    raise SystemExit("H= must be 1 or 2")
if reg not in ("sobolev", "wd"):
    raise SystemExit("reg= must be sobolev or wd")
if pts not in ("offgrid", "train"):
    raise SystemExit("pts= must be offgrid or train")
if reg == "wd" and pts == "train":
    raise SystemExit("reg=wd adds no penalty; pts=train does not apply")

if reg == "wd":                                  # weight decay
    kmin, kmax, trials = 1e-5, 1.0, 10
elif pts == "train":                             # penalty at D_train
    kmin, kmax, trials = {1: (2.0833e-5, 2.0833, 10),
                          2: (3.6111e-5, 3.6111, 10)}[H]
    if img.startswith("sst"):                    # one decade lower on SST
        kmin /= 10.0
else:                                            # off-grid penalty
    kmin, kmax, trials = {1: (2.0833e-5, 2.0833e-2, 8),
                          2: (3.6111e-5, 3.6111e-2, 8)}[H]

# For the two baselines the kappa = 0 run is the unregularized model,
# which is already run with reg=sobolev pts=offgrid.
run_baseline = (opts["baseline"] == "1"
                and reg == "sobolev" and pts == "offgrid")


# ------------------------------------------------------------------
# Arguments passed to TrainingApp / KappaSearchApp
# ------------------------------------------------------------------
SEED = 0


def common_args(name):
    return [
        f"--model_name={name}",
        f"--image={image}",
        f"--N={N}",
        f"--train_factor={factor}",
        "--width=256",
        f"--layers_per_grade={layers}",
        f"--num_grades={grades}",
        f"--grade_epochs={epochs}",
        f"--learning_rate={lr}",
        "--lr_decay=0.9",
        "--lr_decays_per_grade=7",
        "--batch_size=4096",
        "--mc_samples=4096",
        f"--reg_order={H}",
        f"--reg_type={reg}",
        f"--penalty_points={pts}",
        f"--seed={SEED}",
    ]


base = f"{img}_{opts['arch']}"
method = ("WD" if reg == "wd"
          else f"H{H}" + ("_train" if pts == "train" else ""))


# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------
if run_baseline:
    print(f"=== {base}_noreg: kappa = 0 ===", flush=True)
    TrainingApp(common_args(f"{base}_noreg") + ["--kappa=0"]).main()

print(f"=== {base}_{method}: kappa search over [{kmin:g}, {kmax:g}], "
      f"at most {trials} trials per grade ===", flush=True)
KappaSearchApp(common_args(f"{base}_{method}")
               + [f"--kappa_min={kmin}",
                  f"--kappa_max={kmax}",
                  f"--max_trials={trials}"]).main()
