# Off-grid generalization via Sobolev regularization in deep learning

Code for the image-regression experiments of the paper

> R. Fang and Y. Xu, *Off-grid generalization via Sobolev regularization in deep learning*.

A network trained only on the values of a target at the points of a grid is not constrained between the grid points. The paper bounds the error on the whole domain by the error on the grid plus a Sobolev seminorm of the network. It then trains with this seminorm as a penalty, evaluated at random points of the whole domain rather than at the training points.

This repository trains single-grade (SGDL) and multi-grade (MGDL) networks on RGB images in three ways:

- without a penalty;
- with the off-grid $H^1$ or $H^2$ penalty;
- with two baselines: weight decay, and the same penalty evaluated at the training pixels.

## Requirements

- Python 3.10 or later
- `torch`, `numpy`, `scikit-image`, `pandas`, `matplotlib`; `jupyter` for the notebook

```bash
pip install torch numpy scikit-image pandas matplotlib jupyter
```

A GPU is strongly recommended. The second-order penalty uses Hessian-vector products at every step.

## Files

| file | content |
|---|---|
| `main_run.py` | runs one experiment: one image, one model, one penalty |
| `TrainingApp.py` | training and evaluation of SGDL / MGDL for a fixed $\kappa$ |
| `KappaSearchApp.py` | selection of $\kappa$ at every grade on the validation pixels |
| `network.py` | the multi-grade network (SGDL is the case of one grade) |
| `dataset.py` | training, validation and test pixels of an image |
| `evaluate.py` | result tables (CSV) and the energy figure from the training outputs |
| `analysis.ipynb` | runs `evaluate.py`, compares the energies on $\Omega$ and at the training pixels, and checks the run settings |

## Data

Put the target images in `images/`. Each image is used at its full resolution without resampling, so its side must be exactly $f N$, where $f$ and $N$ are set in `main_run.py`.

| file | size (pixels) | source |
|---|---|---|
| `santorini.png` | 1284 × 1284 | DIV2K (Agustsson and Timofte, 2017) |
| `tokyo.png` | 1356 × 1356 | DIV2K (Agustsson and Timofte, 2017) |
| `cathedral.jpg` | 5376 × 5376 | DIV8K (Gu et al., 2019) |
| `sst1284_turbo.tif` | 1284 × 1284 | GHRSST Level 4 MUR sea surface temperature (JPL MUR MEaSUREs Project, 2015), central North Pacific, rendered in RGB with the turbo colormap |

Images must be 8-bit RGB. The pixels are mapped to $\Omega=[-1,1]^2$; each pixel sits at the centre of its cell.

## Data split

With subsampling factor $f$, the image has $fN\times fN$ pixels and is divided into $N\times N$ cells of $f\times f$ pixels:

- **train**: one pixel per cell, $N^2$ pixels;
- **validation**: the diagonally opposite pixel of each cell, used only to select $\kappa$;
- **unseen**: all remaining pixels, used neither for training nor for selection;
- **test**: all pixels.

For $f=4$ the test error is also computed on the nested grids $N$, $2N$ and $4N$.

## Running the experiments

Each call of `main_run.py` runs the unregularized model and the model with the chosen penalty, unless `baseline=0`.

```bash
# MGDL, off-grid penalties
python main_run.py img=santorini H=1
python main_run.py img=santorini H=2 baseline=0

# SGDL, off-grid penalties
python main_run.py img=santorini arch=sgdl H=1
python main_run.py img=santorini arch=sgdl H=2 baseline=0

# MGDL on the 4N x 4N images, evaluated on the nested grids
python main_run.py img=cathedral4 H=1
python main_run.py img=cathedral4 H=2 baseline=0

# Baselines: weight decay, and the penalty at the training pixels
python main_run.py img=santorini reg=wd
python main_run.py img=santorini H=1 pts=train
python main_run.py img=santorini H=2 pts=train
```

Replace `santorini` by `tokyo` or `sst`, and `cathedral4` by `sst4`, for the other images.

| token | values | default |
|---|---|---|
| `img=` | `santorini`, `tokyo`, `sst`, `cathedral4`, `sst4` | required |
| `arch=` | `mgdl`, `sgdl` | `mgdl` |
| `H=` | `1`, `2` (order of the penalty) | `1` |
| `reg=` | `sobolev`, `wd` (weight decay) | `sobolev` |
| `pts=` | `offgrid`, `train` (penalty at the training pixels) | `offgrid` |
| `baseline=` | `1` also runs the model without penalty, `0` does not | `1` |

All settings are in `main_run.py`: number of grades, layers, epochs, learning rate, and search interval of $\kappa$. They are the ones reported in the paper.

## Outputs

Each run is named `<image>_<arch>_<method>`, for example `santorini_mgdl_H1`, `santorini_mgdl_noreg`, `santorini_mgdl_H1_train` or `santorini_mgdl_WD`. The following files are written to `checkpoints/`:

- `<run>_g<k>.pt`: the model after grade $k$, with its history and arguments;
- `<run>_history.json`: one record per grade, containing
  - PSNR on the training, validation, unseen and test pixels;
  - Sobolev energies $\mathcal O^{(1)}_\Omega$ and $\mathcal O^{(2)}_\Omega$, and the same energies at the training pixels;
  - $\kappa$ and the penalty weight;
  - for $f=4$, the PSNR on the nested grids;
- `<run>_kappa_search.json`: for the regularized runs, every trial value of $\kappa$ with its validation PSNR, and the chosen $\kappa$ of every grade.

## Results

```bash
python evaluate.py
```

This writes the result tables as CSV files and the figure of the MGDL energies per grade (`mgdl_energy.pdf`) to `results/`. The CSV files are:

| file | content |
|---|---|
| `nested_grids.csv` | MGDL, PSNR on the nested grids |
| `sgdl_psnr.csv` | SGDL, PSNR on the training and test pixels |
| `sgdl_energy.csv` | SGDL, Sobolev energies |
| `mgdl_psnr_per_grade.csv` | MGDL, PSNR on the training and test pixels after each grade |
| `comparison_psnr.csv` | MGDL with weight decay, with the penalty at the training pixels, and with the off-grid penalty: PSNR |
| `comparison_energy.csv` | the same models: energies on $\Omega$ and at the training pixels |
| `kappa_per_grade.csv` | the chosen $\kappa$ of every grade |

`analysis.ipynb` runs `evaluate.py` and adds, for one image, the ratio of the energy on $\Omega$ to the energy at the training pixels after each grade.

## Notes

- **Penalty weight.** $\kappa$ is in the units of the regularization weight $\lambda_{X,r}(\kappa)$ of the paper. For square cells of side $h=2/N$ in $d=2$, this gives $\lambda_{X,1}=4d\,h^2\kappa$ and $\lambda_{X,2}=d^2h^4\kappa$. $\kappa$ is selected at every grade by a golden-section search in $\log\kappa$ that maximizes the validation PSNR. Every trial starts from the same model and the same random stream.
- **Residual scaling.** In MGDL, grade $k$ is added as $F_k=F_{k-1}+\epsilon_k g_k$. Here $\epsilon_k$ is the RMS training residual of $F_{k-1}$, computed once at the start of grade $k$ and then fixed.
- **Penalty samples.** The penalty points are uniform on $\Omega$ and are drawn anew at every step.
- **Reproducibility.** Runs use seed 0. On a GPU, cuDNN autotuning (`cudnn.benchmark = True`) makes results vary slightly from run to run.

## Citation

```bibtex
@article{fang_offgrid,
  author = {Fang, Ronglong and Xu, Yuesheng},
  title  = {Off-grid generalization via {S}obolev regularization in deep learning},
  year   = {2026},
  note   = {Submitted}
}
```

## License

See `LICENSE`.
