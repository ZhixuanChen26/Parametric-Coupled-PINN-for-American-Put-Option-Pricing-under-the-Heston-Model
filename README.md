# Parametric Coupled PINNs for American Put Pricing under the Heston Model

Code, trained checkpoints and result files for

> Zhixuan Chen, *Parametric Coupled Physics-Informed Neural Networks for American Put Option Pricing under the Heston Model*. Research paper (thesis) for the degree of Master of Mathematics in Computational Mathematics, University of Waterloo, 2026.

A single pair of networks prices an American put under the Heston stochastic-volatility model **over a range of Heston parameters** (κ, θ, σ_v, ρ), instead of being retrained for each parameter set:

| Network | Inputs | Output | Parameters |
|---|---|---|---|
| Price network p_ω | s, v, τ, κ̂, θ̂, σ̂_v, ρ̂ (7) | normalised price | 83,073 |
| Free-boundary network s\*_φ | v, τ, κ̂, θ̂, σ̂_v, ρ̂ (6) | normalised exercise boundary | 45,953 |

Here s = S/K is the strike-normalised asset price, τ the remaining maturity, v the variance, and hats denote normalised Heston parameters. Both networks are GELU multilayer perceptrons (`models.py`).

The networks are trained jointly. The price network is trained with the PDE residual, terminal condition, upper-asset-price boundary condition, an intrinsic-value loss inside the predicted exercise region, and the value-matching and smooth-pasting conditions at the free boundary. The free-boundary network is trained with value matching, smooth pasting and, in the refinement phase only, the boundary condition at maturity. After training, the price is the payoff below the predicted boundary and the price-network output above it, projected onto [intrinsic value, K] (`pricer.py`).

## Repository layout

**Model and training**

| File | Purpose |
|---|---|
| `config.py` | Reference Heston parameters, parameter box, domain bounds, seed (59), device selection |
| `models.py` | The two networks (`create_models`) |
| `loss.py` | PDE, terminal, boundary, value-matching, smooth-pasting, boundary-at-maturity and intrinsic-value residuals |
| `data_generation.py`, `param_sampling.py` | Collocation sampling, resampling, Feller-admissible parameter sampling and normalisation |
| `train.py` | Phase 1–2 (parameter-range curriculum) and Phase 3 (refinement) training loops |
| `pricer.py` | Post-training pricing rule: price and exercise boundary in original units |
| `main.py` | End-to-end training run; writes `saved_models/` and diagnostic plots |
| `visualize.py` | Plotting and table helpers, including every figure in Chapter 4 |
| `benchmark_data.py` | Ikonen–Toivanen reference prices for the benchmark comparison |

**Evaluation scripts** (no retraining; the five `evaluate_*` scripts run on CPU with seed 59)

| File | Purpose |
|---|---|
| `evaluate_chapter4_boundary.py` | Boundary-condition residuals, exercise-region payoff residuals, intrinsic-value shortfalls |
| `evaluate_intrinsic_ablation.py` | The four-model comparison for the intrinsic-value loss ablation |
| `evaluate_near_zero_variance.py`, `evaluate_upper_variance.py`, `evaluate_asset_boundary.py` | Behaviour near v → 0, v = 1 and the upper asset-price boundary |
| `benchmark_inference.py` | Pricing latency and throughput |
| `diagnose_clamp.py`, `diagnose_smax.py`, `diagnose_vmax.py` | Development-time diagnostics on a trained model. `diagnose_clamp.load_trained_models` is the shared checkpoint loader |

**Ablation training and cluster scripts**

| File | Purpose |
|---|---|
| `ablation_no_intrinsic.py`, `ablation_intrinsic_w5.py`, `ablation_intrinsic_w20.py` | Retrain with the intrinsic-value loss weight set to 0, 5 and 20 (the main model uses 10) |
| `run_gpu.sh`, `run_gpu_ablation_*.sh` | SLURM job scripts for `main.py` and the three ablation runs |

**Models, results and figures**

| Path | Contents |
|---|---|
| `saved_models/` | Final model (intrinsic-value weight 10) used throughout the thesis |
| `saved_models_ablation_no_intrinsic/`, `saved_models_ablation_intrinsic_w5/`, `saved_models_ablation_intrinsic_w20/` | Ablation models (weights 0, 5, 20) |
| `chapter4_boundary_results.json` | Output of `evaluate_chapter4_boundary.py` |
| `output/ablation/` | Output of `evaluate_intrinsic_ablation.py` (JSON and CSV) |
| `output/diagnostics/` | Outputs of the near-zero-variance, upper-variance and upper-asset-boundary scripts |
| `Final Graph/` | The exact PNG files included in the thesis |

Each checkpoint directory holds `solutionNN.pth` (price network) and `boundaryNN.pth` (free-boundary network) as PyTorch state dicts.

## Setup

```bash
pip install -r requirements.txt
```

The code was run with Python 3.9, PyTorch 2.7.1, NumPy 2.0, pandas 2.2 and Matplotlib 3.9 on macOS (Apple silicon). The cluster scripts use Python 3.11. Matplotlib 3.6 or newer is needed for the 3-D exercise-boundary figure. Training uses CUDA or MPS when available (`config.device`); the five `evaluate_*` scripts always run on CPU. Run every command below from the repository root.

## Quick start: price an option with the trained model

```python
from diagnose_clamp import load_trained_models
from pricer import american_put_price

price_net, free_boundary_net = load_trained_models("saved_models")

# Benchmark configuration: K = 10, kappa = 5, theta = 0.16, sigma_v = 0.9, rho = 0.1,
# remaining maturity tau = T - t = 0.25. The rate r = 0.10 and q = 0 are fixed in config.py.
price, boundary = american_put_price(
    S=10.0, v=0.25, t=0.0,
    price_net=price_net, free_boundary_net=free_boundary_net,
    K=10.0, T=0.25, kappa=5.0, theta=0.16, sigma_v=0.9, rho=0.1,
)
print(float(price), float(boundary))   # 0.7929 (Ikonen-Toivanen: 0.7959) and S* = 6.996
```

`S`, `v` and `t` also accept NumPy arrays or tensors, and the Heston parameters can be tensors with one value per row, so one call prices many states and parameter sets at once.

## Reproducing the thesis results

Table and figure numbers follow the October 2026 version of the thesis.

| Thesis item | Command or function | Output |
|---|---|---|
| Table 4.3, proposed-model rows (benchmark prices) | `visualize.evaluate_and_compare` | printed table |
| Figure 4.1 (price curves) | `visualize.plot_price_curves_benchmark` | `Final Graph/figure421.png` |
| Figure 4.3 (parameter response) | `visualize.plot_parametric_response_delta` | `Final Graph/figure423.png` |
| Figure 4.4 (boundary curves) | `visualize.plot_exercise_boundary_curves` | `Final Graph/figure431.png` |
| Figure 4.5 (boundary surface) | `visualize.plot_exercise_boundary_3d_surface` | `Final Graph/figure432.png` |
| Tables 4.6–4.8 (boundary residuals, exercise-region payoff residuals, intrinsic shortfalls) | `python evaluate_chapter4_boundary.py` | `chapter4_boundary_results.json` |
| Table 4.9 (intrinsic-value loss ablation) | `python evaluate_intrinsic_ablation.py` | `output/ablation/` |
| Figure A.1 (boundaries for the four weights) | `visualize.plot_exercise_boundary_ablation` | `Final Graph/figure44.png` |
| Table 4.10 (near zero variance) | `python evaluate_near_zero_variance.py` | `output/diagnostics/near_zero_variance.json` |
| Table 4.11 (upper variance boundary) | `python evaluate_upper_variance.py` | `output/diagnostics/upper_variance.json` |
| Table 4.12 (upper asset boundary) | `python evaluate_asset_boundary.py` | `output/diagnostics/upper_asset_boundary.json` |
| Table 4.13 (inference timing) | `python benchmark_inference.py` | printed (depends on hardware) |
| Tables 3.1–3.5 (settings) | `config.py`, `main.py`, `train.py`, `data_generation.py` | |

The figure functions save PNG files to the working directory. To regenerate all of them:

```python
from collections import OrderedDict
import torch
from config import device
from models import create_models
from diagnose_clamp import load_trained_models
from visualize import (evaluate_and_compare, plot_price_curves_benchmark, plot_parametric_response_delta,
                       plot_exercise_boundary_curves, plot_exercise_boundary_3d_surface,
                       plot_exercise_boundary_ablation)

price_net, free_boundary_net = load_trained_models("saved_models")
evaluate_and_compare(price_net, free_boundary_net, device)
plot_price_curves_benchmark(price_net, free_boundary_net)
plot_parametric_response_delta(price_net, free_boundary_net)
plot_exercise_boundary_curves(free_boundary_net)
plot_exercise_boundary_3d_surface(free_boundary_net)

ablation_dirs = OrderedDict([(0, "saved_models_ablation_no_intrinsic"), (5, "saved_models_ablation_intrinsic_w5"),
                             (10, "saved_models"), (20, "saved_models_ablation_intrinsic_w20")])
boundary_nets = OrderedDict()
for weight, directory in ablation_dirs.items():
    _, net = create_models(device)
    net.load_state_dict(torch.load(f"{directory}/boundaryNN.pth", map_location=device))
    net.eval()
    boundary_nets[weight] = net
plot_exercise_boundary_ablation(boundary_nets)
```

The five `evaluate_*` scripts were re-run for this release and reproduce the stored JSON and CSV files exactly (macOS, Apple silicon, CPU, PyTorch 2.7.1); last-digit differences are possible on other platforms. The regenerated figures match the files in `Final Graph/` apart from sub-pixel rendering differences. Running a script overwrites its stored output in place. The three diagnostic scripts take `--output` to write elsewhere.

## Training from scratch

```bash
python main.py
```

This runs Phases 1–2 (10,000 epochs, with the parameter range widened from 5% to 40% to 100% of the full box), then Phase 3 (7,000 epochs over the full range, adding the boundary-at-maturity loss), with batch size 8192 and resampling every 1,000 epochs. See Tables 3.3–3.5 of the thesis for pool sizes and the optimiser settings. It writes `saved_models/solutionNN.pth` and `saved_models/boundaryNN.pth`, so it **overwrites the shipped final model**; copy `saved_models/` first if you want to keep it. A GPU is recommended. Seeds are fixed (59), but GPU runs are not guaranteed to be bit-reproducible.

The ablation scripts do the same with a different intrinsic-value weight and write to `saved_models_ablation_*`, the directories that `evaluate_intrinsic_ablation.py` reads:

```bash
python ablation_no_intrinsic.py      # weight 0
python ablation_intrinsic_w5.py      # weight 5
python ablation_intrinsic_w20.py     # weight 20
```

The SLURM scripts were written for an Alliance (Compute Canada) style cluster. Before submitting, replace `YOUR_ALLOCATION_ACCOUNT` and set the virtual-environment path and project directory to your own, then:

```bash
mkdir -p logs
sbatch run_gpu.sh                    # or run_gpu_ablation_no_intrinsic.sh, etc.
```

## Not included

- **Comparator networks.** Figure 4.2, Tables 4.2 and 4.5 and the reproduced rows of Table 4.4 (Section 4.1.1 of the thesis) come from separately trained comparator networks. Their code and checkpoints are not part of this repository. `visualize.plot_rohan_scaled_surface_comparison` only draws the figure from price grids that you supply.
- **Literature prices.** The prices of other methods in Table 4.3 are transcribed from the cited papers. `benchmark_data.py` holds only the Ikonen–Toivanen values (Table 1 of their paper, operator-splitting method, finest grid (320, 128, 64)), stored to four decimals.

## Reference data

The numerical reference prices come from S. Ikonen and J. Toivanen, "Operator splitting methods for pricing American options under stochastic volatility", *Numerische Mathematik* 113 (2009), 299–324.

## Citation

```bibtex
@mastersthesis{chen2026parametric,
  author = {Zhixuan Chen},
  title  = {Parametric Coupled Physics-Informed Neural Networks for American Put Option Pricing under the Heston Model},
  school = {University of Waterloo},
  year   = {2026},
  type   = {Research paper, Master of Mathematics in Computational Mathematics}
}
```

## License

Released under the MIT License; see [LICENSE](LICENSE).
