"""Measure post-training pricing latency and throughput.

The benchmark times the complete ``american_put_price`` rule, including
evaluation of both networks, exercise/continuation assembly, projection, and
recovery of the price and exercise boundary. Checkpoint loading and input
generation are excluded because the experiment measures repeated evaluation
after the model has been loaded.
"""

from __future__ import annotations

import platform
import statistics
import time

import numpy as np
import torch

from config import K_ref, device, random_seed
from diagnose_clamp import load_trained_models
from param_sampling import sample_heston_params
from pricer import american_put_price


BATCH_SIZES = (1, 10, 100, 1_000, 10_000, 100_000)
WARMUP_RUNS = 50


def repeat_count(batch_size: int) -> int:
    if batch_size <= 100:
        return 1_000
    if batch_size <= 1_000:
        return 500
    if batch_size <= 10_000:
        return 200
    return 100


def make_inputs(n: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(random_seed + n)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(random_seed + n)

    s = 2.0 * torch.rand((n, 1), device=device)
    v = torch.rand((n, 1), device=device)
    kappa, theta, sigma_v, rho, maturity = sample_heston_params(
        n, width_fraction=1.0
    )
    tau = maturity * torch.rand((n, 1), device=device)

    return {
        "S": K_ref * s,
        "v": v,
        "t": maturity - tau,
        "K": K_ref,
        "T": maturity,
        "kappa": kappa,
        "theta": theta,
        "sigma_v": sigma_v,
        "rho": rho,
    }


def synchronize() -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def time_batch(price_net, boundary_net, inputs: dict[str, torch.Tensor]) -> tuple[float, float]:
    for _ in range(WARMUP_RUNS):
        american_put_price(
            price_net=price_net,
            free_boundary_net=boundary_net,
            **inputs,
        )
    synchronize()

    timings_ms = []
    for _ in range(repeat_count(inputs["S"].shape[0])):
        synchronize()
        start = time.perf_counter_ns()
        american_put_price(
            price_net=price_net,
            free_boundary_net=boundary_net,
            **inputs,
        )
        synchronize()
        timings_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)

    median_ms = statistics.median(timings_ms)
    p95_ms = float(np.percentile(timings_ms, 95))
    return median_ms, p95_ms


def main() -> None:
    torch.set_num_threads(8)
    price_net, boundary_net = load_trained_models()

    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    print(f"Warm-up runs per batch: {WARMUP_RUNS}")
    print("Checkpoint loading and input generation: excluded")
    print()
    print(f"{'Batch':>10} {'Repeats':>10} {'Median (ms)':>14} {'P95 (ms)':>12} {'Prices/s':>14}")

    for batch_size in BATCH_SIZES:
        inputs = make_inputs(batch_size)
        median_ms, p95_ms = time_batch(price_net, boundary_net, inputs)
        throughput = batch_size / (median_ms / 1_000.0)
        print(
            f"{batch_size:>10,d} {repeat_count(batch_size):>10,d} "
            f"{median_ms:>14.6f} {p95_ms:>12.6f} {throughput:>14,.1f}"
        )


if __name__ == "__main__":
    main()
