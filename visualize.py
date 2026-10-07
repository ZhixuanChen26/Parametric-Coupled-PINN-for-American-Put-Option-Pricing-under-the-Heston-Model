import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pricer import american_put_price
from loss import get_PDE_residual
from config import device, r, kappa_ref, theta_ref, sigma_v_ref, rho_ref, K_ref, v0_ref, T_s_ref as T_ref
from config import FELLER_MARGIN
from param_sampling import normalize_params
from benchmark_data import S_VALS as BENCHMARK_S_VALS, IKONEN_TOIVANEN


def visualize_sampling_distributions(data_pde, free_boundary_net):
    s_pde, v_pde, tau_pde = data_pde[0], data_pde[1], data_pde[2]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax1 = axes[0]
    ax1.hist(s_pde.cpu().numpy(), bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Strike s=1 (S=K)')
    ax1.set_xlabel('Dimensionless Stock Price s = S/K')
    ax1.set_ylabel('Density')
    ax1.set_title('PDE Points: s Distribution (Power-Law)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.hist(tau_pde.cpu().numpy(), bins=50, density=True, alpha=0.7, color='orange', edgecolor='black')
    ax2.axvline(x=0.0, color='red', linestyle='--', linewidth=2, label='Maturity tau=0')
    ax2.set_xlabel('Remaining Time to Maturity tau = T-t')
    ax2.set_ylabel('Density')
    ax2.set_title('PDE Points: tau Distribution (Power-Law towards maturity)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.hist(v_pde.cpu().numpy(), bins=50, density=True, alpha=0.7, color='green', edgecolor='black')
    ax3.set_xlabel('Variance v')
    ax3.set_ylabel('Density')
    ax3.set_title('PDE Points: v Distribution (Beta(0.6, 0.9))')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sampling_distributions_1d.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- 2D Scatter Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))

    n_plot = min(5000, s_pde.shape[0])
    idx = torch.randperm(s_pde.shape[0])[:n_plot]

    scatter = ax.scatter(
        s_pde[idx].cpu().numpy(),
        tau_pde[idx].cpu().numpy(),
        c=v_pde[idx].cpu().numpy(),
        cmap='viridis',
        alpha=0.5,
        s=5
    )

    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Strike s=1')
    ax.axhline(y=0.0, color='orange', linestyle='--', linewidth=2, label='Maturity tau=0')

    plt.colorbar(scatter, label='Variance v')
    ax.set_xlabel('Dimensionless Stock Price s = S/K')
    ax.set_ylabel('Remaining Time to Maturity tau = T-t')
    ax.set_title('PDE Training Points Distribution (Power-Law Sampling)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sampling_distributions_2d_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_free_boundaries(free_boundary_net, K, T,
                          kappa=kappa_ref, theta=theta_ref,
                          sigma_v=sigma_v_ref, rho=rho_ref):
    """Plot free boundary in PHYSICAL coordinates for a fixed Heston
    parameter set (default: the reference set, for comparison against
    Rohan et al. Figure 6)."""
    print("Generating Free Boundary curves...")

    v_levels = [0.0021, 0.0093, 0.0484, 0.0972, 0.2392, 1.0]
    colors = ['blue', 'green', 'yellow', 'red', 'purple', 'orange']

    t_physical = np.linspace(0.0, T, 100).astype(np.float32)
    tau_plot = T - t_physical
    tau_tensor = torch.tensor(tau_plot, dtype=torch.float32, device=device).reshape(-1, 1)

    kappa_t = torch.full_like(tau_tensor, kappa)
    theta_t = torch.full_like(tau_tensor, theta)
    sigma_v_t = torch.full_like(tau_tensor, sigma_v)
    rho_t = torch.full_like(tau_tensor, rho)
    T_t = torch.full_like(tau_tensor, T)
    kappa_hat, theta_hat, sigma_v_hat, rho_hat, T_s_n = normalize_params(kappa_t, theta_t, sigma_v_t, rho_t, T_t)

    plt.figure(figsize=(7, 4))
    free_boundary_net.eval()

    for i, v_val in enumerate(v_levels):
        v_tensor = torch.full_like(tau_tensor, v_val)

        X_bound = torch.cat([v_tensor, tau_tensor, kappa_hat, theta_hat, sigma_v_hat, rho_hat], dim=1)

        with torch.no_grad():
            s_star = free_boundary_net(X_bound)  # dimensionless s_star = S_star/K

        S_star_plot = (s_star * K).detach().cpu().numpy().flatten()

        plt.plot(t_physical, S_star_plot, label=f'v = {v_val}', color=colors[i], linewidth=2)

    plt.axhline(y=K, color='k', linestyle='--', alpha=0.5, label='Strike K')
    plt.xlabel('Time ($t$)')
    plt.ylabel('Critical Exercise Price ($S_f$)')
    plt.title(f'American Put Free Boundary (K={K}, kappa={kappa}, theta={theta}, sigma_v={sigma_v}, rho={rho})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("free_boundary_curves.png", dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_and_compare(price_net, free_boundary_net, device, K=K_ref, T_maturity=T_ref,
                          kappa=kappa_ref, theta=theta_ref,
                          sigma_v=sigma_v_ref, rho=rho_ref):
    """
    Generates plots and tables to compare with the reference paper.
    Defaults to the reference Heston parameter set so results are
    directly comparable to Rohan et al. Table 2/3 (Ikonen & Toivanen
    benchmark). Pass custom kappa/theta/sigma_v/rho to evaluate the
    parametric model at a different point in parameter space.
    """
    print("Generating Table 9 Comparison...")

    S_vals_table = [8.0, 9.0, 10.0, 11.0, 12.0]
    v_vals_table = [0.0625, 0.25]

    results = []

    for v_val in v_vals_table:
        row_prices = []
        for S_val in S_vals_table:
            t_input = 0.0

            price, _ = american_put_price(
                S=S_val, v=v_val, t=t_input,
                price_net=price_net, free_boundary_net=free_boundary_net,
                K=K, T=T_maturity,
                kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho
            )
            row_prices.append(price.item())

        results.append({
            "Variance": v_val,
            "S=8": row_prices[0], "S=9": row_prices[1], "S=10": row_prices[2],
            "S=11": row_prices[3], "S=12": row_prices[4]
        })

    df_table = pd.DataFrame(results)
    print("\n=== Table 9: Computed Prices (This Work) ===")
    print(df_table.to_string(index=False))
    print("-" * 60)

    print("Generating Figure 4 Plots...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    S_plot = np.linspace(K - K/3, K + (2*K)/3, 100)
    times_to_plot = [0.0, T_maturity/2, T_maturity]

    for idx, v_val in enumerate([0.25, 0.0625]):
        ax = axes[idx]

        for i, tau in enumerate(times_to_plot):
            t_input = T_maturity - tau
            t_input = max(0.0, min(T_maturity, t_input))

            prices = []
            for S_val in S_plot:
                p, _ = american_put_price(
                    S=S_val, v=v_val, t=t_input,
                    price_net=price_net, free_boundary_net=free_boundary_net,
                    K=K, T=T_maturity,
                    kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho
                )
                prices.append(p.item())

            ax.plot(S_plot, prices, label=f"t={tau}", linewidth=1.5)

        ax.set_title(f"Option Prices for v = {v_val}")
        ax.set_xlabel("Asset price S")
        ax.set_ylabel("Option price")
        ax.set_ylim(-0.1, 2.6)
        ax.set_xlim(7.5, 18.5)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    plt.tight_layout()
    plt.savefig("option_price_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    return df_table


def plot_training_history(history, history2, log_every=20):
    epochs_1 = np.arange(0, len(history['total_loss']) * log_every, log_every)

    start_epoch_2 = epochs_1[-1] + log_every
    epochs_2 = np.arange(
        start_epoch_2,
        start_epoch_2 + len(history2['total_loss']) * log_every,
        log_every
    )

    colors = {
        'total_loss': 'black',
        'PDE_loss': 'tab:blue',
        'term_loss': 'tab:orange',
        'B_loss': 'tab:green',
        'VM_loss': 'tab:red',
        'SP_loss': 'tab:purple',
        'BT_loss': 'tab:brown',
        'intrinsic_loss': 'tab:cyan'
    }

    plt.figure(figsize=(14, 6))

    shared_losses = [
        'total_loss', 'PDE_loss', 'term_loss',
        'B_loss', 'VM_loss', 'SP_loss', 'intrinsic_loss'
    ]

    for loss_name in shared_losses:
        plt.plot(
            epochs_1, history[loss_name],
            color=colors[loss_name],
            linewidth=2 if loss_name == 'total_loss' else 1,
            label=loss_name.replace('_', ' ').title()
        )
        plt.plot(
            epochs_2, history2[loss_name],
            color=colors[loss_name],
            linewidth=2 if loss_name == 'total_loss' else 1
        )

    plt.plot(
        epochs_2, history2['BT_loss'],
        color=colors['BT_loss'],
        linestyle='--',
        linewidth=1.5,
        label='BT Loss'
    )

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses vs Epochs (Combined)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("training_loss_history.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_3d_price_surface(priceNN, freeBoundaryNN, K, device,
                           kappa=kappa_ref, theta=theta_ref,
                           sigma_v=sigma_v_ref, rho=rho_ref):
    N_plot = 100
    S_range = np.linspace(1e-3, 25, N_plot)
    v_range = np.linspace(1e-3, 1.0, N_plot)
    S_grid, v_grid = np.meshgrid(S_range, v_range)

    S_flat = torch.tensor(S_grid.flatten(), dtype=torch.float32, device=device)
    v_flat = torch.tensor(v_grid.flatten(), dtype=torch.float32, device=device)
    t_flat = torch.full_like(S_flat, 0.0)  # t=0: now, matching the "(t=0)" title below

    V_pred_flat, _ = american_put_price(S_flat, v_flat, t_flat, priceNN, freeBoundaryNN,
                                         K=K, kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho)
    V_pred = V_pred_flat.cpu().numpy().reshape(N_plot, N_plot)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(S_grid, v_grid, V_pred, cmap='viridis', edgecolor='none')

    ax.set_xlabel('Asset Price (S)')
    ax.set_ylabel('Variance (v)')
    ax.set_zlabel('Option Price (V)')
    ax.set_title('3D Option Price Surface (t=0)')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.savefig("3d_price_surface.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_2d_price_slices(priceNN, freeBoundaryNN, K, device,
                          kappa=kappa_ref, theta=theta_ref,
                          sigma_v=sigma_v_ref, rho=rho_ref):
    t_eval = [0.0, 0.125, 0.25]
    v_fixed_1 = 0.0625
    v_fixed_2 = 0.25

    S_range = np.linspace(1e-3, 25, 100)
    S_1d = torch.tensor(S_range, dtype=torch.float32, device=device)
    payoff = np.maximum(K - S_range, 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    v_1d_1 = torch.full_like(S_1d, v_fixed_1)
    for t_val in t_eval:
        t_1d = torch.full_like(S_1d, t_val)
        V_slice, _ = american_put_price(S_1d, v_1d_1, t_1d, priceNN, freeBoundaryNN,
                                         K=K, kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho)
        ax1.plot(S_range, V_slice.cpu().numpy().flatten(), label=f't = {t_val}')

    ax1.set_xlabel('Asset Price (S)')
    ax1.set_ylabel('Option Price (V)')
    ax1.set_title(f'2D Option Price (v = {v_fixed_1})')
    ax1.legend()
    ax1.grid(True)

    v_1d_2 = torch.full_like(S_1d, v_fixed_2)
    for t_val in t_eval:
        t_1d = torch.full_like(S_1d, t_val)
        V_slice, _ = american_put_price(S_1d, v_1d_2, t_1d, priceNN, freeBoundaryNN,
                                         K=K, kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho)
        ax2.plot(S_range, V_slice.cpu().numpy().flatten(), label=f't = {t_val}')

    ax2.set_xlabel('Asset Price (S)')
    ax2.set_ylabel('Option Price (V)')
    ax2.set_title(f'2D Option Price (v = {v_fixed_2})')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("2d_price_slices.png", dpi=300, bbox_inches="tight")
    plt.close()


def get_greeks_data(priceNN, freeBoundaryNN, K, T, device,
                     kappa=kappa_ref, theta=theta_ref,
                     sigma_v=sigma_v_ref, rho=rho_ref):
    N_plot = 100
    S_range = np.linspace(1e-3, 30, N_plot)
    v_range = np.linspace(1e-3, 1.0, N_plot)
    S_grid, v_grid = np.meshgrid(S_range, v_range)

    S_tensor = torch.tensor(S_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
    v_tensor = torch.tensor(v_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
    t_tensor = torch.full_like(S_tensor, 0.0)

    s_dim = S_tensor / K
    T_t = torch.full_like(S_tensor, T)
    tau_dim = T_t - t_tensor

    kappa_t   = torch.full_like(S_tensor, kappa)
    theta_t   = torch.full_like(S_tensor, theta)
    sigma_v_t = torch.full_like(S_tensor, sigma_v)
    rho_t     = torch.full_like(S_tensor, rho)
    kappa_hat, theta_hat, sigma_v_hat, rho_hat, T_s_n = normalize_params(kappa_t, theta_t, sigma_v_t, rho_t, T_t)

    s_star_dim = freeBoundaryNN(torch.cat([v_tensor, tau_dim, kappa_hat, theta_hat, sigma_v_hat, rho_hat], dim=1))
    p_cont = priceNN(torch.cat([s_dim, v_tensor, tau_dim, kappa_hat, theta_hat, sigma_v_hat, rho_hat], dim=1))

    price_norm = torch.where(s_dim < s_star_dim, 1.0 - s_dim, p_cont)
    price_norm = torch.clamp(price_norm, min=0.0)
    V_physical = price_norm * K

    grads_3d = torch.autograd.grad(
        outputs=V_physical,
        inputs=(S_tensor, v_tensor),
        grad_outputs=torch.ones_like(V_physical),
        create_graph=False
    )

    delta_3d = grads_3d[0]
    vega_v_3d = grads_3d[1]

    # Convert derivative wrt variance (v) to derivative wrt volatility (sigma)
    vega_3d = vega_v_3d * (2.0 * torch.sqrt(v_tensor))

    Delta_grid = delta_3d.detach().cpu().numpy().reshape(N_plot, N_plot)
    Vega_grid = vega_3d.detach().cpu().numpy().reshape(N_plot, N_plot)
    return S_grid, v_grid, Delta_grid, Vega_grid


def plot_greeks(priceNN, freeBoundaryNN, K, T, device,
                kappa=kappa_ref, theta=theta_ref,
                sigma_v=sigma_v_ref, rho=rho_ref):
    S_grid, v_grid, Delta_grid, Vega_grid = get_greeks_data(
        priceNN, freeBoundaryNN, K, T, device, kappa, theta, sigma_v, rho)

    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf_delta = ax1.plot_surface(S_grid, v_grid, Delta_grid, cmap='coolwarm', edgecolor='none')
    ax1.set_xlabel('Asset Price (S)')
    ax1.set_ylabel('Variance (v)')
    ax1.set_zlabel(r'Delta ($\Delta$)')
    ax1.set_title(r'Delta Surface')
    fig.colorbar(surf_delta, ax=ax1, shrink=0.5, aspect=10)

    ax3 = fig.add_subplot(1, 2, 2, projection='3d')
    surf_vega = ax3.plot_surface(S_grid, v_grid, Vega_grid, cmap='plasma', edgecolor='none')
    ax3.set_xlabel('Asset Price (S)')
    ax3.set_ylabel('Variance (v)')
    ax3.set_zlabel(r'Vega ($\nu$)')
    ax3.set_title(r'Vega Surface (Corrected for $\sigma$)')
    fig.colorbar(surf_vega, ax=ax3, shrink=0.5, aspect=10)

    plt.tight_layout()
    plt.savefig("greeks_surfaces.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_combined_greeks_with_table(priceNN, freeBoundaryNN, K, T, device,
                                     kappa=kappa_ref, theta=theta_ref,
                                     sigma_v=sigma_v_ref, rho=rho_ref):
    S_grid, v_grid, Delta_grid, Vega_grid = get_greeks_data(
        priceNN, freeBoundaryNN, K, T, device, kappa, theta, sigma_v, rho)

    S_vals_table = [8.0, 9.0, 10.0, 11.0, 12.0]
    v_val_table = 0.0625  # typical variance for these benchmarks
    t_val_table = 0.0

    S_t = torch.tensor(S_vals_table, dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
    v_t = torch.full_like(S_t, v_val_table).requires_grad_(True)
    t_t = torch.full_like(S_t, t_val_table)

    s_dim_t = S_t / K
    T_t = torch.full_like(S_t, T)
    tau_dim_t = T_t - t_t

    kappa_t   = torch.full_like(S_t, kappa)
    theta_t   = torch.full_like(S_t, theta)
    sigma_v_t = torch.full_like(S_t, sigma_v)
    rho_t     = torch.full_like(S_t, rho)
    kappa_hat, theta_hat, sigma_v_hat, rho_hat, T_s_n = normalize_params(kappa_t, theta_t, sigma_v_t, rho_t, T_t)

    s_star_dim_t = freeBoundaryNN(torch.cat([v_t, tau_dim_t, kappa_hat, theta_hat, sigma_v_hat, rho_hat], dim=1))
    p_cont_t = priceNN(torch.cat([s_dim_t, v_t, tau_dim_t, kappa_hat, theta_hat, sigma_v_hat, rho_hat], dim=1))

    price_norm_t = torch.where(s_dim_t < s_star_dim_t, 1.0 - s_dim_t, p_cont_t)
    V_phys_t = torch.clamp(price_norm_t, min=0.0) * K

    grads = torch.autograd.grad(outputs=V_phys_t, inputs=(S_t, v_t), grad_outputs=torch.ones_like(V_phys_t), create_graph=False)
    pinn_delta = grads[0].detach().cpu().numpy().flatten()
    pinn_vega_v_raw = grads[1].detach().cpu().numpy().flatten()
    pinn_vega = pinn_vega_v_raw * (2.0 * np.sqrt(v_val_table))

    df = pd.DataFrame({
        'S': S_vals_table,
        'This Work (Delta)': pinn_delta,
        'This Work (Vega)': pinn_vega,
        'MATLAB F.D. (Delta)': [-0.9474, -0.7421, -0.4476, -0.2169, -0.0892],
        'MATLAB F.D. (Vega)': [0.0000, 0.6907, 0.8950, 0.6909, 0.3912],
        'Rouah [34] (Delta)': [-0.999, -0.731, -0.431, -0.211, -0.086],
        'Rouah [34] (Vega)': [-0.005, 0.848, 0.983, 0.675, 0.317],
    }).round(4)

    fig = plt.figure(figsize=(16, 12))

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    surf_delta = ax1.plot_surface(S_grid, v_grid, Delta_grid, cmap='coolwarm', edgecolor='none')
    ax1.set_xlabel('Asset Price (S)')
    ax1.set_ylabel('Variance (v)')
    ax1.set_zlabel(r'Delta ($\Delta$)')
    ax1.set_title(r'Delta Surface ($\partial V / \partial S$) at t=0')
    fig.colorbar(surf_delta, ax=ax1, shrink=0.5, aspect=10)

    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    surf_vega = ax2.plot_surface(S_grid, v_grid, Vega_grid, cmap='plasma', edgecolor='none')
    ax2.set_xlabel('Asset Price (S)')
    ax2.set_ylabel('Variance (v)')
    ax2.set_zlabel(r'Vega ($\nu$)')
    ax2.set_title(r'Vega Surface (Corrected for $\sigma$) at t=0')
    fig.colorbar(surf_vega, ax=ax2, shrink=0.5, aspect=10)

    ax3 = fig.add_subplot(2, 1, 2)
    ax3.axis('tight')
    ax3.axis('off')

    table = ax3.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    for i in range(len(df.columns)):
        table[(0, i)].set_text_props(weight='bold')

    ax3.set_title(f"Benchmark Comparison: Delta and Vega at v={v_val_table}, t={t_val_table}", weight='bold', pad=10)

    plt.tight_layout()
    plt.savefig("greeks_combined_with_table.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_pde_residuals(priceNN, freeBoundaryNN, K, T, device,
                        kappa=kappa_ref, theta=theta_ref,
                        sigma_v=sigma_v_ref, rho=rho_ref):
    N_grid = 100
    S_range = np.linspace(1e-3, 20, N_grid)
    t_range = np.linspace(0.0, T, N_grid)
    S_grid, t_grid = np.meshgrid(S_range, t_range)

    S_flat = torch.tensor(S_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    t_flat = torch.tensor(t_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)

    def get_masked_residual(v_val):
        v_flat = torch.full_like(S_flat, v_val)

        s_dim = (S_flat / K).requires_grad_(True)
        v_dim = v_flat.requires_grad_(True)

        kappa_t   = torch.full_like(S_flat, kappa)
        theta_t   = torch.full_like(S_flat, theta)
        sigma_v_t = torch.full_like(S_flat, sigma_v)
        rho_t     = torch.full_like(S_flat, rho)
        T_t       = torch.full_like(S_flat, T)
        tau_dim = (T_t - t_flat).requires_grad_(True)
        kappa_hat, theta_hat, sigma_v_hat, rho_hat, T_s_n = normalize_params(kappa_t, theta_t, sigma_v_t, rho_t, T_t)
        params_phys = (kappa_t, theta_t, sigma_v_t, rho_t, T_t)
        params_norm = (kappa_hat, theta_hat, sigma_v_hat, rho_hat, T_s_n)

        target_zeros = torch.zeros_like(s_dim)

        priceNN.eval()
        freeBoundaryNN.eval()

        with torch.no_grad():
            s_star_dim = freeBoundaryNN(torch.cat([v_dim, tau_dim, kappa_hat, theta_hat, sigma_v_hat, rho_hat], dim=1))
            S_star_physical_np = (s_star_dim * K).cpu().numpy().reshape(N_grid, N_grid)

        pde_res_flat = get_PDE_residual(s_dim, v_dim, tau_dim, params_phys, params_norm,
                                         target_zeros, priceNN)
        PDE_residual_grid = pde_res_flat.detach().cpu().numpy().reshape(N_grid, N_grid)

        mask_continuation = S_grid > S_star_physical_np
        PDE_residual_masked = np.abs(np.where(mask_continuation, PDE_residual_grid, np.nan))

        return PDE_residual_masked, S_star_physical_np[:, 0]

    variances = [0.0625, 0.25, 0.5, 1.0]
    results = [get_masked_residual(v) for v in variances]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    axes_flat = axes.flatten()

    cmap = plt.get_cmap('coolwarm')
    cmap.set_bad(color='lightgrey')

    vmin_val = 0.0
    vmax_val = 0.010

    for i, v_val in enumerate(variances):
        ax = axes_flat[i]
        res_masked, S_star = results[i]

        heatmap = ax.pcolormesh(t_grid, S_grid, res_masked, cmap=cmap, shading='auto',
                                vmin=vmin_val, vmax=vmax_val)
        ax.plot(t_range, S_star, color='black', linewidth=2.5, label='Free Boundary $S_f(t)$')

        ax.set_xlabel('Time to Maturity (t)')
        ax.set_ylabel('Asset Price (S)')
        ax.set_title(f'Absolute PDE Residual (v = {v_val})')
        ax.set_ylim(0, 20)
        ax.set_xlim(0, T)
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.5)

    cbar_ticks = [0.000, 0.002, 0.004, 0.006, 0.008, 0.010]
    fig.colorbar(heatmap, ax=axes, location='right', aspect=50, pad=0.02,
                 label='|PDE Residual|', ticks=cbar_ticks)

    plt.savefig("pde_residual_heatmaps.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_3d_free_boundary_surface(freeBoundaryNN, K, T, device,
                                   kappa=kappa_ref, theta=theta_ref,
                                   sigma_v=sigma_v_ref, rho=rho_ref):
    N_plot = 100

    v_range = np.linspace(1e-3, 1.0, N_plot)
    t_range = np.linspace(0.0, T, N_plot)
    v_grid, t_grid = np.meshgrid(v_range, t_range)

    v_tensor = torch.tensor(v_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    t_tensor = torch.tensor(t_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    tau_tensor = T - t_tensor

    kappa_t   = torch.full_like(v_tensor, kappa)
    theta_t   = torch.full_like(v_tensor, theta)
    sigma_v_t = torch.full_like(v_tensor, sigma_v)
    rho_t     = torch.full_like(v_tensor, rho)
    T_t       = torch.full_like(v_tensor, T)
    kappa_hat, theta_hat, sigma_v_hat, rho_hat, T_s_n = normalize_params(kappa_t, theta_t, sigma_v_t, rho_t, T_t)

    freeBoundaryNN.eval()
    with torch.no_grad():
        X_bound = torch.cat([v_tensor, tau_tensor, kappa_hat, theta_hat, sigma_v_hat, rho_hat], dim=1)
        s_star_dim = freeBoundaryNN(X_bound)
        S_star_physical = (s_star_dim * K).cpu().numpy().reshape(N_plot, N_plot)

    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(111, projection='3d')
    surf = ax1.plot_surface(v_grid, t_grid, S_star_physical, cmap='viridis', edgecolor='none')

    ax1.set_xlabel('Variance ($v$)')
    ax1.set_ylabel('Time ($t$)')
    ax1.set_zlabel('Free Boundary ($S_f$)')
    ax1.set_title('Learned 3D Free Boundary Surface $S_f(v, t)$')
    plt.savefig("3d_free_boundary_surface.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_price_curves_benchmark(price_net, free_boundary_net, K=K_ref, T_maturity=T_ref,
                                 kappa=kappa_ref, theta=theta_ref,
                                 sigma_v=sigma_v_ref, rho=rho_ref):
    """
    Sec 4.2.2 Figure 1: American put price curves at the benchmark Heston
    parameters. One panel per current variance (v=0.0625, v=0.25); each
    panel shows three remaining maturities (tau=0.25, 0.125, 0.025), the
    terminal payoff H(S)=(K-S)^+, and the Ikonen and Toivanen reference
    prices at tau=0.25 (benchmark_data.py).
    """
    S_plot = np.linspace(7.5, 16.0, 200)
    tau_values = [0.25, 0.125, 0.025]
    v_panels = [0.0625, 0.25]
    N_plot = len(S_plot)

    payoff = np.maximum(K - S_plot, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)

    tau_linestyles = ['-', '-.', ':']

    for ax, v_val in zip(axes, v_panels):
        v_arr = np.full(N_plot, v_val)
        for tau, ls in zip(tau_values, tau_linestyles):
            t_arr = np.full(N_plot, T_maturity - tau)
            prices, _ = american_put_price(
                S=S_plot, v=v_arr, t=t_arr,
                price_net=price_net, free_boundary_net=free_boundary_net,
                K=K, T=T_maturity,
                kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho
            )
            ax.plot(S_plot, prices.cpu().numpy().flatten(), linestyle=ls,
                    label=fr'$\tau = {tau}$', linewidth=1.8)

        ax.plot(S_plot, payoff, color='black', linestyle='--', linewidth=1.3,
                label=r'$H(S) = (K-S)^+$')

        ref_prices = IKONEN_TOIVANEN[v_val]
        ax.scatter(BENCHMARK_S_VALS, ref_prices, facecolors='none', edgecolors='black',
                   marker='o', s=28, zorder=5, linewidths=1.1,
                   label='I&T reference')

        ax.set_xlabel('Asset price $S$')
        ax.set_title(fr'$v = {v_val}$')
        ax.set_xlim(7.5, 16.0)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('American put price $V$')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08),
               ncol=5, frameon=False, fontsize=12,
               handlelength=1.8, columnspacing=1.5, markerscale=1.1)

    plt.tight_layout()
    plt.savefig("price_curves_benchmark.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_parametric_price_response(price_net, free_boundary_net, K=K_ref, T_maturity=T_ref,
                                    v_fixed=v0_ref, tau_fixed=0.25):
    """
    Sec 4.2.3 Figure: parametric price response. One panel per Heston
    parameter (kappa, theta, sigma_v, rho); each panel varies that
    parameter across three Feller-admissible values (a low value, the
    benchmark reference, and a high value), holding the other three at
    their reference values, and plots V(S) at the benchmark (v, tau).
    This demonstrates that the single trained model responds smoothly and
    sensibly across the Heston parameter domain -- it is not a
    parameter-domain accuracy check, since no independent numerical
    reference exists off the single benchmark point (Sec 4.7.2).
    """
    S_plot = np.linspace(7.5, 16.0, 200)
    N_plot = len(S_plot)
    v_arr = np.full(N_plot, v_fixed)
    t_arr = np.full(N_plot, T_maturity - tau_fixed)
    payoff = np.maximum(K - S_plot, 0.0)

    # Each value list is [low, reference, high]; low/high were chosen to
    # satisfy the Feller margin (3.7) with the OTHER three parameters held
    # at their reference values -- kappa >= ~2.80 and theta >= ~0.0898 at
    # (theta,sigma_v)=(0.16,0.9) and (kappa,sigma_v)=(5,0.9) respectively,
    # so e.g. kappa=1 or theta=0.01 (the domain's own bounds) would be
    # off-domain here even though they're valid elsewhere in D.
    panels = [
        (r'$\kappa$', 'kappa',   [3.0, kappa_ref, 10.0]),
        (r'$\theta$', 'theta',   [0.10, theta_ref, 0.50]),
        (r'$\sigma_v$', 'sigma_v', [0.1, 0.5, sigma_v_ref]),
        (r'$\rho$',   'rho',     [-0.9, -0.4, rho_ref]),
    ]
    ref_lookup = {'kappa': kappa_ref, 'theta': theta_ref, 'sigma_v': sigma_v_ref, 'rho': rho_ref}

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    cmap = plt.cm.viridis

    for ax, (symbol, pname, values) in zip(axes_flat, panels):
        shades = cmap(np.linspace(0.15, 0.85, len(values)))
        for val, color in zip(values, shades):
            kwargs = dict(kappa=kappa_ref, theta=theta_ref, sigma_v=sigma_v_ref, rho=rho_ref)
            kwargs[pname] = val
            is_ref = (val == ref_lookup[pname])
            prices, _ = american_put_price(
                S=S_plot, v=v_arr, t=t_arr,
                price_net=price_net, free_boundary_net=free_boundary_net,
                K=K, T=T_maturity, **kwargs
            )
            label = fr'{symbol}$={val:g}$' + (' (ref)' if is_ref else '')
            ax.plot(S_plot, prices.cpu().numpy().flatten(), color=color,
                    linewidth=2.4 if is_ref else 1.6, label=label)

        ax.plot(S_plot, payoff, color='black', linestyle='--', linewidth=1.1,
                label=r'$H(S)$')

        ax.set_title(f'Varying {symbol}')
        ax.set_xlim(7.5, 16.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

    axes[0, 0].set_ylabel('American put price $V$')
    axes[1, 0].set_ylabel('American put price $V$')
    axes[1, 0].set_xlabel('Asset price $S$')
    axes[1, 1].set_xlabel('Asset price $S$')

    plt.tight_layout()
    plt.savefig("parametric_price_response.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_parametric_response_delta(price_net, free_boundary_net, K=K_ref, T_maturity=T_ref,
                                    v_fixed=v0_ref, tau_fixed=0.25, n_sweep=150):
    """
    Sec 4.2.3 Figure: parametric price response relative to the benchmark
    configuration. One panel per Heston parameter; each panel sweeps that
    parameter continuously across its Feller-admissible range (the other
    three held at their reference values) and plots
    Delta V(x) = V(mu with that parameter = x) - V(mu = reference)
    at S = 9, 10, 11, at the benchmark (v, tau). Each panel keeps its own
    y-axis scale so that small effects (e.g. sigma_v) are not hidden by
    the much larger effect of theta. This shows how the trained model
    responds to each parameter -- it is not a parameter-domain accuracy
    check, since no independent numerical reference exists off the single
    benchmark point (Sec 4.7.2).
    """
    t_fixed = T_maturity - tau_fixed
    S_curves = [8.0, 9.0, 10.0, 11.0, 12.0]

    ref_prices = {}
    for S_val in S_curves:
        p, _ = american_put_price(
            S=S_val, v=v_fixed, t=t_fixed,
            price_net=price_net, free_boundary_net=free_boundary_net,
            K=K, T=T_maturity, kappa=kappa_ref, theta=theta_ref,
            sigma_v=sigma_v_ref, rho=rho_ref
        )
        ref_prices[S_val] = p.item()

    # Feller-admissible endpoints for kappa and theta, with the OTHER
    # three parameters held at their reference values (see the discussion
    # of (3.7) with sigma_v, theta, kappa fixed in turn) -- sigma_v and
    # rho are unconstrained by Feller over their full domain here.
    kappa_min = (sigma_v_ref / FELLER_MARGIN) ** 2 / (2.0 * theta_ref)
    theta_min = (sigma_v_ref / FELLER_MARGIN) ** 2 / (2.0 * kappa_ref)

    panels = [
        (r'$\kappa$', 'kappa',     kappa_min, 10.0, kappa_ref),
        (r'$\theta$', 'theta',     theta_min, 0.5,  theta_ref),
        (r'$\sigma_v$', 'sigma_v', 0.1,       1.0,  sigma_v_ref),
        (r'$\rho$',   'rho',       -0.9,      0.1,  rho_ref),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    axes_flat = axes.flatten()
    colors = plt.cm.turbo(np.linspace(0.08, 0.92, len(S_curves)))

    for ax, (symbol, pname, lo, hi, ref_val) in zip(axes_flat, panels):
        sweep = np.linspace(lo, hi, n_sweep)
        sweep_t = torch.tensor(sweep, dtype=torch.float32, device=device).unsqueeze(1)
        v_arr = np.full(n_sweep, v_fixed)
        t_arr = np.full(n_sweep, t_fixed)

        for S_val, color in zip(S_curves, colors):
            S_arr = np.full(n_sweep, S_val)
            kwargs = dict(kappa=kappa_ref, theta=theta_ref, sigma_v=sigma_v_ref, rho=rho_ref)
            kwargs[pname] = sweep_t
            prices, _ = american_put_price(
                S=S_arr, v=v_arr, t=t_arr,
                price_net=price_net, free_boundary_net=free_boundary_net,
                K=K, T=T_maturity, **kwargs
            )
            delta = prices.cpu().numpy().flatten() - ref_prices[S_val]
            ax.plot(sweep, delta, color=color, linewidth=1.8, label=fr'$S={S_val:g}$')

        ax.axhline(0.0, color='gray', linewidth=0.9, label=r'$\Delta V = 0$')
        ax.axvline(ref_val, color='gray', linewidth=0.9, linestyle='--', label='Benchmark value')
        ax.set_xlabel(symbol)
        ax.set_ylabel(r'$\Delta V$')
        ax.set_title(f'Varying {symbol}')
        ax.grid(True, alpha=0.3)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.06),
               ncol=7, frameon=False, fontsize=11, handlelength=1.8, columnspacing=1.3)

    plt.tight_layout()
    plt.savefig("parametric_price_response_delta.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_rohan_scaled_surface_comparison(S_grid, v_grid, V_proposed, V_rohan):
    """
    Sec 4.2.2 Figure 2: proposed-model vs Rohan-scaled price surfaces at
    the benchmark configuration (tau=0.25), and their absolute difference.
    V_proposed and V_rohan must already be evaluated on the same
    (S_grid, v_grid) mesh -- this function only plots, it does not know
    how either model was evaluated.

    This is a comparison BETWEEN two trained models, not an error against
    an independent numerical reference (no such reference exists off the
    single benchmark point, Sec 4.7.2). Panel 3 is therefore labelled
    "cross-model absolute difference", not "pricing error".

    Returns (mean, 95th percentile, max) of the absolute difference.
    """
    diff = np.abs(V_proposed - V_rohan)
    mean_diff = diff.mean()
    p95_diff = np.percentile(diff, 95)
    max_diff = diff.max()

    vmin = min(V_proposed.min(), V_rohan.min())
    vmax = max(V_proposed.max(), V_rohan.max())

    # Row 1 holds the two price panels side by side, sharing one colorbar;
    # row 2 holds the difference panel alone, widened, with its own
    # colorbar. Positioned by hand (figure-fraction coordinates) rather
    # than gridspec + tight_layout/constrained_layout: with a shared
    # multi-axes colorbar on a custom GridSpec, both layout engines either
    # collapse the axes or place the colorbar on top of a panel. This
    # native aspect ratio is close to square, so shrinking the figure to
    # \textwidth in the thesis does not crush the text the way the
    # original 1x3 layout did.
    fig = plt.figure(figsize=(10, 9.5))
    ax0     = fig.add_axes([0.08, 0.54, 0.37, 0.38])
    ax1     = fig.add_axes([0.47, 0.54, 0.37, 0.38])
    cbar_ax = fig.add_axes([0.87, 0.54, 0.025, 0.38])
    ax2     = fig.add_axes([0.16, 0.06, 0.58, 0.40])
    cbar_ax2 = fig.add_axes([0.77, 0.06, 0.03, 0.40])

    ax0.pcolormesh(S_grid, v_grid, V_proposed, shading='auto', vmin=vmin, vmax=vmax, cmap='viridis')
    ax0.set_title('(a) Proposed model')
    im1 = ax1.pcolormesh(S_grid, v_grid, V_rohan, shading='auto', vmin=vmin, vmax=vmax, cmap='viridis')
    ax1.set_title('(b) Rohan scaled')
    ax1.set_yticklabels([])
    fig.colorbar(im1, cax=cbar_ax, label='American put price $V$')

    im2 = ax2.pcolormesh(S_grid, v_grid, diff, shading='auto', cmap='inferno')
    ax2.set_title('(c) Absolute difference')
    fig.colorbar(im2, cax=cbar_ax2, label=r'$|\hat{V}_{\mathrm{proposed}} - \hat{V}_{\mathrm{Rohan}}|$')

    for ax in (ax0, ax1, ax2):
        ax.set_xlabel('Asset price $S$')
    ax0.set_ylabel('Variance $v$')
    ax2.set_ylabel('Variance $v$')

    plt.savefig("rohan_scaled_surface_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    return mean_diff, p95_diff, max_diff


def _exercise_boundary_s_star_fn(K, kappa, theta, sigma_v, rho):
    """Returns a function (v_arr, tau_arr) -> predicted S*(v, tau) in asset price units,
    using the raw (unclipped) boundary-network output, at fixed Heston
    parameters. Shared by the two exercise-boundary figures below."""
    kappa_hat, theta_hat, sigma_v_hat, rho_hat, _ = normalize_params(
        torch.tensor([[kappa]], dtype=torch.float32, device=device),
        torch.tensor([[theta]], dtype=torch.float32, device=device),
        torch.tensor([[sigma_v]], dtype=torch.float32, device=device),
        torch.tensor([[rho]], dtype=torch.float32, device=device),
        torch.tensor([[T_ref]], dtype=torch.float32, device=device),
    )

    def s_star_batch(free_boundary_net, v_arr, tau_arr):
        n = len(v_arr)
        v_t = torch.tensor(v_arr, dtype=torch.float32, device=device).unsqueeze(1)
        tau_t = torch.tensor(tau_arr, dtype=torch.float32, device=device).unsqueeze(1)
        X_bound = torch.cat([v_t, tau_t, kappa_hat.repeat(n, 1), theta_hat.repeat(n, 1),
                              sigma_v_hat.repeat(n, 1), rho_hat.repeat(n, 1)], dim=1)
        free_boundary_net.eval()
        with torch.no_grad():
            s_phi = free_boundary_net(X_bound)
        return (s_phi * K).cpu().numpy().flatten()

    return s_star_batch


def plot_exercise_boundary_curves(free_boundary_net, K=K_ref,
                                   kappa=kappa_ref, theta=theta_ref,
                                   sigma_v=sigma_v_ref, rho=rho_ref):
    """
    Sec 4.3.1 Figure: exercise boundary S*(tau) at the benchmark Heston
    parameters, for the two benchmark variance levels v = 0.0625 and
    v = 0.25 (Table 4.1), over the benchmark maturity range tau in
    [0, 0.25]. Uses the raw (unclipped) boundary-network output. The
    black dashed line marks S* = K, the tau = 0 terminal condition.
    """
    s_star_batch = _exercise_boundary_s_star_fn(K, kappa, theta, sigma_v, rho)

    fig, ax = plt.subplots(figsize=(7, 5))
    tau_curve = np.linspace(0.0, 0.25, 200)
    for v_val in [0.0625, 0.25]:
        S_star_curve = s_star_batch(free_boundary_net, np.full(len(tau_curve), v_val), tau_curve)
        ax.plot(tau_curve, S_star_curve, linewidth=1.8, label=fr'$v={v_val:g}$')

    ax.axhline(K, color='black', linestyle='--', linewidth=1.2, label='$S^*=K$')
    ax.set_xlabel(r'Remaining maturity $\tau$')
    ax.set_ylabel(r'Predicted exercise boundary $\widehat{S}^*$')
    ax.set_xlim(0, 0.25)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)

    plt.tight_layout()
    plt.savefig("exercise_boundary_curves.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_exercise_boundary_3d_surface(free_boundary_net, K=K_ref,
                                       kappa=kappa_ref, theta=theta_ref,
                                       sigma_v=sigma_v_ref, rho=rho_ref):
    """
    Sec 4.3.1 Figure: exercise boundary S*(v, tau) 3D surface over the
    full trained ranges tau in [0, 1] and v in [0.01, 1], at the
    benchmark Heston parameters. Uses the raw (unclipped) boundary-network
    output. Shows the shape of the learned boundary across the model's
    whole trained maturity range (beyond the single fixed-maturity
    comparison in Sec 4.2), not an error against an independent numerical
    boundary.
    """
    s_star_batch = _exercise_boundary_s_star_fn(K, kappa, theta, sigma_v, rho)

    N_plot = 100
    # Power-law grid in tau, concentrating points near tau=0: S*(tau) drops
    # steeply within roughly tau < 0.03 (see the benchmark curve figure),
    # and a uniform grid only places 2-3 of the N_plot points there, which
    # renders as faceted/jagged rather than smooth in a 3D surface.
    tau_ax = np.linspace(0.0, 1.0, N_plot) ** 3
    v_ax = np.linspace(0.01, 1.0, N_plot)
    tau_grid, v_grid = np.meshgrid(tau_ax, v_ax)
    S_star_grid = s_star_batch(free_boundary_net, v_grid.flatten(), tau_grid.flatten()).reshape(tau_grid.shape)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(tau_grid, v_grid, S_star_grid, cmap='viridis',
                     edgecolor='none', linewidth=0, antialiased=True)

    # Black cross-section at the benchmark maturity tau=0.25, connecting
    # this surface back to the fixed-maturity results in Sec 4.2. Lifted by
    # a tiny epsilon so it renders on top of the surface instead of
    # z-fighting with it.
    v_slice = np.linspace(0.01, 1.0, 200)
    tau_slice = np.full_like(v_slice, 0.25)
    S_star_slice = s_star_batch(free_boundary_net, v_slice, tau_slice)
    ax.plot(tau_slice, v_slice, S_star_slice + 0.02, color='black', linestyle='--',
            linewidth=1.2, zorder=10)

    ax.set_xlabel(r'Remaining maturity $\tau$')
    ax.set_ylabel('Variance $v$')
    ax.set_zlabel(r'Predicted exercise boundary $\widehat{S}^*$', labelpad=12)

    # mplot3d reserves a large fixed margin around the box (for rotation
    # headroom) regardless of figure size; set_position with an overscanned
    # rect just grows that margin further once bbox_inches="tight" expands
    # to include it. `zoom` instead scales the box itself within its
    # allotted area, which is what actually shrinks the surrounding margin.
    ax.set_box_aspect(None, zoom=1.35)

    # matplotlib's 3D axes report an incorrect tight bbox for the z-label,
    # so bbox_inches="tight" alone still clips it -- pad_inches adds a
    # fixed safety margin on top of whatever (too-small) box it computes.
    plt.savefig("exercise_boundary_3d_surface.png", dpi=300, bbox_inches="tight", pad_inches=0.6)
    plt.close()


def plot_exercise_boundary_ablation(free_boundary_nets, K=K_ref,
                                     kappa=kappa_ref, theta=theta_ref,
                                     sigma_v=sigma_v_ref, rho=rho_ref):
    """
    Sec 4.4 ablation diagnostic: exercise boundary S*(tau) at the benchmark
    Heston parameters, for the intrinsic-value-loss weight ablation, at the
    two benchmark variance levels v = 0.0625 and v = 0.25, over the
    benchmark maturity range tau in [0, 0.25]. Uses the raw (unclipped)
    boundary-network output.

    free_boundary_nets: mapping w_intrinsic (float) -> trained
    free_boundary_net, one per ablation checkpoint. No independent
    numerical boundary reference is available (see benchmark_data.py), so
    this compares the checkpoints against each other, not against a
    ground-truth boundary.
    """
    s_star_batch = _exercise_boundary_s_star_fn(K, kappa, theta, sigma_v, rho)

    tau_curve = np.linspace(0.0, 0.25, 200)
    colors = plt.cm.turbo(np.linspace(0.08, 0.92, len(free_boundary_nets)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, v_val in zip(axes, [0.0625, 0.25]):
        for color, (w, net) in zip(colors, free_boundary_nets.items()):
            S_star_curve = s_star_batch(net, np.full(len(tau_curve), v_val), tau_curve)
            ax.plot(tau_curve, S_star_curve, color=color, linewidth=1.8,
                    label=fr'$\lambda_{{\mathrm{{intrinsic}}}}={w:g}$')
        ax.axhline(K, color='black', linestyle='--', linewidth=1.2,
                   label=r'Theoretical terminal boundary $S^*(v,0)=K$')
        ax.set_xlabel(r'Remaining maturity $\tau$')
        ax.set_xlim(0, 0.25)
        ax.grid(True, alpha=0.3)
        ax.set_title(fr'$v={v_val:g}$')
    axes[0].set_ylabel(r'Predicted exercise boundary $\widehat{S}^*$')
    axes[1].tick_params(labelleft=True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.06),
               ncol=len(free_boundary_nets) + 1, frameon=False, fontsize=10)

    plt.tight_layout()
    plt.savefig("exercise_boundary_ablation.png", dpi=300, bbox_inches="tight")
    plt.close()
