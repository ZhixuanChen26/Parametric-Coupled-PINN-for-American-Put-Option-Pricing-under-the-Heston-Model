import os
import torch
from config import (
    device, K_ref, T_s_ref as T_ref, S_BOUNDS, V_BOUNDS, TAU_BOUNDS, POWER_S, POWER_TAU,
    PARAM_CURRICULUM_STAGES,
)
from models import create_models
from data_generation import generate_all_training_data
from train import train_initial_model, train_refined_model
from pricer import american_put_price
from visualize import (
    evaluate_and_compare, plot_free_boundaries, visualize_sampling_distributions,
    plot_training_history, plot_3d_price_surface, plot_2d_price_slices,
    plot_combined_greeks_with_table, plot_pde_residuals, plot_3d_free_boundary_surface
)

def main():
    print(f"Using device: {device}")

    # 1. Initialize Models (7-dim priceNN: s, v, tau, kappa_hat, theta_hat, sigma_v_hat, rho_hat;
    #    6-dim freeBoundaryNN: v, tau, kappa_hat, theta_hat, sigma_v_hat, rho_hat -- tau = T-t is the
    #    remaining time to maturity; T drops out of the PDE under this change of variables
    #    (Heston is time-homogeneous), see loss.py)
    priceNN, freeBoundaryNN = create_models(device)
    print("Models initialized successfully.")

    # 2. Configure Training Parameters
    BATCH_SIZE = 8192
    N_PDE = 150000
    N_TERM = 60000
    N_SPAT = 15000
    N_FREE = 60000
    N_FREE_TERM = 15000
    N_INTRINSIC = 40000

    # 3. Generating Initial Data (start of the parameter-space curriculum,
    #    i.e. a narrow window around the reference Heston parameters)
    all_data = generate_all_training_data(
        free_boundary_net=freeBoundaryNN,
        N_pde=N_PDE, N_term=N_TERM, N_spat=N_SPAT,
        N_free=N_FREE, N_free_term=N_FREE_TERM, N_intrinsic=N_INTRINSIC,
        s_bounds=S_BOUNDS, v_bounds=V_BOUNDS, tau_bounds=TAU_BOUNDS,
        power_s=POWER_S, power_tau=POWER_TAU,
        width_fraction=PARAM_CURRICULUM_STAGES[0]
    )

    data_pde = all_data['pde']
    data_term = all_data['term']
    data_spat = all_data['spat']
    data_free = all_data['free']
    data_free_term = all_data['free_term']
    data_intrinsic = all_data['intrinsic']

    # 4. Train Phase 1 & 2 (curriculum widens automatically across epochs,
    #    see train.get_curriculum_width)
    history = train_initial_model(
        priceNN, freeBoundaryNN, data_pde, data_term, data_spat, data_free, data_intrinsic,
        epochs=10000, batch_size=BATCH_SIZE,
        s_bounds=S_BOUNDS, v_bounds=V_BOUNDS, tau_bounds=TAU_BOUNDS,
        power_s=POWER_S, power_tau=POWER_TAU,
        N_pde=N_PDE, N_term=N_TERM, N_spat=N_SPAT, N_free=N_FREE, N_intrinsic=N_INTRINSIC,
        resample_every=1000
    )

    # 5. Train Phase 3 (Refined) -- stays at the full parameter range
    history2 = train_refined_model(
        priceNN, freeBoundaryNN, data_pde, data_term, data_spat, data_free, data_free_term, data_intrinsic,
        epochs=7000, batch_size=BATCH_SIZE,
        s_bounds=S_BOUNDS, v_bounds=V_BOUNDS, tau_bounds=TAU_BOUNDS,
        power_s=POWER_S, power_tau=POWER_TAU,
        N_pde=N_PDE, N_term=N_TERM, N_spat=N_SPAT, N_free=N_FREE, N_free_term=N_FREE_TERM, N_intrinsic=N_INTRINSIC,
        resample_every=1000
    )

    # 6. Evaluation and Checkpoint Saving
    save_dir = "./saved_models"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(priceNN.state_dict(), os.path.join(save_dir, "solutionNN.pth"))
    torch.save(freeBoundaryNN.state_dict(), os.path.join(save_dir, "boundaryNN.pth"))
    print(f"Models saved to '{save_dir}/'")

    # Running Benchmark Evaluation -- uses the REFERENCE Heston parameter
    # set (config.K_ref, T_s_ref, kappa_ref, ...) so results are directly
    # comparable against Rohan et al. Table 2/3/4.
    visualize_sampling_distributions(data_pde, freeBoundaryNN)
    df_results = evaluate_and_compare(priceNN, freeBoundaryNN, device, K=K_ref, T_maturity=T_ref)
    plot_free_boundaries(freeBoundaryNN, K_ref, T_ref)
    plot_training_history(history, history2)
    plot_3d_price_surface(priceNN, freeBoundaryNN, K_ref, device)
    plot_2d_price_slices(priceNN, freeBoundaryNN, K_ref, device)
    plot_combined_greeks_with_table(priceNN, freeBoundaryNN, K_ref, T_ref, device)
    plot_pde_residuals(priceNN, freeBoundaryNN, K_ref, T_ref, device)
    plot_3d_free_boundary_surface(freeBoundaryNN, K_ref, T_ref, device)

if __name__ == '__main__':
    main()
