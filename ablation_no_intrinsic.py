import os
import torch
from config import (
    device, K_ref, T_s_ref as T_ref, S_BOUNDS, V_BOUNDS, TAU_BOUNDS, POWER_S, POWER_TAU,
    PARAM_CURRICULUM_STAGES,
)
from models import create_models
from data_generation import generate_all_training_data
from train import train_initial_model, train_refined_model
from visualize import evaluate_and_compare, plot_training_history

# Ablation: L_intrinsic removed (w_intrinsic = 0) to test the effect of
# dropping the intrinsic-value supervision term (Table 3.2/3.3) on
# accuracy near the exercise boundary. Everything else matches main.py.

def main():
    print(f"Using device: {device}")

    priceNN, freeBoundaryNN = create_models(device)
    print("Models initialized successfully.")

    BATCH_SIZE = 8192
    N_PDE = 150000
    N_TERM = 60000
    N_SPAT = 15000
    N_FREE = 60000
    N_FREE_TERM = 15000
    N_INTRINSIC = 40000

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

    history = train_initial_model(
        priceNN, freeBoundaryNN, data_pde, data_term, data_spat, data_free, data_intrinsic,
        epochs=10000, batch_size=BATCH_SIZE,
        s_bounds=S_BOUNDS, v_bounds=V_BOUNDS, tau_bounds=TAU_BOUNDS,
        power_s=POWER_S, power_tau=POWER_TAU,
        N_pde=N_PDE, N_term=N_TERM, N_spat=N_SPAT, N_free=N_FREE, N_intrinsic=N_INTRINSIC,
        resample_every=1000,
        w_intrinsic=0.0,
    )

    history2 = train_refined_model(
        priceNN, freeBoundaryNN, data_pde, data_term, data_spat, data_free, data_free_term, data_intrinsic,
        epochs=7000, batch_size=BATCH_SIZE,
        s_bounds=S_BOUNDS, v_bounds=V_BOUNDS, tau_bounds=TAU_BOUNDS,
        power_s=POWER_S, power_tau=POWER_TAU,
        N_pde=N_PDE, N_term=N_TERM, N_spat=N_SPAT, N_free=N_FREE, N_free_term=N_FREE_TERM, N_intrinsic=N_INTRINSIC,
        resample_every=1000,
        w_intrinsic=0.0,
    )

    save_dir = "./saved_models_ablation_no_intrinsic"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(priceNN.state_dict(), os.path.join(save_dir, "solutionNN.pth"))
    torch.save(freeBoundaryNN.state_dict(), os.path.join(save_dir, "boundaryNN.pth"))
    print(f"Models saved to '{save_dir}/'")

    df_results = evaluate_and_compare(priceNN, freeBoundaryNN, device, K=K_ref, T_maturity=T_ref)
    plot_training_history(history, history2)

if __name__ == '__main__':
    main()
