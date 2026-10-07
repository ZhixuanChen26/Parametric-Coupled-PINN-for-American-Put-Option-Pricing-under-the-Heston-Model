import torch
import time
from tqdm.auto import tqdm
from loss import (get_PDE_residual, get_terminal_residual,
                  get_B_residual, get_VM_SP_residuals,
                  get_BT_residual, get_intrinsic_residual)
from data_generation import generate_all_training_data
from config import PARAM_CURRICULUM_STAGES


def get_curriculum_width(epoch, total_epochs):
    n_stages = len(PARAM_CURRICULUM_STAGES)
    if n_stages == 1:
        return PARAM_CURRICULUM_STAGES[0]
    progress = epoch / max(total_epochs - 1, 1)
    stage_pos = progress * (n_stages - 1)
    idx_lo = int(stage_pos)
    idx_hi = min(idx_lo + 1, n_stages - 1)
    frac = stage_pos - idx_lo
    return PARAM_CURRICULUM_STAGES[idx_lo] + frac * (PARAM_CURRICULUM_STAGES[idx_hi] - PARAM_CURRICULUM_STAGES[idx_lo])


def train_initial_model(price_net, free_boundary_net, pde_tensors, term_tensors, spat_tensors, free_tensors, intrinsic_tensors,
                        epochs=2000, batch_size=8192,
                        s_bounds=None, v_bounds=None, tau_bounds=None,
                        power_s=3.0, power_tau=2.0,
                        N_pde=80000, N_term=10000, N_spat=10000, N_free=40000, N_intrinsic=40000,
                        resample_every=500,
                        w_PDE=15.0, w_term=15.0, w_B=5.0, w_VM=15.0, w_SP=5.0, w_intrinsic=10.0):
    s_pde_all, v_pde_all, tau_pde_all, pp_pde_all, pn_pde_all, y_pde_all = pde_tensors
    s_term_all, v_term_all, tau_term_all, pp_term_all, pn_term_all, y_term_all = term_tensors
    s_spat_all, v_spat_all, tau_spat_all, pp_spat_all, pn_spat_all, y_spat_all = spat_tensors
    v_free_all, tau_free_all, pp_free_all, pn_free_all, y_free_all = free_tensors
    s_intrinsic_all, v_intrinsic_all, tau_intrinsic_all, pp_intrinsic_all, pn_intrinsic_all, y_intrinsic_all = intrinsic_tensors

    device = s_pde_all.device
    N_term_current = s_term_all.shape[0]
    N_spat_current = s_spat_all.shape[0]
    N_free_current = v_free_all.shape[0]
    N_intrinsic_current = s_intrinsic_all.shape[0]

    optimizer_sol   = torch.optim.Adam(price_net.parameters(), lr=1e-3)
    optimizer_bound = torch.optim.Adam(free_boundary_net.parameters(), lr=1e-3)

    # Learning-rate warmup on price_net: ramp lr linearly from 1e-5 to the
    # 1e-3 target over the first LR_WARMUP_EPOCHS, then hand over to StepLR.
    # Smooths the large loss spike at random-init startup. NOTE: distinct from
    # WARMUP_EPOCHS below, which is the boundary-net freeze period, not an lr.
    LR_WARMUP_EPOCHS = 500
    scheduler_sol = torch.optim.lr_scheduler.SequentialLR(
        optimizer_sol,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer_sol, start_factor=0.01, total_iters=LR_WARMUP_EPOCHS),
            torch.optim.lr_scheduler.StepLR(optimizer_sol, step_size=1000, gamma=0.9),
        ],
        milestones=[LR_WARMUP_EPOCHS],
    )

    history = {'total_loss': [], 'PDE_loss': [], 'term_loss': [],
               'B_loss': [], 'VM_loss': [], 'SP_loss': [], 'intrinsic_loss': [],
               'update_boundary': []}

    start_time = time.time()
    pbar = tqdm(range(epochs), desc='Training Heston PINN (Phase 1 & 2)', unit='epoch')
    WARMUP_EPOCHS = 1000

    for epoch in pbar:
        width_fraction = get_curriculum_width(epoch, epochs)

        if epoch > 0 and epoch % resample_every == 0 and s_bounds is not None:
            tqdm.write(f'=== Resampling at epoch {epoch} (param width={width_fraction:.3f}) ===')
            new_data = generate_all_training_data(
                free_boundary_net=free_boundary_net,
                N_pde=N_pde, N_term=N_term, N_spat=N_spat,
                N_free=N_free, N_free_term=1, N_intrinsic=N_intrinsic,
                s_bounds=s_bounds, v_bounds=v_bounds, tau_bounds=tau_bounds,
                power_s=power_s, power_tau=power_tau, width_fraction=width_fraction
            )
            s_pde_all, v_pde_all, tau_pde_all, pp_pde_all, pn_pde_all, y_pde_all = new_data['pde']
            s_term_all, v_term_all, tau_term_all, pp_term_all, pn_term_all, y_term_all = new_data['term']
            s_spat_all, v_spat_all, tau_spat_all, pp_spat_all, pn_spat_all, y_spat_all = new_data['spat']
            v_free_all, tau_free_all, pp_free_all, pn_free_all, y_free_all = new_data['free']
            s_intrinsic_all, v_intrinsic_all, tau_intrinsic_all, pp_intrinsic_all, pn_intrinsic_all, y_intrinsic_all = new_data['intrinsic']
            N_term_current = s_term_all.shape[0]
            N_spat_current = s_spat_all.shape[0]
            N_free_current = v_free_all.shape[0]
            N_intrinsic_current = s_intrinsic_all.shape[0]

        epoch_loss_total = 0.0
        epoch_PDE_total = 0.0
        epoch_term_total = 0.0
        epoch_B_total = 0.0
        epoch_VM_total = 0.0
        epoch_SP_total = 0.0
        epoch_intrinsic_total = 0.0
        update_boundary = (epoch >= WARMUP_EPOCHS) and ((epoch - WARMUP_EPOCHS) % 11 == 10)

        free_boundary_net.eval()
        with torch.no_grad():
            X_bound_all = torch.cat([v_pde_all, tau_pde_all, *pn_pde_all[:4]], dim=1)
            s_star_all = free_boundary_net(X_bound_all)
            valid_mask = (s_pde_all > s_star_all).flatten()
            valid_indices = torch.nonzero(valid_mask).flatten()
            valid_indices = valid_indices[torch.randperm(len(valid_indices))]
        free_boundary_net.train()

        num_valid_points = len(valid_indices)
        num_batches = 0

        for i in range(0, num_valid_points, batch_size):
            num_batches += 1
            idx_batch = valid_indices[i: i + batch_size]

            s_pde   = s_pde_all[idx_batch].detach().requires_grad_(True)
            v_pde   = v_pde_all[idx_batch].detach().requires_grad_(True)
            tau_pde = tau_pde_all[idx_batch].detach().requires_grad_(True)
            pp_pde  = tuple(p[idx_batch].detach() for p in pp_pde_all)
            pn_pde  = tuple(p[idx_batch].detach() for p in pn_pde_all)
            y_pde   = y_pde_all[idx_batch].detach()

            idx_term = torch.randint(0, N_term_current, (batch_size,), device=device)
            idx_spat = torch.randint(0, N_spat_current, (batch_size,), device=device)
            idx_free = torch.randint(0, N_free_current, (batch_size,), device=device)
            idx_intrinsic  = torch.randint(0, N_intrinsic_current,  (batch_size,), device=device)

            s_term   = s_term_all[idx_term].detach()
            v_term   = v_term_all[idx_term].detach()
            tau_term = tau_term_all[idx_term].detach()
            pn_term  = tuple(p[idx_term].detach() for p in pn_term_all)
            y_term   = y_term_all[idx_term].detach()

            s_spat   = s_spat_all[idx_spat].detach()
            v_spat   = v_spat_all[idx_spat].detach()
            tau_spat = tau_spat_all[idx_spat].detach()
            pn_spat  = tuple(p[idx_spat].detach() for p in pn_spat_all)
            y_spat   = y_spat_all[idx_spat].detach()

            v_free   = v_free_all[idx_free].detach().requires_grad_(True)
            tau_free = tau_free_all[idx_free].detach().requires_grad_(True)
            pn_free  = tuple(p[idx_free].detach() for p in pn_free_all)
            y_free   = y_free_all[idx_free].detach()

            s_intrinsic   = s_intrinsic_all[idx_intrinsic].detach()
            v_intrinsic   = v_intrinsic_all[idx_intrinsic].detach()
            tau_intrinsic = tau_intrinsic_all[idx_intrinsic].detach()
            pn_intrinsic  = tuple(p[idx_intrinsic].detach() for p in pn_intrinsic_all)
            y_intrinsic   = y_intrinsic_all[idx_intrinsic].detach()

            optimizer_sol.zero_grad()
            if update_boundary:
                optimizer_bound.zero_grad()

            res_PDE     = get_PDE_residual(s_pde, v_pde, tau_pde, pp_pde, pn_pde, y_pde, price_net)
            loss_PDE    = torch.mean(res_PDE ** 2)

            res_term    = get_terminal_residual(s_term, v_term, tau_term, pn_term, y_term, price_net)
            loss_term   = torch.mean(res_term ** 2)

            res_B = get_B_residual(s_spat, v_spat, tau_spat, pn_spat, y_spat, price_net)
            loss_B = torch.mean(res_B ** 2)

            res_VM, res_SP = get_VM_SP_residuals(v_free, tau_free, pn_free, y_free, price_net, free_boundary_net)
            loss_VM = torch.mean(res_VM ** 2)
            loss_SP = torch.mean(res_SP ** 2)

            res_intrinsic     = get_intrinsic_residual(s_intrinsic, v_intrinsic, tau_intrinsic, pn_intrinsic, y_intrinsic, price_net)
            loss_intrinsic    = torch.mean(res_intrinsic ** 2)

            total_loss_sol = (w_PDE * loss_PDE + w_term * loss_term +
                              w_B * loss_B + w_VM * loss_VM + w_SP * loss_SP +
                              w_intrinsic * loss_intrinsic)
            total_loss_boundary = w_VM * loss_VM + w_SP * loss_SP
            monitor_loss = total_loss_sol

            if update_boundary:
                total_loss_boundary.backward()
                optimizer_bound.step()
            else:
                total_loss_sol.backward()
                optimizer_sol.step()

            epoch_loss_total += monitor_loss.item()
            epoch_PDE_total += loss_PDE.item()
            epoch_term_total += loss_term.item()
            epoch_B_total += loss_B.item()
            epoch_VM_total += loss_VM.item()
            epoch_SP_total += loss_SP.item()
            epoch_intrinsic_total += loss_intrinsic.item()

        scheduler_sol.step()

        avg_loss = epoch_loss_total / max(num_batches, 1)
        avg_PDE = epoch_PDE_total / max(num_batches, 1)
        avg_term = epoch_term_total / max(num_batches, 1)
        avg_B = epoch_B_total / max(num_batches, 1)
        avg_VM = epoch_VM_total / max(num_batches, 1)
        avg_SP = epoch_SP_total / max(num_batches, 1)
        avg_intrinsic = epoch_intrinsic_total / max(num_batches, 1)
        pbar.set_postfix({'Loss': f'{avg_loss:.5f}', 'BoundUpd': update_boundary, 'W': f'{width_fraction:.2f}'})

        if epoch % 20 == 0:
            msg = (f"Ep {epoch:04d} | Total: {avg_loss:.5f} | "
                   f"PDE: {avg_PDE:.5f} | Term: {avg_term:.5f} | "
                   f"B: {avg_B:.5f} | VM: {avg_VM:.5f} | "
                   f"SP: {avg_SP:.5f} | Intrinsic: {avg_intrinsic:.5f} | ParamW: {width_fraction:.3f}")
            tqdm.write(msg)
            history['total_loss'].append(avg_loss)
            history['PDE_loss'].append(avg_PDE)
            history['term_loss'].append(avg_term)
            history['B_loss'].append(avg_B)
            history['VM_loss'].append(avg_VM)
            history['SP_loss'].append(avg_SP)
            history['intrinsic_loss'].append(avg_intrinsic)
            history['update_boundary'].append(update_boundary)

    print(f'Training phase 1 completed in {time.time() - start_time:.2f} seconds.')
    return history


def train_refined_model(price_net, free_boundary_net, pde_tensors, term_tensors, spat_tensors, free_tensors, free_term_tensors, intrinsic_tensors,
                        epochs=2000, batch_size=8192,
                        s_bounds=None, v_bounds=None, tau_bounds=None,
                        power_s=3.0, power_tau=2.0,
                        N_pde=80000, N_term=10000, N_spat=10000, N_free=40000, N_free_term=10000, N_intrinsic=40000,
                        resample_every=500,
                        w_PDE=15.0, w_term=15.0, w_B=5.0, w_VM=15.0, w_SP=5.0, w_BT=2.0, w_intrinsic=10.0):
    s_pde_all, v_pde_all, tau_pde_all, pp_pde_all, pn_pde_all, y_pde_all = pde_tensors
    s_term_all, v_term_all, tau_term_all, pp_term_all, pn_term_all, y_term_all = term_tensors
    s_spat_all, v_spat_all, tau_spat_all, pp_spat_all, pn_spat_all, y_spat_all = spat_tensors
    v_free_all, tau_free_all, pp_free_all, pn_free_all, y_free_all = free_tensors
    v_free_term_all, tau_free_term_all, pp_free_term_all, pn_free_term_all, y_free_term_all = free_term_tensors
    s_intrinsic_all, v_intrinsic_all, tau_intrinsic_all, pp_intrinsic_all, pn_intrinsic_all, y_intrinsic_all = intrinsic_tensors

    device = s_pde_all.device
    N_term_current     = s_term_all.shape[0]
    N_spat_current     = s_spat_all.shape[0]
    N_free_current     = v_free_all.shape[0]
    N_free_term_current = v_free_term_all.shape[0]
    N_intrinsic_current       = s_intrinsic_all.shape[0]

    optimizer_sol   = torch.optim.Adam(price_net.parameters(), lr=5e-4)
    optimizer_bound = torch.optim.Adam(free_boundary_net.parameters(), lr=5e-4)

    # Shorter lr warmup for Phase 3: the net is already trained, but a fresh
    # optimizer, the lower 5e-4 target, and the newly-introduced free-term
    # loss create a transition spike -- ramp 5e-5 -> 5e-4 over 300 epochs.
    LR_WARMUP_EPOCHS = 300
    scheduler_sol = torch.optim.lr_scheduler.SequentialLR(
        optimizer_sol,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer_sol, start_factor=0.1, total_iters=LR_WARMUP_EPOCHS),
            torch.optim.lr_scheduler.StepLR(optimizer_sol, step_size=1000, gamma=0.9),
        ],
        milestones=[LR_WARMUP_EPOCHS],
    )

    history = {'total_loss': [], 'PDE_loss': [], 'term_loss': [],
               'B_loss': [], 'VM_loss': [], 'SP_loss': [], 'BT_loss': [], 'intrinsic_loss': [],
               'update_boundary': []}

    start_time = time.time()
    pbar = tqdm(range(epochs), desc='Training Heston PINN (Phase 3)', unit='epoch')

    for epoch in pbar:
        width_fraction = PARAM_CURRICULUM_STAGES[-1]

        if epoch % resample_every == 0 and s_bounds is not None:
            tqdm.write(f'=== Resampling at epoch {epoch} (param width={width_fraction:.3f}) ===')
            new_data = generate_all_training_data(
                free_boundary_net=free_boundary_net,
                N_pde=N_pde, N_term=N_term, N_spat=N_spat,
                N_free=N_free, N_free_term=N_free_term, N_intrinsic=N_intrinsic,
                s_bounds=s_bounds, v_bounds=v_bounds, tau_bounds=tau_bounds,
                power_s=power_s, power_tau=power_tau, width_fraction=width_fraction
            )
            s_pde_all, v_pde_all, tau_pde_all, pp_pde_all, pn_pde_all, y_pde_all = new_data['pde']
            s_term_all, v_term_all, tau_term_all, pp_term_all, pn_term_all, y_term_all = new_data['term']
            s_spat_all, v_spat_all, tau_spat_all, pp_spat_all, pn_spat_all, y_spat_all = new_data['spat']
            v_free_all, tau_free_all, pp_free_all, pn_free_all, y_free_all = new_data['free']
            v_free_term_all, tau_free_term_all, pp_free_term_all, pn_free_term_all, y_free_term_all = new_data['free_term']
            s_intrinsic_all, v_intrinsic_all, tau_intrinsic_all, pp_intrinsic_all, pn_intrinsic_all, y_intrinsic_all = new_data['intrinsic']
            N_term_current      = s_term_all.shape[0]
            N_spat_current      = s_spat_all.shape[0]
            N_free_current      = v_free_all.shape[0]
            N_free_term_current = v_free_term_all.shape[0]
            N_intrinsic_current       = s_intrinsic_all.shape[0]

        epoch_loss_total = 0.0
        epoch_PDE_total = 0.0
        epoch_term_total = 0.0
        epoch_B_total = 0.0
        epoch_VM_total = 0.0
        epoch_SP_total = 0.0
        epoch_BT_total = 0.0
        epoch_intrinsic_total = 0.0
        update_boundary = (epoch % 11 == 10)

        free_boundary_net.eval()
        with torch.no_grad():
            X_bound_all = torch.cat([v_pde_all, tau_pde_all, *pn_pde_all[:4]], dim=1)
            s_star_all = free_boundary_net(X_bound_all)
            valid_mask = (s_pde_all > s_star_all).flatten()
            valid_indices = torch.nonzero(valid_mask).flatten()
            valid_indices = valid_indices[torch.randperm(len(valid_indices))]
        free_boundary_net.train()

        num_valid_points = len(valid_indices)
        num_batches = 0

        for i in range(0, num_valid_points, batch_size):
            num_batches += 1
            idx_batch = valid_indices[i: i + batch_size]

            s_pde   = s_pde_all[idx_batch].detach().requires_grad_(True)
            v_pde   = v_pde_all[idx_batch].detach().requires_grad_(True)
            tau_pde = tau_pde_all[idx_batch].detach().requires_grad_(True)
            pp_pde  = tuple(p[idx_batch].detach() for p in pp_pde_all)
            pn_pde  = tuple(p[idx_batch].detach() for p in pn_pde_all)
            y_pde   = y_pde_all[idx_batch].detach()

            idx_term      = torch.randint(0, N_term_current,      (batch_size,), device=device)
            idx_spat      = torch.randint(0, N_spat_current,      (batch_size,), device=device)
            idx_free      = torch.randint(0, N_free_current,      (batch_size,), device=device)
            idx_free_term = torch.randint(0, N_free_term_current, (batch_size,), device=device)
            idx_intrinsic       = torch.randint(0, N_intrinsic_current,       (batch_size,), device=device)

            s_term   = s_term_all[idx_term].detach()
            v_term   = v_term_all[idx_term].detach()
            tau_term = tau_term_all[idx_term].detach()
            pn_term  = tuple(p[idx_term].detach() for p in pn_term_all)
            y_term   = y_term_all[idx_term].detach()

            s_spat   = s_spat_all[idx_spat].detach()
            v_spat   = v_spat_all[idx_spat].detach()
            tau_spat = tau_spat_all[idx_spat].detach()
            pn_spat  = tuple(p[idx_spat].detach() for p in pn_spat_all)
            y_spat   = y_spat_all[idx_spat].detach()

            v_free   = v_free_all[idx_free].detach().requires_grad_(True)
            tau_free = tau_free_all[idx_free].detach().requires_grad_(True)
            pn_free  = tuple(p[idx_free].detach() for p in pn_free_all)
            y_free   = y_free_all[idx_free].detach()

            v_free_term   = v_free_term_all[idx_free_term].detach()
            tau_free_term = tau_free_term_all[idx_free_term].detach()
            pn_free_term  = tuple(p[idx_free_term].detach() for p in pn_free_term_all)
            y_free_term   = y_free_term_all[idx_free_term].detach()

            s_intrinsic   = s_intrinsic_all[idx_intrinsic].detach()
            v_intrinsic   = v_intrinsic_all[idx_intrinsic].detach()
            tau_intrinsic = tau_intrinsic_all[idx_intrinsic].detach()
            pn_intrinsic  = tuple(p[idx_intrinsic].detach() for p in pn_intrinsic_all)
            y_intrinsic   = y_intrinsic_all[idx_intrinsic].detach()

            optimizer_sol.zero_grad()
            if update_boundary:
                optimizer_bound.zero_grad()

            res_PDE      = get_PDE_residual(s_pde, v_pde, tau_pde, pp_pde, pn_pde, y_pde, price_net)
            loss_PDE     = torch.mean(res_PDE ** 2)

            res_term     = get_terminal_residual(s_term, v_term, tau_term, pn_term, y_term, price_net)
            loss_term    = torch.mean(res_term ** 2)

            res_B  = get_B_residual(s_spat, v_spat, tau_spat, pn_spat, y_spat, price_net)
            loss_B = torch.mean(res_B ** 2)

            res_VM, res_SP = get_VM_SP_residuals(v_free, tau_free, pn_free, y_free, price_net, free_boundary_net)
            loss_VM = torch.mean(res_VM ** 2)
            loss_SP = torch.mean(res_SP ** 2)

            res_BT = get_BT_residual(v_free_term, tau_free_term, pn_free_term, y_free_term, free_boundary_net)
            loss_BT = torch.mean(res_BT ** 2)

            res_intrinsic      = get_intrinsic_residual(s_intrinsic, v_intrinsic, tau_intrinsic, pn_intrinsic, y_intrinsic, price_net)
            loss_intrinsic     = torch.mean(res_intrinsic ** 2)

            total_loss_sol = (w_PDE * loss_PDE + w_term * loss_term +
                              w_B * loss_B + w_VM * loss_VM + w_SP * loss_SP +
                              w_intrinsic * loss_intrinsic)
            total_loss_boundary = (w_VM * loss_VM +
                                   w_SP * loss_SP + w_BT * loss_BT)
            monitor_loss = total_loss_sol + w_BT * loss_BT

            if update_boundary:
                total_loss_boundary.backward()
                optimizer_bound.step()
            else:
                total_loss_sol.backward()
                optimizer_sol.step()

            epoch_loss_total += monitor_loss.item()
            epoch_PDE_total += loss_PDE.item()
            epoch_term_total += loss_term.item()
            epoch_B_total += loss_B.item()
            epoch_VM_total += loss_VM.item()
            epoch_SP_total += loss_SP.item()
            epoch_BT_total += loss_BT.item()
            epoch_intrinsic_total += loss_intrinsic.item()

        scheduler_sol.step()

        avg_loss = epoch_loss_total / max(num_batches, 1)
        avg_PDE = epoch_PDE_total / max(num_batches, 1)
        avg_term = epoch_term_total / max(num_batches, 1)
        avg_B = epoch_B_total / max(num_batches, 1)
        avg_VM = epoch_VM_total / max(num_batches, 1)
        avg_SP = epoch_SP_total / max(num_batches, 1)
        avg_BT = epoch_BT_total / max(num_batches, 1)
        avg_intrinsic = epoch_intrinsic_total / max(num_batches, 1)
        pbar.set_postfix({'Loss': f'{avg_loss:.5f}', 'BoundUpd': update_boundary})

        if epoch % 20 == 0:
            msg = (f"Ep {epoch:04d} | Total: {avg_loss:.5f} | "
                   f"PDE: {avg_PDE:.5f} | Term: {avg_term:.5f} | "
                   f"B: {avg_B:.5f} | VM: {avg_VM:.5f} | "
                   f"SP: {avg_SP:.5f} | BT: {avg_BT:.5f} | "
                   f"Int: {avg_intrinsic:.5f}")
            tqdm.write(msg)
            history['total_loss'].append(avg_loss)
            history['PDE_loss'].append(avg_PDE)
            history['term_loss'].append(avg_term)
            history['B_loss'].append(avg_B)
            history['VM_loss'].append(avg_VM)
            history['SP_loss'].append(avg_SP)
            history['BT_loss'].append(avg_BT)
            history['intrinsic_loss'].append(avg_intrinsic)
            history['update_boundary'].append(update_boundary)

    print(f'Training phase 3 completed in {time.time() - start_time:.2f} seconds.')
    return history
