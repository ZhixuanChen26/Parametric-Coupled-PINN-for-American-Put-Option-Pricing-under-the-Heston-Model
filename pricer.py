import torch
from config import DTYPE_TORCH, device, T_s_ref as T_ref, K_ref, kappa_ref, theta_ref, sigma_v_ref, rho_ref
from param_sampling import normalize_params


def _to_tensor_col(x, dtype=DTYPE_TORCH, device=device):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=dtype, device=device)
    if x.dim() == 0:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 1:
        x = x.unsqueeze(1)
    return x


def american_put_price(S, v, t, price_net, free_boundary_net,
                        K=K_ref, T=T_ref,
                        kappa=kappa_ref, theta=theta_ref,
                        sigma_v=sigma_v_ref, rho=rho_ref):
    S = _to_tensor_col(S)
    v = _to_tensor_col(v)
    t = _to_tensor_col(t)

    N = S.shape[0]
    kappa_t   = kappa   if isinstance(kappa,   torch.Tensor) else torch.full((N, 1), float(kappa),   dtype=DTYPE_TORCH, device=device)
    theta_t   = theta   if isinstance(theta,   torch.Tensor) else torch.full((N, 1), float(theta),   dtype=DTYPE_TORCH, device=device)
    sigma_v_t = sigma_v if isinstance(sigma_v, torch.Tensor) else torch.full((N, 1), float(sigma_v), dtype=DTYPE_TORCH, device=device)
    rho_t     = rho     if isinstance(rho,     torch.Tensor) else torch.full((N, 1), float(rho),     dtype=DTYPE_TORCH, device=device)
    T_t       = T       if isinstance(T,       torch.Tensor) else torch.full((N, 1), float(T),       dtype=DTYPE_TORCH, device=device)

    price_net.eval()
    free_boundary_net.eval()

    with torch.no_grad():
        s     = S / K
        tau = T_t - t  # remaining time to maturity (see loss.py)

        kappa_hat, theta_hat, sigma_v_hat, rho_hat, T_s_n = normalize_params(kappa_t, theta_t, sigma_v_t, rho_t, T_t)

        X_bound = torch.cat([v, tau, kappa_hat, theta_hat, sigma_v_hat, rho_hat], dim=1)
        s_star = free_boundary_net(X_bound)

        X_sol = torch.cat([s, v, tau, kappa_hat, theta_hat, sigma_v_hat, rho_hat], dim=1)
        p_continuation = price_net(X_sol)

        p_intrinsic = 1.0 - s
        price_norm  = torch.where(s < s_star, p_intrinsic, p_continuation)
        # No-arbitrage bounds: intrinsic value from below, strike (p=1 in
        # normalized units) from above -- an American put can never be worth
        # more than K.
        price_norm  = torch.clamp(price_norm, min=torch.relu(p_intrinsic)).clamp(max=1.0)

        price = price_norm * K
        S_star = s_star * K

    return price, S_star
