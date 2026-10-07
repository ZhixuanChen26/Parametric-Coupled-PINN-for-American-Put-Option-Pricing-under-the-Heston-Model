import torch
from torch.autograd import grad
from config import r


def get_PDE_residual(s, v, tau, params_phys, params_norm, target_zero, price_net):
    kappa, theta, sigma_v, rho, T_s = params_phys
    kappa_hat, theta_hat, sigma_v_hat, rho_hat, T_s_n = params_norm

    X = torch.cat([s, v, tau, kappa_hat, theta_hat, sigma_v_hat, rho_hat], dim=1)
    p = price_net(X)

    grads = torch.autograd.grad(outputs=p, inputs=[tau, s, v],
                                 grad_outputs=torch.ones_like(p),
                                 create_graph=True, retain_graph=True)
    p_tau, p_s, p_v = grads[0], grads[1], grads[2]

    grads_s = torch.autograd.grad(outputs=p_s, inputs=[s, v],
                                   grad_outputs=torch.ones_like(p_s),
                                   create_graph=True, retain_graph=True)
    p_ss, p_sv = grads_s[0], grads_s[1]

    p_vv = torch.autograd.grad(outputs=p_v, inputs=v,
                                grad_outputs=torch.ones_like(p_v),
                                create_graph=True)[0]

    # Written in terms of remaining time to maturity: tau = T - t.
    # Heston's PDE is time-homogeneous (kappa/theta/sigma_v/rho/r don't
    # depend on calendar time), so under this change of variables from the
    # fractional-time parameterization (t/T), the (1/T)*d/d(t/T) term
    # becomes exactly -p_tau -- T drops out entirely, it no longer needs
    # to be a network input (see models.py: priceNN is now 7-dim, not 8).
    pde_val = (
        -p_tau
        + 0.5 * v * (s ** 2) * p_ss
        + rho * sigma_v * v * s * p_sv
        + 0.5 * (sigma_v ** 2) * v * p_vv
        + r * s * p_s
        + kappa * (theta - v) * p_v
        - r * p
    )
    return pde_val - target_zero


def get_terminal_residual(s, v, tau, params_norm, target_payoff_norm, price_net):
    kappa_hat, theta_hat, sigma_v_hat, rho_hat, T_s_n = params_norm
    X = torch.cat([s, v, tau, kappa_hat, theta_hat, sigma_v_hat, rho_hat], dim=1)
    p_pred = price_net(X)
    return p_pred - target_payoff_norm


def get_B_residual(s, v, tau, params_norm, target_norm, price_net):
    kappa_hat, theta_hat, sigma_v_hat, rho_hat, T_s_n = params_norm
    X = torch.cat([s, v, tau, kappa_hat, theta_hat, sigma_v_hat, rho_hat], dim=1)
    p_pred = price_net(X)
    return p_pred - target_norm


def get_intrinsic_residual(s, v, tau, params_norm, target_intrinsic, price_net):
    kappa_hat, theta_hat, sigma_v_hat, rho_hat, T_s_n = params_norm
    X = torch.cat([s, v, tau, kappa_hat, theta_hat, sigma_v_hat, rho_hat], dim=1)
    p_pred = price_net(X)
    return p_pred - target_intrinsic


def get_VM_SP_residuals(v, tau, params_norm, target_zero, price_net, free_boundary_net):
    kappa_hat, theta_hat, sigma_v_hat, rho_hat, T_s_n = params_norm

    X_bound = torch.cat([v, tau, kappa_hat, theta_hat, sigma_v_hat, rho_hat], dim=1)
    s_star = free_boundary_net(X_bound)

    X_sol = torch.cat([s_star, v, tau, kappa_hat, theta_hat, sigma_v_hat, rho_hat], dim=1)
    p_at_boundary = price_net(X_sol)

    p_s_at_boundary = grad(p_at_boundary, s_star,
                            grad_outputs=torch.ones_like(p_at_boundary),
                            create_graph=True)[0]

    res_VM = p_at_boundary - (1.0 - s_star) - target_zero
    res_SP = p_s_at_boundary - (-1.0) - target_zero

    return res_VM, res_SP


def get_BT_residual(v, tau, params_norm, target_s_star_norm, free_boundary_net):
    kappa_hat, theta_hat, sigma_v_hat, rho_hat, T_s_n = params_norm
    X_bound = torch.cat([v, tau, kappa_hat, theta_hat, sigma_v_hat, rho_hat], dim=1)
    s_star = free_boundary_net(X_bound)
    return s_star - target_s_star_norm
