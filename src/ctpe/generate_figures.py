"""Generate experiment figures for the high-order generator time-dependent manuscript.

This script reproduces the additional figures introduced in the v5 revision:
  - figures/convergence_dt.pdf
  - figures/ou10_scaling_M.pdf
  - figures/lqr_scaling_M.pdf

It uses only numpy and matplotlib.

Run from the project root:
  python scripts/generate_figures.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


def a_coeffs(i: int) -> np.ndarray:
    """Finite-difference coefficients a_0..a_i satisfying the moment conditions.

    Sum_{j=0}^i a_j j^k = 1 if k=1, and 0 for k=0,2,...,i.
    """
    j = np.arange(i + 1)
    A = np.vander(j, N=i + 1, increasing=True).T  # rows k, cols j
    b = np.zeros(i + 1)
    b[1] = 1.0
    return np.linalg.solve(A, b)


# ------------------------------------------------------------
# Figure: convergence_dt.pdf
# ------------------------------------------------------------

def sigma_t(t: np.ndarray, sigma0: float = 1.0, amp: float = 0.5, omega: float = 8.0) -> np.ndarray:
    return sigma0 + amp * np.sin(omega * t)


def Sigma_t(t: np.ndarray, sigma0: float = 1.0, amp: float = 0.5, omega: float = 8.0) -> np.ndarray:
    s = sigma_t(t, sigma0, amp, omega)
    return s * s


def alpha_true(t: np.ndarray, T: float = 1.0, beta: float = 1.0) -> np.ndarray:
    # Solution of alpha'(t) = beta alpha(t) - 1 with terminal alpha(T)=0.
    return (1.0 / beta) * (1.0 - np.exp(-beta * (T - t)))


def Sigma_hat_i(t_grid: np.ndarray, Dt: float, i: int, sigma0: float = 1.0, amp: float = 0.5, omega: float = 8.0) -> np.ndarray:
    """Population generator-based diffusion approximation for dS = sigma(t) dW.

    For each t, Sigma_hat_i(t) = (1/Dt) sum_{j=1}^i a_j E[(S_{t+jDt}-S_t)^2].
    Here E[(S_{t+u}-S_t)^2] = integral_0^u Sigma(t+s) ds.
    """
    coeffs = a_coeffs(i)
    sub = 200  # trapezoid sub-steps per Dt
    out = np.zeros_like(t_grid)
    for j in range(1, i + 1):
        aj = coeffs[j]
        u = j * Dt
        M = sub * j + 1
        us = np.linspace(0.0, u, M)
        vals = Sigma_t(t_grid[:, None] + us[None, :], sigma0=sigma0, amp=amp, omega=omega)
        integ = np.trapezoid(vals, us, axis=1)
        out += (aj / Dt) * integ
    return out


def kappa_from_Sigma(t_grid: np.ndarray, Sigma_vals: np.ndarray, T: float = 1.0, beta: float = 1.0) -> np.ndarray:
    """Compute kappa(t) by backward recursion on a uniform grid."""
    alpha_vals = alpha_true(t_grid, T=T, beta=beta)
    integrand = Sigma_vals * alpha_vals
    dt = float(t_grid[1] - t_grid[0])
    N = len(t_grid)
    kappa = np.zeros(N)
    for idx in range(N - 2, -1, -1):
        t = float(t_grid[idx])
        tau1 = float(t_grid[idx + 1])
        f0 = float(integrand[idx])
        f1 = float(integrand[idx + 1])
        w0 = 1.0
        w1 = math.exp(-beta * (tau1 - t))
        interval = 0.5 * dt * (w0 * f0 + w1 * f1)
        kappa[idx] = interval + math.exp(-beta * dt) * kappa[idx + 1]
    return kappa


def make_convergence_dt(out_pdf: str) -> None:
    T = 1.0
    beta = 1.0
    # Use small Dt values to clearly reveal asymptotic rates.
    Dt_list = np.array([0.025, 0.0125, 0.00625, 0.003125, 0.0015625])

    t_fine = np.linspace(0.0, T, 20001)
    Sigma_true_vals = Sigma_t(t_fine)
    kappa_true = kappa_from_Sigma(t_fine, Sigma_true_vals, T=T, beta=beta)
    V_true_0 = float(alpha_true(np.array([0.0]), T=T, beta=beta)[0] + kappa_true[0])

    errs = {}
    for i in [1, 2, 3]:
        e = []
        for Dt in Dt_list:
            Sigma_hat_vals = Sigma_hat_i(t_fine, Dt, i)
            kappa_hat = kappa_from_Sigma(t_fine, Sigma_hat_vals, T=T, beta=beta)
            V_hat_0 = float(alpha_true(np.array([0.0]), T=T, beta=beta)[0] + kappa_hat[0])
            e.append(abs(V_hat_0 - V_true_0))
        errs[i] = np.array(e)

    plt.figure()
    for i, marker in zip([1, 2, 3], ["o", "s", "^"]):
        plt.loglog(Dt_list, errs[i], marker=marker, label=f"order i={i}")

    # Reference slopes
    x0 = float(Dt_list[-1])
    y0 = float(errs[1][-1])
    for p in [1, 2, 3]:
        plt.loglog(Dt_list, y0 * (Dt_list / x0) ** p, "--", label=f"ref slope {p}")

    plt.xlabel(r"$\Delta t$")
    plt.ylabel(r"$|V_i(1,0)-V(1,0)|$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


# ------------------------------------------------------------
# Figure: ou10_scaling_M.pdf
# ------------------------------------------------------------

def simulate_ou(d: int, M: int, N: int, Dt: float, theta_func, sigma: float, seed: int = 0, extra: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    S = np.zeros((M, N + extra + 1, d))
    S[:, 0, :] = rng.normal(size=(M, d))
    for n in range(N + extra):
        t = n * Dt
        theta = float(theta_func(t))
        S[:, n + 1, :] = S[:, n, :] + (-theta * S[:, n, :]) * Dt + sigma * math.sqrt(Dt) * rng.normal(size=(M, d))
    return S


def phi_diag_squares(s: np.ndarray) -> np.ndarray:
    d = s.shape[-1]
    return np.concatenate([np.ones((*s.shape[:-1], 1)), s ** 2], axis=-1)


def run_high_order_generator_from_trajectories(S: np.ndarray, Dt: float, beta: float, i: int = 2) -> np.ndarray:
    M, Tsteps, d = S.shape
    extra = i
    N_T = Tsteps - 1 - extra
    Phi = phi_diag_squares(S)
    p = d + 1
    a = a_coeffs(i)
    w = np.zeros((N_T + 1, p))
    for n in range(N_T - 1, -1, -1):
        phi_n = Phi[:, n, :]
        G = (phi_n.T @ phi_n) / M
        r = np.sum(S[:, n, :] ** 2, axis=1)
        b = (phi_n.T @ r) / M
        delta_phi = np.zeros_like(phi_n)
        for j in range(i + 1):
            delta_phi += a[j] * Phi[:, n + j, :]
        A = (phi_n.T @ (delta_phi / Dt)) / M
        Mmat = (beta + 1.0 / Dt) * G - A
        rhs = b + (1.0 / Dt) * (G @ w[n + 1])
        w[n] = np.linalg.solve(Mmat, rhs)
    return w


def run_td_be_from_trajectories(S: np.ndarray, Dt: float, beta: float) -> np.ndarray:
    M, Tsteps, d = S.shape
    N_T = Tsteps - 1
    Phi = phi_diag_squares(S[:, : N_T + 1, :])
    p = d + 1
    w = np.zeros((N_T + 1, p))
    disc = math.exp(-beta * Dt)
    for n in range(N_T - 1, -1, -1):
        phi_n = Phi[:, n, :]
        phi_next = Phi[:, n + 1, :]
        G = (phi_n.T @ phi_n) / M
        r = np.sum(S[:, n, :] ** 2, axis=1)
        b = Dt * (phi_n.T @ r) / M
        P = disc * (phi_n.T @ phi_next) / M
        rhs = b + P @ w[n + 1]
        w[n] = np.linalg.solve(G, rhs)
    return w


def solve_alpha_kappa(theta_func, sigma: float, d: int, T: float = 1.0, beta: float = 1.0, dt: float = 1e-4):
    N = int(T / dt)
    t_grid = np.linspace(0.0, T, N + 1)
    alpha = np.zeros(N + 1)
    kappa = np.zeros(N + 1)
    alpha[-1] = 0.0
    kappa[-1] = 0.0

    for idx in range(N, 0, -1):
        t = float(t_grid[idx])
        h = -dt

        def f_alpha(tt: float, aa: float) -> float:
            return beta * aa + 2.0 * float(theta_func(tt)) * aa - 1.0

        def f_kappa(_tt: float, aa: float, kk: float) -> float:
            return beta * kk - d * (sigma**2) * aa

        a0 = float(alpha[idx])
        k0 = float(kappa[idx])

        k1a = f_alpha(t, a0)
        k1k = f_kappa(t, a0, k0)

        k2a = f_alpha(t + h / 2, a0 + h * k1a / 2)
        k2k = f_kappa(t + h / 2, a0 + h * k1a / 2, k0 + h * k1k / 2)

        k3a = f_alpha(t + h / 2, a0 + h * k2a / 2)
        k3k = f_kappa(t + h / 2, a0 + h * k2a / 2, k0 + h * k2k / 2)

        k4a = f_alpha(t + h, a0 + h * k3a)
        k4k = f_kappa(t + h, a0 + h * k3a, k0 + h * k3k)

        alpha[idx - 1] = a0 + (h / 6) * (k1a + 2 * k2a + 2 * k3a + k4a)
        kappa[idx - 1] = k0 + (h / 6) * (k1k + 2 * k2k + 2 * k3k + k4k)

    return t_grid, alpha, kappa


def make_ou10_scaling(out_pdf: str) -> None:
    d = 10
    T = 1.0
    Dt = 0.05
    N_T = int(T / Dt)
    beta = 1.0

    theta0 = 1.0
    theta_amp = 0.6
    omega = 20.0
    theta_func = lambda t: theta0 + theta_amp * math.sin(omega * t)

    sigma = 0.5

    _, alpha_ref, kappa_ref = solve_alpha_kappa(theta_func, sigma, d, T=T, beta=beta, dt=1e-4)
    alpha0 = float(alpha_ref[0])
    kappa0 = float(kappa_ref[0])

    rng = np.random.default_rng(123)
    S_test = rng.normal(size=(20000, d))
    V_true_test = kappa0 + alpha0 * np.sum(S_test**2, axis=1)

    M_list = [200, 400, 800, 1600]
    rep = 3

    err_generator = []
    err_be = []
    for M in M_list:
        e_ph = []
        e_be = []
        for rseed in range(rep):
            S_traj = simulate_ou(d, M, N_T, Dt, theta_func, sigma, seed=1000 + 37 * rseed, extra=2)
            w_generator = run_high_order_generator_from_trajectories(S_traj, Dt, beta, i=2)
            w_be = run_td_be_from_trajectories(S_traj[:, : N_T + 1, :], Dt, beta)
            w0_gen = w_generator[0]
            w0_be = w_be[0]
            V_hat_ph = w0_gen[0] + np.sum(w0_gen[1:] * (S_test**2), axis=1)
            V_hat_be = w0_be[0] + np.sum(w0_be[1:] * (S_test**2), axis=1)
            e_ph.append(math.sqrt(float(np.mean((V_hat_ph - V_true_test) ** 2))))
            e_be.append(math.sqrt(float(np.mean((V_hat_be - V_true_test) ** 2))))
        err_generator.append(float(np.mean(e_ph)))
        err_be.append(float(np.mean(e_be)))

    plt.figure()
    plt.loglog(M_list, err_be, marker="o", label="BE")
    plt.loglog(M_list, err_generator, marker="s", label="Generator (order 2)")
    plt.xlabel(r"number of episodes $M$")
    plt.ylabel(r"$L^2$ value error at $t=0$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


# ------------------------------------------------------------
# Figure: lqr_scaling_M.pdf
# ------------------------------------------------------------


def A_cl_func_factory():
    d = 4
    A0 = np.array(
        [
            [0.2, 1.0, 0.0, 0.0],
            [-1.0, 0.2, 0.3, 0.0],
            [0.0, 0.0, 0.2, 1.0],
            [0.0, 0.0, -1.0, 0.2],
        ]
    )
    A1 = np.array(
        [
            [0.0, 0.5, 0.0, 0.0],
            [-0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5],
            [0.0, 0.0, -0.5, 0.0],
        ]
    )
    amp = 0.5
    omega = 15.0
    K = 0.8 * np.eye(d)

    def A_cl(t: float) -> np.ndarray:
        return (A0 + amp * math.sin(omega * t) * A1) - K

    return A_cl, K


def phi_quadratic(s: np.ndarray) -> np.ndarray:
    d = s.shape[-1]
    feats = [np.ones((*s.shape[:-1], 1)), s**2]
    crosses = []
    for i in range(d):
        for j in range(i + 1, d):
            crosses.append((2.0 * s[..., i] * s[..., j])[..., None])
    if crosses:
        feats.append(np.concatenate(crosses, axis=-1))
    return np.concatenate(feats, axis=-1)


def simulate_linear(S0: np.ndarray, A_cl_func, sigma: float, Dt: float, N: int, extra: int = 2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M, d = S0.shape
    S = np.zeros((M, N + extra + 1, d))
    S[:, 0, :] = S0
    for n in range(N + extra):
        t = n * Dt
        A = A_cl_func(t)
        drift = S[:, n, :] @ A.T
        S[:, n + 1, :] = S[:, n, :] + drift * Dt + sigma * math.sqrt(Dt) * rng.normal(size=(M, d))
    return S


def solve_P_c(A_cl_func, Q_cl: np.ndarray, sigma: float, d: int, T: float = 1.0, beta: float = 1.0, dt: float = 1e-4):
    N = int(T / dt)
    t_grid = np.linspace(0.0, T, N + 1)
    P = np.zeros((N + 1, d, d))
    c = np.zeros(N + 1)
    P[-1] = 0.0
    c[-1] = 0.0
    Sigma = (sigma**2) * np.eye(d)

    for idx in range(N, 0, -1):
        t = float(t_grid[idx])
        h = -dt

        def fP(tt: float, Pmat: np.ndarray) -> np.ndarray:
            A = A_cl_func(tt)
            return beta * Pmat - Pmat @ A - A.T @ Pmat - Q_cl

        def fc(_tt: float, Pmat: np.ndarray, cval: float) -> float:
            return beta * cval - float(np.trace(Sigma @ Pmat))

        P0 = P[idx]
        c0 = float(c[idx])

        k1P = fP(t, P0)
        k1c = fc(t, P0, c0)

        k2P = fP(t + h / 2, P0 + h * k1P / 2)
        k2c = fc(t + h / 2, P0 + h * k1P / 2, c0 + h * k1c / 2)

        k3P = fP(t + h / 2, P0 + h * k2P / 2)
        k3c = fc(t + h / 2, P0 + h * k2P / 2, c0 + h * k2c / 2)

        k4P = fP(t + h, P0 + h * k3P)
        k4c = fc(t + h, P0 + h * k3P, c0 + h * k3c)

        P[idx - 1] = P0 + (h / 6) * (k1P + 2 * k2P + 2 * k3P + k4P)
        c[idx - 1] = c0 + (h / 6) * (k1c + 2 * k2c + 2 * k3c + k4c)

    return t_grid, P, c


def run_high_order_generator_linear(S: np.ndarray, Dt: float, beta: float, i: int, Q_cl: np.ndarray) -> np.ndarray:
    M, Tsteps, d = S.shape
    extra = i
    N_T = Tsteps - 1 - extra
    Phi = phi_quadratic(S)
    p = Phi.shape[-1]
    a = a_coeffs(i)
    w = np.zeros((N_T + 1, p))

    for n in range(N_T - 1, -1, -1):
        phi_n = Phi[:, n, :]
        G = (phi_n.T @ phi_n) / M
        r = np.einsum("bi,ij,bj->b", S[:, n, :], Q_cl, S[:, n, :])
        bvec = (phi_n.T @ r) / M
        delta_phi = np.zeros_like(phi_n)
        for j in range(i + 1):
            delta_phi += a[j] * Phi[:, n + j, :]
        A = (phi_n.T @ (delta_phi / Dt)) / M
        Mmat = (beta + 1.0 / Dt) * G - A
        rhs = bvec + (1.0 / Dt) * (G @ w[n + 1])
        w[n] = np.linalg.solve(Mmat, rhs)

    return w


def run_td_be_linear(S: np.ndarray, Dt: float, beta: float, Q_cl: np.ndarray) -> np.ndarray:
    M, Tsteps, d = S.shape
    N_T = Tsteps - 1
    Phi = phi_quadratic(S)
    p = Phi.shape[-1]
    w = np.zeros((N_T + 1, p))
    disc = math.exp(-beta * Dt)

    for n in range(N_T - 1, -1, -1):
        phi_n = Phi[:, n, :]
        phi_next = Phi[:, n + 1, :]
        G = (phi_n.T @ phi_n) / M
        r = np.einsum("bi,ij,bj->b", S[:, n, :], Q_cl, S[:, n, :])
        bvec = Dt * (phi_n.T @ r) / M
        Pmat = disc * (phi_n.T @ phi_next) / M
        rhs = bvec + Pmat @ w[n + 1]
        w[n] = np.linalg.solve(G, rhs)

    return w


def make_lqr_scaling(out_pdf: str) -> None:
    d = 4
    A_cl_func, K = A_cl_func_factory()
    Q = np.eye(d)
    R = 0.1 * np.eye(d)
    Q_cl = Q + K.T @ R @ K
    sigma = 0.3

    T = 1.0
    Dt = 0.05
    N_T = int(T / Dt)
    beta = 1.0

    _, P_ref, c_ref = solve_P_c(A_cl_func, Q_cl, sigma, d, T=T, beta=beta, dt=1e-4)

    rng = np.random.default_rng(321)
    S_test = rng.normal(size=(20000, d))
    Phi_test = phi_quadratic(S_test)
    V_true_test = c_ref[0] + np.einsum("bi,ij,bj->b", S_test, P_ref[0], S_test)

    M_list = [200, 400, 800, 1600]
    rep = 3

    err_generator = []
    err_be = []
    for M in M_list:
        e_ph = []
        e_be = []
        for rseed in range(rep):
            S0 = rng.normal(size=(M, d))
            S_traj = simulate_linear(S0, A_cl_func, sigma, Dt, N_T, extra=2, seed=100 + 13 * rseed)
            w_ph = run_high_order_generator_linear(S_traj, Dt, beta, i=2, Q_cl=Q_cl)
            w_be = run_td_be_linear(S_traj[:, : N_T + 1, :], Dt, beta, Q_cl=Q_cl)
            V_hat_ph = Phi_test @ w_ph[0]
            V_hat_be = Phi_test @ w_be[0]
            e_ph.append(math.sqrt(float(np.mean((V_hat_ph - V_true_test) ** 2))))
            e_be.append(math.sqrt(float(np.mean((V_hat_be - V_true_test) ** 2))))
        err_generator.append(float(np.mean(e_ph)))
        err_be.append(float(np.mean(e_be)))

    plt.figure()
    plt.loglog(M_list, err_be, marker="o", label="BE")
    plt.loglog(M_list, err_generator, marker="s", label="Generator (order 2)")
    plt.xlabel(r"number of episodes $M$")
    plt.ylabel(r"$L^2$ value error at $t=0$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------


def main() -> None:
    make_convergence_dt("figures/convergence_dt.pdf")
    make_ou10_scaling("figures/ou10_scaling_M.pdf")
    make_lqr_scaling("figures/lqr_scaling_M.pdf")
    print("Saved figures to ./figures/")


if __name__ == "__main__":
    main()
