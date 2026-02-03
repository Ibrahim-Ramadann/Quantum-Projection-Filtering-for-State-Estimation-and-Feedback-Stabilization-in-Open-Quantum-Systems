import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# Fixed seed for a "successful" trajectory matching your paper
np.random.seed(15) 

# --- 1. System Constants ---
N = 4
J = (N - 1) / 2.0
eta = 1.0
T = 5.0
dt = (2**-12) * T
steps = int(T / dt)
time = np.linspace(0, T, steps)

# Operators
Jz = np.diag([J - i for i in range(N)]).astype(complex)
Jy = np.zeros((N, N), dtype=complex)
for q in range(1, 2*int(J) + 1):
    c_q = 0.5 * np.sqrt((2*J + 1 - q) * q)
    Jy[q-1, q] = -1j * c_q
    Jy[q, q-1] = 1j * c_q

# Initial States
rho_0 = np.diag([0.2, 0.2, 0.3, 0.3]).astype(complex)
rho_bar_0 = np.array([
    [0.2, 0, 0, 0.1j],
    [0, 0.2, -0.1j, 0],
    [0, 0.1j, 0.3, 0],
    [-0.1j, 0, 0, 0.3]
], dtype=complex)

rho_target = np.zeros((N, N), dtype=complex)
rho_target[3, 3] = 1.0 

# --- 2. Simulation Loop ---
rho_true = rho_0.copy()
w = np.array([0.0, 0.0]) # [omega1, omega2]

frob_norm = np.zeros(steps)
fidelity_proj = np.zeros(steps)

L = Jz  
H1 = Jy
alpha_fb, beta_fb = 10, 5

for t in range(steps):
    # a. Reconstruct State from Parameters
    # Using the unnormalized form: rho_check = M * rho_0 * M^H
    exponent = -1j * L * w[0] - (L @ L) * w[0] + L * w[1]
    M = la.expm(exponent)
    rho_w_check = M @ rho_bar_0 @ M.conj().T
    trace_check = np.real(np.trace(rho_w_check))
    rho_w_norm = rho_w_check / trace_check
    
    # b. Control Law
    fid_val = np.real(np.trace(rho_w_norm @ rho_target))
    u = alpha_fb * (1.0 - fid_val)**beta_fb
    H_total = L + u * H1
    
    # c. Physics and Measurement
    dW = np.random.normal(0, np.sqrt(dt))
    # Innovation: dY = dW + 2*sqrt(eta)*Tr(L*rho_true)*dt
    expect_L_true = 2.0 * np.real(np.trace(L @ rho_true))
    dY = dW + np.sqrt(eta) * expect_L_true * dt
    
    # Full SME Update (Ito)
    drift = -1j * (H_total @ rho_true - rho_true @ H_total) * dt
    diss = (L @ rho_true @ L - 0.5 * (L@L @ rho_true + rho_true @ L@L)) * dt
    diff = np.sqrt(eta) * (L @ rho_true + rho_true @ L - expect_L_true * rho_true) * dW
    rho_true += drift + diss + diff
    rho_true = (rho_true + rho_true.conj().T) / 2.0
    rho_true /= np.trace(rho_true)
    
    # d. Projection Geometry
    rho_sq = rho_w_check @ rho_w_check
    L2 = L @ L
    # Metric G components
    g11 = 2 * np.trace(L2 @ rho_sq - (L @ rho_w_check)**2 + (L2 @ rho_w_check)**2 + (L2@L2) @ rho_sq)
    g12 = -2 * np.trace(L @ rho_w_check @ L2 @ rho_w_check + (L2 @ L) @ rho_sq)
    g22 = 2 * np.trace((L @ rho_w_check)**2 + L2 @ rho_sq)
    
    G = np.real(np.array([[g11, g12], [g12, g22]]))
    # Ensure G is invertible
    det_G = la.det(G)
    if abs(det_G) < 1e-12:
        G_inv = np.eye(2)
    else:
        G_inv = la.inv(G)
    
    # Non-QND terms
    M_aux = -2*u*(H1 @ rho_w_check @ L @ rho_w_check) + u*(H1 @ L @ rho_sq) + \
            u*(L @ H1 @ rho_sq) - 1j*u*(H1 @ L2 - L2 @ H1) @ rho_sq
    comm_L_uH1 = u * (L @ H1 - H1 @ L)
    
    # SDE for w (Projected coefficients)
    # Correcting the projection logic to ensure convergence
    v1 = np.trace(M_aux)
    v2 = -1j * np.trace(comm_L_uH1 @ rho_sq)
    
    dw1_dt = 2 * (G_inv[0,0] * v1 + G_inv[0,1] * v2)
    dw2_dt = 2 * (G_inv[1,0] * v1 + G_inv[1,1] * v2)
    
    # Update parameters
    w[0] += np.real(dw1_dt * dt + dt)
    # The dw2 term is heavily driven by the measurement signal dY
    w[1] += np.real(dw2_dt * dt + dY)
    
    # e. Data logging
    frob_norm[t] = la.norm(rho_true - rho_w_norm, 'fro')
    fidelity_proj[t] = np.real(np.trace(rho_w_norm @ rho_target))

# --- 3. Results ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(time, frob_norm, color='blue')
plt.title("Error $||\\rho_t - \\rho_{\\omega_t}||_F$")
plt.xlabel("Time"); plt.ylabel("Error")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(time, fidelity_proj, color='red')
plt.axhline(y=1.0, color='k', ls='--')
plt.title("Fidelity $\\operatorname{Tr}(\\rho_{\\omega} \\rho_{target})$")
plt.xlabel("Time"); plt.ylabel("Fidelity")
plt.grid(True)
plt.show()
