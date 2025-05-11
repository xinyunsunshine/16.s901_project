import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def gray_scott_imex(u, v, Du, Dv, F, k, dt, n_steps, L):
    """Performs IMEX Euler steps on given u, v fields."""
    N = u.shape[0]
    dx = L / N
    kx = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    ky = kx.copy()
    k2 = kx[:, None]**2 + ky[None, :]**2
    denom_u = 1 + dt * Du * k2
    denom_v = 1 + dt * Dv * k2
    
    for _ in range(n_steps):
        uv2 = u * v * v
        R_u = -uv2 + F * (1 - u)
        R_v = uv2 - (F + k) * v
        u_star = u + dt * R_u
        v_star = v + dt * R_v
        
        u_hat = np.fft.fft2(u_star)
        v_hat = np.fft.fft2(v_star)
        u = np.real(np.fft.ifft2(u_hat / denom_u))
        v = np.real(np.fft.ifft2(v_hat / denom_v))
    return u, v

#############################
### Temporal convergence #####
#############################

# Parameters
N = 64
L = 2.5
Du, Dv = 2e-5, 1e-5
F, k = 0.04, 0.060
T_end = 10.24
dt_ref = 0.04
n_steps_ref = int(T_end / dt_ref)

# Initial condition
def init_uv(N):
    u = np.ones((N, N))
    v = np.zeros((N, N))
    r = N // 10
    cx = cy = N // 2
    u[cx-r:cx+r, cy-r:cy+r] = 0.50
    v[cx-r:cx+r, cy-r:cy+r] = 0.25
    return u, v

# Reference solution
u0_ref, v0_ref = init_uv(N)
u_ref, v_ref = gray_scott_imex(u0_ref, v0_ref, Du, Dv, F, k, dt_ref, n_steps_ref, L)

# Test various dt values
dt_list = [0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12]
results = []

for dt in dt_list:
    n_steps = int(T_end / dt)
    u0, v0 = init_uv(N)
    try:
        u_dt, _ = gray_scott_imex(u0, v0, Du, Dv, F, k, dt, n_steps, L)
        if np.any(np.isnan(u_dt)) or not np.all(np.isfinite(u_dt)):
            stable = False
            error = None
        else:
            stable = True
            # Compute L2 error norm
            diff = u_dt - u_ref
            error = np.linalg.norm(diff) / np.sqrt(N*N)
    except Exception:
        stable = False
        error = None
    results.append({"dt": dt, "stable": stable, "L2_error": error})

# Display results
df = pd.DataFrame(results)

# Plot convergence for stable dt values with error
stable_errors = df[df["stable"] & df["L2_error"].notnull()]
plt.figure()
plt.loglog(stable_errors["dt"], stable_errors["L2_error"], marker='o')
plt.xlabel('Time step Δt (log)')
plt.ylabel('L2 error norm (log)')
plt.title('Temporal Convergence of IMEX Euler (Gray–Scott)')
plt.savefig('gray_scott_convergence.png', dpi=300)
plt.show()
plt.close()

# Identify maximum stable dt
max_stable_dt = df[df["stable"]]["dt"].max()
print(f"Maximum stable time-step: {max_stable_dt}")

print(df)

#############################
### Spacial convergence #####
#############################
# Parameters
L = 2.5
Du, Dv = 2e-5, 1e-5
F, k = 0.04, 0.060
T_end = 1.0
dt = 0.005
n_steps = int(T_end / dt)

# Grid sizes to test (divisors of 128)
grid_sizes = [16, 32, 64, 128, 256]
results = []

# Reference solution on finest grid
N_ref = grid_sizes[-1]
u0_ref = np.ones((N_ref, N_ref)); v0_ref = np.zeros((N_ref, N_ref))
r = N_ref // 10; cx = cy = N_ref // 2
u0_ref[cx-r:cx+r, cy-r:cy+r] = 0.50
v0_ref[cx-r:cx+r, cy-r:cy+r] = 0.25
u_ref, v_ref = gray_scott_imex(u0_ref.copy(), v0_ref.copy(), Du, Dv, F, k, dt, n_steps, L)

# Compute errors for coarser grids
for N in grid_sizes:
    # Initialize
    u0 = np.ones((N, N)); v0 = np.zeros((N, N))
    r = N // 10; cx = cy = N // 2
    u0[cx-r:cx+r, cy-r:cy+r] = 0.50
    v0[cx-r:cx+r, cy-r:cy+r] = 0.25
    
    u_N, _ = gray_scott_imex(u0.copy(), v0.copy(), Du, Dv, F, k, dt, n_steps, L)
    
    # Sample reference at coarse grid points
    step = N_ref // N
    u_ref_coarse = u_ref[::step, ::step]
    
    # Compute L2 error on coarse grid
    diff = u_N - u_ref_coarse
    L2_error = np.linalg.norm(diff) / np.sqrt(N*N)
    dx = L / N
    
    results.append({"N": N, "dx": dx, "L2_error": L2_error})

# Display results table
df = pd.DataFrame(results)

# # Plot error vs dx
plt.figure()
plt.semilogx(df["N"].iloc[:4], df["L2_error"].iloc[:4], marker='o', linestyle='-')
# plt.loglog(df["dx"], df["L2_error"], marker='o', linestyle='-')
# plt.xlabel('Grid spacing Δx')

plt.ylabel('L2 error norm')
plt.xlabel('Number of grid points N (log)')
plt.title('Spatial Convergence of IMEX Euler (Gray–Scott log-x scale)')
plt.grid(True, which='both', ls='--', lw=0.5)
plt.savefig('gray_scott_spatial_convergence.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()