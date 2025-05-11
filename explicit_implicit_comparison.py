import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from math import log2

# ----------------------------------------------------------
# Solver implementations
# ----------------------------------------------------------
def explicit_fd_gray_scott(u, v, Du, Dv, F, k, dt, n_steps, L):
    N, dx = u.shape[0], L/u.shape[0]
    for _ in range(n_steps):
        lap_u = (np.roll(u,1,0)+np.roll(u,-1,0)+np.roll(u,1,1)+np.roll(u,-1,1)-4*u)/dx**2
        lap_v = (np.roll(v,1,0)+np.roll(v,-1,0)+np.roll(v,1,1)+np.roll(v,-1,1)-4*v)/dx**2
        uv2 = u*v*v
        R_u = -uv2 + F*(1-u)
        R_v = uv2 - (F+k)*v
        u = u + dt*(Du*lap_u + R_u)
        v = v + dt*(Dv*lap_v + R_v)
        if not np.all(np.isfinite(u)):
            break
    return u, v

def imex_euler_gray_scott(u, v, Du, Dv, F, k, dt, n_steps, L):
    N, dx = u.shape[0], L/u.shape[0]
    kx = 2*np.pi*np.fft.fftfreq(N, d=dx)
    ky = kx.copy()
    k2 = kx[:,None]**2 + ky[None,:]**2
    denom_u = 1 + dt*Du*k2
    denom_v = 1 + dt*Dv*k2
    for _ in range(n_steps):
        uv2 = u*v*v
        R_u = -uv2 + F*(1-u)
        R_v = uv2 - (F+k)*v
        u_star = u + dt*R_u
        v_star = v + dt*R_v
        u_hat = np.fft.fft2(u_star)
        v_hat = np.fft.fft2(v_star)
        u = np.real(np.fft.ifft2(u_hat/denom_u))
        v = np.real(np.fft.ifft2(v_hat/denom_v))
    return u, v

def newton_implicit_gray_scott(u, v, Du, Dv, F, k, dt, n_steps, L, iters=3):
    N, dx = u.shape[0], L/u.shape[0]
    kx = 2*np.pi*np.fft.fftfreq(N, d=dx)
    ky = kx.copy()
    k2 = kx[:,None]**2 + ky[None,:]**2
    denom_u = 1 + dt*Du*k2
    denom_v = 1 + dt*Dv*k2
    for _ in range(n_steps):
        u_new, v_new = u.copy(), v.copy()
        for _ in range(iters):
            denom_react = 1 + dt*(v_new**2 + F)
            u_new = (u + dt*F) / denom_react
            a = dt*u_new
            b = -(1+dt*(F+k)); c = v
            disc = b**2 - 4*a*c
            v_new = np.where(a!=0, (-b - np.sqrt(np.maximum(disc,0)))/(2*a),
                             v/(1+dt*(F+k)))
            # implicit diffusion
            u_hat = np.fft.fft2(u_new); v_hat = np.fft.fft2(v_new)
            u_new = np.real(np.fft.ifft2(u_hat/denom_u))
            v_new = np.real(np.fft.ifft2(v_hat/denom_v))
        u, v = u_new, v_new
    return u, v

# ----------------------------------------------------------
# Experiment parameters
# ----------------------------------------------------------
Du, Dv, F, k = 2e-5, 1e-5, 0.04, 0.06
L, N = 2.5, 64
dt_ref = 0.00125
T_end = 1.0

# Initial condition
def init_uv(N):
    u = np.ones((N,N)); v = np.zeros((N,N))
    r = N//10; cx = cy = N//2
    u[cx-r:cx+r, cy-r:cy+r] = 0.5
    v[cx-r:cx+r, cy-r:cy+r] = 0.25
    return u, v

# ----------------------------------------------------------
# 1) Temporal L2 Error study
# ----------------------------------------------------------
dt_error_list = [0.005, 0.01, 0.02, 0.05]
schemes = {
    'Explicit FD': explicit_fd_gray_scott,
    'IMEX Euler': imex_euler_gray_scott,
    'Newton Implicit': newton_implicit_gray_scott
}

# compute per-scheme reference solution
refs = {}
n_ref = int(T_end/dt_ref)
for name, solver in schemes.items():
    u0, v0 = init_uv(N)
    refs[name] = solver(u0.copy(), v0.copy(), Du, Dv, F, k, dt_ref, n_ref, L)

# collect errors
err_rows = []
for name, solver in schemes.items():
    u_ref, _ = refs[name]
    for dt in dt_error_list:
        u0, v0 = init_uv(N)
        n_steps = int(T_end/dt)
        u_dt, _ = solver(u0.copy(), v0.copy(), Du, Dv, F, k, dt, n_steps, L)
        if np.all(np.isfinite(u_dt)):
            err = np.linalg.norm(u_dt - u_ref)/np.sqrt(N*N)
        else:
            err = np.nan
        err_rows.append({'Scheme': name, 'Δt': dt, 'L2_Error': err})

df_error = pd.DataFrame(err_rows)


# plot errors
plt.figure(figsize=(6,4))
for name in schemes:
    subset = df_error[df_error['Scheme']==name]
    plt.plot(subset['Δt'], subset['L2_Error'], marker='o', label=name)
plt.xlabel('Δt'); plt.ylabel('L2 Error'); plt.title('L2 Error vs Δt for Different Methods'); plt.legend(); plt.grid(True)
plt.savefig('gray_scott_temporal_error_comparison.png', dpi=300, bbox_inches='tight')
                                                                                           
plt.show()
plt.close()

# ----------------------------------------------------------
# 2) Runtime comparison
# ----------------------------------------------------------

# ----------------------------------------------------------
# Flop count estimation
# ----------------------------------------------------------
def estimate_flops(N, n_steps, newton_iters=3):
    # Explicit FD per step flops ~24*N^2
    flops_explicit_per_step = 24 * N * N
    flops_explicit = flops_explicit_per_step * n_steps
    # IMEX: reaction flops ~16*N^2, FFT flops ~40*N^2*log2(N) per step
    flops_imex_per_step = 16 * N * N + 40 * N * N * log2(N)
    flops_imex = flops_imex_per_step * n_steps
    # Newton implicit: approx newton_iters * imex flops
    flops_newton = flops_imex * newton_iters
    return flops_explicit, flops_imex, flops_newton

# ----------------------------------------------------------
# Benchmark parameters
# ----------------------------------------------------------
Du, Dv = 2e-5, 1e-5
F, k = 0.04, 0.06
L = 2.5
N = 64
dt = 0.01
n_steps = int(1.0 / dt)
newton_iters = 3

# Initial condition
def init_uv(N):
    u = np.ones((N,N))
    v = np.zeros((N,N))
    r = N//10; cx = cy = N//2
    u[cx-r:cx+r, cy-r:cy+r] = 0.5
    v[cx-r:cx+r, cy-r:cy+r] = 0.25
    return u, v

# Run benchmarks
solvers = {
    'Explicit FD': explicit_fd_gray_scott,
    'IMEX Euler': imex_euler_gray_scott,
    'Newton Implicit': newton_implicit_gray_scott
}
results = []
flops_explicit, flops_imex, flops_newton = estimate_flops(N, n_steps, newton_iters)

for name, solver in solvers.items():
    u0, v0 = init_uv(N)
    start = time.perf_counter()
    if name == 'Newton Implicit':
        solver(u0.copy(), v0.copy(), Du, Dv, F, k, dt, n_steps, L, newton_iters)
    else:
        solver(u0.copy(), v0.copy(), Du, Dv, F, k, dt, n_steps, L)
    elapsed = time.perf_counter() - start
    flop_count = {
        'Explicit FD': flops_explicit,
        'IMEX Euler': flops_imex,
        'Newton Implicit': flops_newton
    }[name]
    results.append({
        'Scheme': name,
        'Total CPU Time (s)': elapsed,
        'Time per step (ms)': elapsed/n_steps*1000,
        'Estimated FLOPs': flop_count,
        'FLOPs per step': flop_count/n_steps
    })

df = pd.DataFrame(results)


# Bar chart for CPU time per step
plt.figure(figsize=(6,3))
plt.bar(df['Scheme'], df['Time per step (ms)'], color=['skyblue','orange','green'])
plt.ylabel('Time per step (ms)')
plt.title('Per-step Cost Comparison')
plt.tight_layout()
plt.savefig('gray_scott_per_step_cost_.png', dpi=300, bbox_inches='tight')
plt.show()
# close the plot
plt.close()
print(df)

