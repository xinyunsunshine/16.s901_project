import numpy as np
import matplotlib.pyplot as plt

def gray_scott_imex(N=256, L=2.5, Du=2e-5, Dv=1e-5,
                    F=0.04, k=0.060,
                    dt=1.0, n_steps=5000,
                    init_noise=0.02):
    """
    Simulate the Gray–Scott model with an IMEX Euler scheme...
    (same as before)
    """
    dx = L / N
    k_vals = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    kx, ky = np.meshgrid(k_vals, k_vals)
    lap = -(kx**2 + ky**2)

    denom_u = 1 - dt * Du * lap
    denom_v = 1 - dt * Dv * lap

    u = np.ones((N, N))
    v = np.zeros((N, N))
    r = int(N/10)
    center = N//2
    u[center-r:center+r, center-r:center+r] = 0.50 + init_noise * np.random.randn(2*r, 2*r)
    v[center-r:center+r, center-r:center+r] = 0.25 + init_noise * np.random.randn(2*r, 2*r)

    for step in range(n_steps):
        uvv = u * v * v
        Ru = -uvv + F * (1 - u)
        Rv =  uvv - (F + k) * v
        u_star = u + dt * Ru
        v_star = v + dt * Rv

        u_hat = np.fft.fft2(u_star)
        v_hat = np.fft.fft2(v_star)
        u = np.real(np.fft.ifft2(u_hat / denom_u))
        v = np.real(np.fft.ifft2(v_hat / denom_v))

    return u, v

if __name__ == "__main__":
    param_list = [
        (0.02, 0.050),  # spot patterns
        (0.035, 0.065), # stripe‐like
        (0.040, 0.060), # worm‐like
        (0.060, 0.062)  # mixed
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for ax, (F, k) in zip(axes.flat, param_list):
        u, v = gray_scott_imex(N=256, L=2.5, Du=2e-5, Dv=1e-5,
                               F=F, k=k,
                               dt=1.0, n_steps=8000)
        im = ax.imshow(v, cmap='viridis', origin='lower')
        ax.set_title(f"F = {F:.3f}, k = {k:.3f}")
        ax.axis('off')

        # individual colorbar for this subplot
        cbar = fig.colorbar(im, ax=ax,
                            orientation='vertical',
                            fraction=0.046, pad=0.04)
        cbar.set_label('v concentration')

    fig.suptitle("Gray–Scott Patterns (v field) under Different (F, k)")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    # high-resolution save
    fig.savefig("gray_scott_patterns_individual_cbs.png",
                dpi=300, bbox_inches='tight')
    plt.show()
