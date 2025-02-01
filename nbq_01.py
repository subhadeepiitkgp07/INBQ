import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters for the ocean model
nx, ny, nz = 200, 200, 30  # Grid dimensions (x, y, z)
dx, dy, dz = 1000, 1000, 20  # Grid spacing in meters
dt = 60.0  # Time step in seconds
nt = 1000  # Number of time steps

rho0 = 1025.0  # Reference density (kg/m^3)
g = 9.81  # Gravitational acceleration (m/s^2)
alpha = 2.0e-4  # Thermal expansion coefficient (1/°C)
beta = 7.5e-4  # Salinity contraction coefficient (1/psu)
kappa = 1.0e-6  # Diffusivity for tracers (m^2/s)
nu = 1.0e-6  # Kinematic viscosity (m^2/s)
Re = 1000  # Reynolds number for turbulence modeling

# MPI Initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Domain decomposition for MPI (split the grid across processors)
px = 4  # Number of processors in the x-direction
py = size // px  # Number of processors in the y-direction
nx_local = nx // px
ny_local = ny // py
x_start = (rank % px) * nx_local
y_start = (rank // px) * ny_local

# Initialize model fields
u = np.zeros((nx_local, ny_local, nz))  # Velocity components in the x-direction
v = np.zeros((nx_local, ny_local, nz))  # Velocity components in the y-direction
w = np.zeros((nx_local, ny_local, nz))  # Vertical velocity
T = np.full((nx_local, ny_local, nz), 15.0)  # Temperature field (°C)
S = np.full((nx_local, ny_local, nz), 35.0)  # Salinity field (psu)
rho = np.full((nx_local, ny_local, nz), rho0)  # Density field (kg/m^3)
p = np.zeros((nx_local, ny_local, nz))  # Pressure field (Pa)

# Hybrid vertical coordinate (simple z-level model for now)
def get_z_coordinate(depth, max_depth=5000):
    z = np.zeros_like(depth)
    z[depth < 200] = -depth[depth < 200]  # Use sigma coordinates in shallow waters
    z[depth >= 200] = -200.0  # Use z-level coordinates for deeper layers
    return z

# Update density based on temperature and salinity
def update_density(T, S):
    return rho0 * (1.0 - alpha * (T - 10.0) + beta * (S - 35.0))

# Compute pressure gradient force (simplified)
def pressure_gradient(p, dx, dy):
    dpdx = np.gradient(p, axis=0) / dx
    dpdy = np.gradient(p, axis=1) / dy
    return dpdx, dpdy

# Solve momentum equations (Navier-Stokes, including advection, pressure gradient, and buoyancy)
def solve_momentum(u, v, w, p, rho, dt, dx, dy, dz, nu):
    u_new = u.copy()
    v_new = v.copy()
    w_new = w.copy()

    # Advection and pressure gradient (simplified)
    dpdx, dpdy = pressure_gradient(p, dx, dy)
    u_new[1:-1, 1:-1, 1:-1] -= dt * (dpdx[1:-1, 1:-1] / rho[1:-1, 1:-1])
    v_new[1:-1, 1:-1, 1:-1] -= dt * (dpdy[1:-1, 1:-1] / rho[1:-1, 1:-1])
    w_new[1:-1, 1:-1, 1:-1] -= dt * g * (rho[1:-1, 1:-1] - rho0) / rho0  # Buoyancy force

    # Eddy viscosity (turbulence model)
    u_new[1:-1, 1:-1, 1:-1] += nu * dt * np.gradient(u, axis=0) / dx
    v_new[1:-1, 1:-1, 1:-1] += nu * dt * np.gradient(v, axis=1) / dy
    w_new[1:-1, 1:-1, 1:-1] += nu * dt * np.gradient(w, axis=2) / dz

    return u_new, v_new, w_new

# Update tracers (temperature and salinity) via advection and diffusion
def update_tracers(T, S, u, v, w, dt, dx, dy, dz, kappa):
    T_new = T.copy()
    S_new = S.copy()

    # Advection-diffusion for temperature and salinity
    T_new[1:-1, 1:-1, 1:-1] += kappa * dt * (
        (T[2:, 1:-1, 1:-1] - 2 * T[1:-1, 1:-1, 1:-1] + T[:-2, 1:-1, 1:-1]) / dx**2 +
        (T[1:-1, 2:, 1:-1] - 2 * T[1:-1, 1:-1, 1:-1] + T[1:-1, :-2, 1:-1]) / dy**2 +
        (T[1:-1, 1:-1, 2:] - 2 * T[1:-1, 1:-1, 1:-1] + T[1:-1, 1:-1, :-2]) / dz**2
    )
    S_new[1:-1, 1:-1, 1:-1] += kappa * dt * (
        (S[2:, 1:-1, 1:-1] - 2 * S[1:-1, 1:-1, 1:-1] + S[:-2, 1:-1, 1:-1]) / dx**2 +
        (S[1:-1, 2:, 1:-1] - 2 * S[1:-1, 1:-1, 1:-1] + S[1:-1, :-2, 1:-1]) / dy**2 +
        (S[1:-1, 1:-1, 2:] - 2 * S[1:-1, 1:-1, 1:-1] + S[1:-1, 1:-1, :-2]) / dz**2
    )

    return T_new, S_new

# Visualization of the temperature field
fig, ax = plt.subplots()
im = ax.imshow(T[:, :, nz // 2], cmap='coolwarm', origin='lower')
ax.set_title("Temperature Field")
plt.colorbar(im)

# Update plot function for animation
def update_plot(frame):
    global T, S, u, v, w, rho
    T, S = update_tracers(T, S, u, v, w, dt, dx, dy, dz, kappa)
    rho = update_density(T, S)
    u, v, w = solve_momentum(u, v, w, p, rho, dt, dx, dy, dz, nu)
    im.set_array(T[:, :, nz // 2])  # Update with temperature at mid-depth
    ax.set_title(f"Temperature Field (Step {frame})")
    return [im]

# Animation of the temperature field over time
ani = FuncAnimation(fig, update_plot, frames=nt, interval=200, blit=True)
plt.show()
