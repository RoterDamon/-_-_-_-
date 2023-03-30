import numpy as np
import matplotlib.pyplot as plt

# problem parameters
V0 = 1
c = 2
D = 0.1
m = 1
T = 10
L = 1

# FD parameters
N = 50
M = 100

# create mesh
t = np.linspace(0, T, N)
x = np.linspace(0, L, M)
tau = t[1] - [0]
h = x[1] - x[0]
u = np.zeros([N, M])
v = np.zeros([N, M])

# initial conditions
phi = np.sin(np.pi * x / L)
psi = 0.0 * x
Phi = -2 * V0 * (L / np.pi) * (np.cos(np.pi * x / L) - 1)
u[0, :] = phi
v[0, :] = Phi - x / L * Phi - V0 * x * phi

# get vector W for t=0
W = []
for ui, vi in zip(u[0], v[0]):
    W.append(ui)
    W.append(vi)
W = np.array(W[2:-2])

# iterate by time
for n in range(len(t)):
    # assemble the FD matrix for the finite difference scheme:
    # W(n+1) = FD * W(n)
    FD = np.zeros([2 * (M - 2), 2 * (M - 2)])
    for i in range(0, 2 * (M - 2), 2):
        alpha = V0 * x[1 + i // 2] + (c**2 - V0**2) * t[n]
        A = np.array([[alpha, 1], [-D / m - alpha**2, -alpha]])
        if not i == 0:
            FD[i : i + 2, i - 2 : i] = tau * A / h**2
        FD[i : i + 2, i : i + 2] = -2 * tau * A / h**2 - np.identity(2)
        if not i == 2 * (M - 3):
            FD[i : i + 2, i + 2 : i + 4] = tau * A / h**2

    # solve linear system
    W = np.linalg.solve(FD, -W)

    # save result in variables u,v
    if n < len(t) - 1:
        u[n + 1, 1:-1] = [W[i] for i in range(0, 2 * (M - 2), 2)]
        v[n + 1, 1:-1] = [W[i + 1] for i in range(0, 2 * (M - 2), 2)]

# plot results in 3d
ax = plt.axes(projection="3d")
T, X = np.meshgrid(t, x)
ax.plot_surface(T, X, u.T, cmap="viridis", antialiased=True)
plt.xlabel("Time")
plt.ylabel("Space")
plt.show()

# plot results in 2d
for ui in u:
    plt.plot(x, ui)
plt.xlabel("Space")
plt.ylabel("u(t,x)")
plt.show()
