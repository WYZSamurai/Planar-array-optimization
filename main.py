import torch


lamb = 1.0
d = 0.5*lamb
m = 20
n = 10
phi0 = 0
theta0 = 0
dp = 360
dt = 360


pi = torch.pi
phi = torch.linspace(-pi/2, pi/2, dp)
theta = torch.linspace(-pi/2, pi/2, dt)
k = 2*pi/lamb


F1 = torch.zeros(dt, dp)
F2 = torch.zeros(dt, dp)
mag = torch.randint(0, 2, (m, n))

for i in range(dt):
    for j in range(dp):
        ang = k*d*torch.arange(0, m)*(torch.sin(theta[i])*torch.sin(
            phi[j])-torch.sin(theta0)*torch.sin(phi0))
        # F1[i,j]=torch.abs(torch.sum())


if __name__ == "__main__":
    pass
