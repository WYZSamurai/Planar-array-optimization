import torch
import pattern


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# device = torch.device("cpu")


lamb = 1.0
d = 0.5*lamb

phi0 = 0.0
theta0 = 0.0
dp = 180
dt = 180

NP = 100
m = 20
n = 10

mag = torch.randint(0, 2, (NP, m, n))
# mag = torch.ones(NP, m, n)
phase0 = torch.zeros_like(mag)
Fdb = torch.zeros(NP, dt, dp)
print("循环开始。")
for i in range(NP):
    Fdb[i] = pattern.pattern(
        mag[i], phase0[i], lamb, d, theta0, phi0, dt, dp)
print("循环完成。")
# pattern.plot(Fdb, dt, dp)
