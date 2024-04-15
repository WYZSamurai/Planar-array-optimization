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
dp = 360
dt = 360

NP = 100
m = 20
n = 10


# 生成dna
dna = torch.randint(0, 2, (NP, m, n), device=device)
# dna = torch.ones(NP, m, n, device=device)
phase0 = torch.zeros_like(dna, device=dna.device)


# 计算群体Fdb
Fdb = torch.zeros(NP, dt, dp)
for i in range(NP):
    Fdb[i] = pattern.pattern(dna[i], phase0[i], lamb, d, theta0, phi0, dt, dp)
# # 使用矩阵操作计算Fdb
# Fdb = pattern.pattern_multiple(dna, phase0, lamb, d, theta0, phi0, dt, dp)


pattern.plot(Fdb[0], dt, dp)
