import torch
import pattern


if __name__ == "__main__":
    lamb = 1.0
    d = 0.5*lamb

    phi0 = 0.0
    theta0 = 0.0
    dp = 360
    dt = 360

    m = 20
    n = 10
    # mag = torch.randint(0, 2, (m, n))
    mag = torch.ones(m, n)
    phase0 = torch.zeros(m, n)

    Fdb = pattern.pattern(mag, phase0, lamb, d, theta0, phi0, dt, dp)
    pattern.plot(Fdb, dt, dp)
