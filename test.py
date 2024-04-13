import numpy as np
import matplotlib.pyplot as plt


def generate_array_factors(M, N, wavelength, d, theta0, phi0, theta_range, phi_range):
    # 设定阵元位置
    x = np.linspace(-M/2, M/2, M) * d
    y = np.linspace(-N/2, N/2, N) * d
    X, Y = np.meshgrid(x, y)

    # 随机生成激励幅度和相位
    A = np.random.rand(M, N)
    phi = np.random.rand(M, N) * 2 * np.pi

    # 波数 k
    k = 2 * np.pi / wavelength

    # 方向图计算
    AF = np.zeros((len(theta_range), len(phi_range)), dtype=complex)

    for i, theta in enumerate(theta_range):
        for j, phi in enumerate(phi_range):
            phase_shift = k * (X * np.sin(theta) *
                               np.cos(phi) + Y * np.sin(theta) * np.sin(phi))
            AF[i, j] = np.sum(A * np.exp(1j * (phi - phase_shift)))

    return np.abs(AF)


# 参数设定
M = 10
N = 10
wavelength = 0.3  # 波长为0.3米
d = 0.5  # 阵元间距为0.5米

# 波束指向
theta0 = np.pi / 4  # 45度
phi0 = np.pi / 2    # 90度

# 角度范围
theta_range = np.linspace(0, np.pi, 180)
phi_range = np.linspace(0, 2*np.pi, 360)

# 计算方向图
AF = generate_array_factors(
    M, N, wavelength, d, theta0, phi0, theta_range, phi_range)

# 可视化方向图
theta, phi = np.meshgrid(np.degrees(theta_range), np.degrees(phi_range))
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
contour = ax.contourf(phi, theta, AF.T, levels=50, cmap='viridis')
fig.colorbar(contour, ax=ax, orientation='vertical')
plt.title('Directional Pattern of the Array')
plt.show()
