import torch
import plotly.graph_objects as go


# 多个个体的方向图
def pattern_multiple(mag: torch.Tensor, phase0: torch.Tensor, lamb: float, d: float, theta0: float, phi0: float, dt: int, dp: int):
    pi = torch.pi
    _, m, n = mag.shape
    k = 2*pi/lamb
    theta0 = torch.tensor(theta0) * pi / 180
    phi0 = torch.tensor(phi0) * pi / 180

    phi = torch.linspace(-pi/2, pi/2, dp, device=mag.device)
    theta = torch.linspace(-pi/2, pi/2, dt, device=mag.device)

    # 构造角度矩阵 (dt, dp)
    ang1 = torch.cos(theta.view(-1, 1)) * torch.sin(phi.view(1, -1)
                                                    ) - torch.cos(theta0) * torch.sin(phi0)
    ang2 = torch.sin(theta.view(-1, 1)) - torch.sin(theta0)

    # 构造距离矩阵 (m, n)
    dm = k * d * torch.arange(m, device=mag.device).view(m, 1)
    dn = k * d * torch.arange(n, device=mag.device).view(1, n)

    # 计算每个天线元的相位贡献(NP,m,n,dt,dp)
    phase_contributions = phase0.unsqueeze(3).unsqueeze(4) + dm.unsqueeze(2).unsqueeze(3).unsqueeze(0) * ang1.unsqueeze(
        0).unsqueeze(0).unsqueeze(0) + dn.unsqueeze(2).unsqueeze(3).unsqueeze(0) * ang2.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    # 计算复数指数项并按天线元求和(NP,dt,dp)
    F = torch.abs(torch.sum(mag.unsqueeze(3).unsqueeze(4) * torch.exp(torch.complex(
        torch.zeros_like(phase_contributions), phase_contributions)), dim=(1, 2)))

    # 转换为分贝并进行归一化
    Fdb = 20 * torch.log10(F / torch.max(F))
    return Fdb


# 单个个体的方向图
def pattern(mag: torch.Tensor, phase0: torch.Tensor, lamb: float, d: float, theta0: float, phi0: float, dt: int, dp: int):
    pi = torch.pi
    k = 2 * pi / lamb

    theta0_rad = torch.tensor(theta0, device=mag.device) * pi / 180
    phi0_rad = torch.tensor(phi0, device=mag.device) * pi / 180

    phi = torch.linspace(-pi / 2, pi / 2, dp, device=mag.device)
    theta = torch.linspace(-pi / 2, pi / 2, dt, device=mag.device)

    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing="ij")

    ang1 = torch.cos(theta_grid) * torch.sin(phi_grid) - \
        torch.cos(theta0_rad) * torch.sin(phi0_rad)
    ang2 = torch.sin(theta_grid) - torch.sin(theta0_rad)

    m, n = mag.shape
    dm = k * d * torch.arange(m, device=mag.device).reshape(m,
                                                            1, 1, 1) * ang1.reshape(1, dt, dp)
    dn = k * d * torch.arange(n, device=mag.device).reshape(1,
                                                            n, 1, 1) * ang2.reshape(1, dt, dp)

    phase_contributions = phase0.unsqueeze(-1).unsqueeze(-1) + dm + dn
    complex_exponentials = torch.exp(torch.complex(
        torch.zeros_like(phase_contributions), phase_contributions))

    F = (mag.unsqueeze(-1).unsqueeze(-1) *
         complex_exponentials).sum(dim=(0, 1)).abs()
    Fdb = 20 * torch.log10(F / F.max())

    return Fdb


def plot(Fdb: torch.Tensor, dt: int, dp: int):
    phi = torch.linspace(-90.0, 90.0, dp)
    theta = torch.linspace(-90.0, 90.0, dt)

    torch.meshgrid(theta, phi, indexing='ij')
    fig = go.Figure(data=[go.Surface(z=Fdb.cpu(), x=theta, y=phi)])

    # 更新图表布局
    fig.update_layout(
        title="3D方向图",
        scene=dict(
            xaxis_title='theta',
            yaxis_title='phi',
            zaxis_title='Fdb'
        ),
        autosize=True,
    )

    # 显示图表
    fig.show()
