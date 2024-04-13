import torch
import plotly.graph_objects as go


def pattern(mag: torch.Tensor, phase0: torch.Tensor, lamb: float, d: float, theta0: float, phi0: float, dt: int, dp: int):
    pi = torch.pi
    (m, n) = mag.shape
    k = 2*pi/lamb
    theta0 = torch.tensor(theta0)*pi/180
    phi0 = torch.tensor(phi0)*pi/180

    phi = torch.linspace(-pi/2, pi/2, dp).reshape(1, dp).repeat(dt, 1)
    theta = torch.linspace(-pi/2, pi/2, dt).reshape(dt, 1).repeat(1, dp)

    # a(dt,dp)
    ang1 = torch.cos(theta)*torch.sin(phi)-torch.cos(theta0)*torch.sin(phi0)
    ang2 = torch.sin(theta)-torch.sin(theta0)

    # d(m,n)
    dm = k*d*torch.arange(0, m).reshape(m, 1).repeat(1, n)
    dn = k*d*torch.arange(0, n).reshape(1, n).repeat(m, 1)

    # phase = phase0+dm*ang1+dn*ang2
    # print(phase.shape)

    F = torch.zeros(dt, dp)
    for i in range(m):
        for j in range(n):
            # phase(dt,dp)
            phase = phase0[i, j]+dm[i, j]*ang1+dn[i, j]*ang2
            F = F+mag[i, j] * \
                torch.exp(torch.complex(torch.zeros_like(phase), phase))
    F = F.abs()
    Fdb = 20*torch.log10(F/F.max())
    return Fdb


def plot(Fdb: torch.Tensor, dt: int, dp: int):
    pi = torch.pi
    phi = torch.linspace(-pi/2, pi/2, dp)
    theta = torch.linspace(-pi/2, pi/2, dt)

    torch.meshgrid(theta, phi, indexing='ij')
    fig = go.Figure(data=[go.Surface(z=Fdb, x=theta, y=phi)])

    # 更新图表布局
    fig.update_layout(
        title="3D 表面图示例",
        scene=dict(
            xaxis_title='theta',
            yaxis_title='phi',
            zaxis_title='Fdb'
        ),
        autosize=True,
    )

    # 显示图表
    fig.show()


if __name__ == "__main__":
    pass
