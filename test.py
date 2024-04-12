import torch
import plotly.graph_objects as go


def plot_array_pattern_with_plotly(m, n, d, wavelength=1.0, beam_direction=0):
    # 转换波束指向角度为弧度
    beam_direction_rad = beam_direction * torch.pi / 180

    # 计算角度范围，这里将波束指向考虑在内
    theta_range = torch.linspace(-90, 90, 100) * \
        torch.pi / 180 + beam_direction_rad  # 仰角，-90到90度
    phi_range = torch.linspace(-90, 90, 100) * torch.pi / 180  # 方位角，-90到90度

    theta, phi = torch.meshgrid(theta_range, phi_range, indexing='ij')

    # 计算方向矢量
    u = torch.sin(theta) * torch.cos(phi)
    v = torch.sin(theta) * torch.sin(phi)

    # 计算阵列因子
    af_m = torch.sin(m * torch.pi * d * u / wavelength) / \
        (m * torch.sin(torch.pi * d * u / wavelength))
    af_n = torch.sin(n * torch.pi * d * v / wavelength) / \
        (n * torch.sin(torch.pi * d * v / wavelength))
    af = af_m * af_n

    # 防止除以零的错误
    af[torch.isnan(af)] = 0

    # 转换为球坐标系以绘图
    x = af * torch.sin(theta) * torch.cos(phi)
    y = af * torch.sin(theta) * torch.sin(phi)
    z = af * torch.cos(theta)

    # 使用plotly绘图
    fig = go.Figure(
        data=[go.Surface(z=z.numpy(), x=x.numpy(), y=y.numpy(), colorscale='Viridis')])

    fig.update_layout(title='3D Array Pattern with Beam Direction', autosize=False,
                      width=800, height=800,
                      margin=dict(l=65, r=50, b=65, t=90))

    # 设置坐标轴范围
    fig.update_layout(scene=dict(
        xaxis=dict(nticks=4, range=[-1, 1],),
        yaxis=dict(nticks=4, range=[-1, 1],),
        zaxis=dict(nticks=4, range=[-1, 1],)),
    )

    fig.show()


# 使用Plotly绘制m行n列的均匀平面阵列3D方向图，并指定波束指向
plot_array_pattern_with_plotly(m=4, n=4, d=0.5, beam_direction=0)
