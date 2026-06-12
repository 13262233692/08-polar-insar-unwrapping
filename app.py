import os
import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.slc_processor import SLCProcessor
from src.interferogram import Interferogram
from src.phase_unwrap import PhaseUnwrapper
from src.geophysical_quant import (
    generate_dem,
    simulate_aps,
    correct_aps_elevation_correlated,
    phase_to_los_displacement,
    los_to_3d_vectors,
    downsample_vectors,
    compute_geophysical_summary
)


processor = SLCProcessor(width=1024, height=1024)
interferogram_calc = Interferogram(multi_look_window=(7, 7), coherence_threshold=0.3)
unwrapper = PhaseUnwrapper(quality_threshold=0.3)
current_unwrap_algorithm = 'quality'

print("=" * 60)
print(" 正在生成极地 SAR 复数影像数据...")
print("=" * 60)

master_slc, master_amp = processor.generate_polar_slc(seed=42)
slave_slc = processor.generate_slave_from_master(
    master_slc,
    offset_x=23.4,
    offset_y=-12.7,
    deformation_phase=15.0,
    seed=123
)

print(f" 主影像尺寸: {master_slc.shape}")
print(f" 辅影像尺寸: {slave_slc.shape}")

print("\n" + "=" * 60)
print(" 执行频域互相关配准...")
print("=" * 60)

coreg_result = processor.frequency_domain_coregistration(master_slc, slave_slc)

print("\n" + "=" * 60)
print(" 计算干涉相位与相干系数...")
print("=" * 60)

ifg_result = interferogram_calc.compute_interferogram(
    master_slc,
    coreg_result.registered_image,
    use_gaussian=True
)

print("\n" + "=" * 60)
print(" 冰川漂移分析...")
print("=" * 60)

drift_analysis = interferogram_calc.glacier_drift_analysis(
    ifg_result.wrapped_phase,
    ifg_result.coherence,
    pixel_size=10.0
)

print(f" 平均漂移速率: {drift_analysis['mean_drift']:.2f} m/月")
print(f" 最大漂移速率: {drift_analysis['max_drift']:.2f} m/月")

print("\n" + "=" * 60)
print(" 生成南极冰川数字高程模型 (DEM)...")
print("=" * 60)

dem = generate_dem(master_slc.shape, center_lat=-75.0, seed=42)
print(f" 地形高程范围: [{dem['elev'].min():.1f}, {dem['elev'].max():.1f}] m")
print(f" 平均高程: {dem['elev'].mean():.1f} m")

print("\n" + "=" * 60)
print(" 模拟极地对流层大气相位屏幕 (APS)...")
print("=" * 60)

aps_true = simulate_aps(dem, strength=2.5, seed=123)
ifg_result = interferogram_calc.inject_atmospheric_phase(ifg_result, aps_true)
print(f" 注入 APS 强度 RMS: {np.sqrt(np.mean(aps_true**2)):.3f} rad")

print("\n" + "=" * 60)
print(" 相位解缠 (初始: 质量引导法)...")
print("=" * 60)

unwrap_result = unwrapper.quality_guided_unwrap(
    ifg_result.wrapped_phase,
    ifg_result.coherence
)
residues = unwrapper.detect_residues(ifg_result.wrapped_phase)

print(f" 残差点总数: {unwrap_result.num_residues}")
print(f" 解缠相位范围: [{np.min(unwrap_result.unwrapped_phase):.2f}, {np.max(unwrap_result.unwrapped_phase):.2f}] rad")

print("\n" + "=" * 60)
print(" 基于高程相关性的大气校正滤波...")
print("=" * 60)

DEFAULT_WAVELENGTH_M = 0.056
DEFAULT_INCIDENCE_DEG = 34.0
DEFAULT_HEADING_DEG = -13.0
DEFAULT_PIXEL_SIZE = 20.0

aps_result = correct_aps_elevation_correlated(
    unwrap_result.unwrapped_phase,
    dem['elev'],
    coherence=ifg_result.coherence,
    mask=(ifg_result.coherence > 0.3) & unwrap_result.unwrap_mask,
    poly_order=3
)
corrected_phase = aps_result['corrected']
print(f" APS 估计 RMS: {np.sqrt(np.mean(aps_result['aps_estimated']**2)):.3f} rad")
print(f" 校正后相位 RMS: {np.sqrt(np.mean(corrected_phase**2)):.3f} rad")

print("\n" + "=" * 60)
print(" 相位 → LOS位移 → 三维矢量分解...")
print("=" * 60)

d_los = phase_to_los_displacement(corrected_phase, wavelength=DEFAULT_WAVELENGTH_M)
d_los[ifg_result.coherence < 0.3] = np.nan

vectors_3d = los_to_3d_vectors(
    d_los,
    incidence_angle_deg=DEFAULT_INCIDENCE_DEG,
    heading_angle_deg=DEFAULT_HEADING_DEG,
    pixel_size=DEFAULT_PIXEL_SIZE
)

vec_arrows = downsample_vectors(
    vectors_3d['vx'], vectors_3d['vy'], vectors_3d['vz'],
    dem['elev'], factor=16, threshold_ratio=0.15
)
print(f" 生成 3D 矢量箭头数量: {len(vec_arrows['X'])}")

geo_summary = compute_geophysical_summary(
    d_los, vectors_3d, aps_result, ifg_result.coherence,
    pixel_size=DEFAULT_PIXEL_SIZE, days_between=30
)
print(f" 年均冰川流速: {geo_summary['yearly_speed_m']:.2f} m/yr")
print(f" 最大沉降量: {geo_summary['max_subsidence_m']:.3f} m")
print(f" 净体积变化: {geo_summary['net_volume_change_km3']:.4f} km³")

print("\n" + "=" * 60)
print(" 数据处理完成，启动可视化大屏...")
print("=" * 60)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "南极冰川漂移监测 - 高维地球物理数据分析大屏"

server = app.server


def downsample_for_plot(arr, max_size=400):
    h, w = arr.shape
    if h <= max_size and w <= max_size:
        return arr
    factor = max(h, w) // max_size + 1
    return arr[::factor, ::factor]


def create_amplitude_figure(amplitude, title):
    amp_log = 20 * np.log10(amplitude + 1e-10)
    amp_ds = downsample_for_plot(amp_log)

    fig = go.Figure(data=go.Heatmap(
        z=amp_ds,
        colorscale='Gray',
        showscale=True,
        colorbar=dict(
            title='dB',
            thickness=15,
            len=0.7,
            title_font=dict(color='white'),
            tickfont=dict(color='white')
        )
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color='white', size=14),
            x=0.5
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor='x', scaleratio=1),
        margin=dict(l=10, r=10, t=40, b=10),
        height=280
    )

    return fig


def create_phase_figure(phase, title, coherence=None):
    phase_ds = downsample_for_plot(phase)

    if coherence is not None:
        coh_ds = downsample_for_plot(coherence)
        phase_masked = phase_ds.copy()
        phase_masked[coh_ds < 0.3] = np.nan
    else:
        phase_masked = phase_ds

    fig = go.Figure(data=go.Heatmap(
        z=phase_masked,
        colorscale='HSV',
        showscale=True,
        zmin=-np.pi,
        zmax=np.pi,
        colorbar=dict(
            title='相位 (rad)',
            thickness=15,
            len=0.7,
            title_font=dict(color='white'),
            tickfont=dict(color='white'),
            tickvals=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
            ticktext=['-π', '-π/2', '0', 'π/2', 'π']
        )
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color='white', size=14),
            x=0.5
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor='x', scaleratio=1),
        margin=dict(l=10, r=10, t=40, b=10),
        height=280
    )

    return fig


def create_coherence_figure(coherence, title):
    coh_ds = downsample_for_plot(coherence)

    fig = go.Figure(data=go.Heatmap(
        z=coh_ds,
        colorscale='Viridis',
        showscale=True,
        zmin=0,
        zmax=1,
        colorbar=dict(
            title='相干系数',
            thickness=15,
            len=0.7,
            title_font=dict(color='white'),
            tickfont=dict(color='white')
        )
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color='white', size=14),
            x=0.5
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor='x', scaleratio=1),
        margin=dict(l=10, r=10, t=40, b=10),
        height=280
    )

    return fig


def create_3d_deformation_figure(wrapped_phase, coherence):
    phase_ds = downsample_for_plot(wrapped_phase, max_size=150)
    coh_ds = downsample_for_plot(coherence, max_size=150)

    deformation = phase_ds * 10
    deformation[coh_ds < 0.3] = np.nan

    h, w = deformation.shape
    x = np.arange(w)
    y = np.arange(h)

    fig = go.Figure(data=[go.Surface(
        z=deformation,
        x=x,
        y=y,
        colorscale='RdYlBu_r',
        showscale=True,
        colorbar=dict(
            title='形变量',
            thickness=15,
            len=0.7,
            title_font=dict(color='white'),
            tickfont=dict(color='white')
        ),
        contours=dict(
            z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="limegreen",
                project=dict(z=True)
            )
        )
    )])

    fig.update_layout(
        title=dict(
            text='<b>冰川形变 3D 可视化</b>',
            font=dict(color='cyan', size=16),
            x=0.5
        ),
        scene=dict(
            xaxis_title='方位向 (像素)',
            yaxis_title='距离向 (像素)',
            zaxis_title='相对形变量',
            xaxis=dict(showbackground=True, backgroundcolor='rgba(0,0,0,0.3)', color='white'),
            yaxis=dict(showbackground=True, backgroundcolor='rgba(0,0,0,0.3)', color='white'),
            zaxis=dict(showbackground=True, backgroundcolor='rgba(0,0,0,0.3)', color='white'),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=50, b=10),
        height=500
    )

    return fig


def create_drift_vector_figure(drift_mag, drift_dir, coherence):
    mag_ds = downsample_for_plot(drift_mag, max_size=80)
    dir_ds = downsample_for_plot(drift_dir, max_size=80)
    coh_ds = downsample_for_plot(coherence, max_size=80)

    h, w = mag_ds.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    valid = coh_ds > 0.3
    u = mag_ds * np.cos(np.radians(dir_ds))
    v = mag_ds * np.sin(np.radians(dir_ds))

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=mag_ds,
        colorscale='Jet',
        showscale=True,
        zmin=0,
        zmax=np.nanpercentile(mag_ds, 95),
        colorbar=dict(
            title='漂移速率 (m/月)',
            thickness=15,
            len=0.7,
            title_font=dict(color='white'),
            tickfont=dict(color='white')
        ),
        opacity=0.7
    ))

    step = 6
    xs = x[::step, ::step].flatten()
    ys = y[::step, ::step].flatten()
    us = u[::step, ::step].flatten() * 50
    vs = v[::step, ::step].flatten() * 50
    cohs = coh_ds[::step, ::step].flatten()

    valid_idx = cohs > 0.3
    xs = xs[valid_idx]
    ys = ys[valid_idx]
    us = us[valid_idx]
    vs = vs[valid_idx]

    arrow_x = []
    arrow_y = []
    for i in range(len(xs)):
        arrow_x.extend([xs[i], xs[i] + us[i], None])
        arrow_y.extend([ys[i], ys[i] + vs[i], None])

    fig.add_trace(go.Scatter(
        x=arrow_x,
        y=arrow_y,
        mode='lines',
        line=dict(color='white', width=1),
        showlegend=False,
        opacity=0.7
    ))

    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode='markers',
        marker=dict(color='white', size=2),
        showlegend=False,
        opacity=0.5
    ))

    fig.update_layout(
        title=dict(
            text='<b>冰川漂移速度场</b>',
            font=dict(color='cyan', size=16),
            x=0.5
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor='x', scaleratio=1),
        margin=dict(l=10, r=10, t=50, b=10),
        height=400
    )

    return fig


def create_coherence_histogram(coherence):
    fig = go.Figure(data=[go.Histogram(
        x=coherence.flatten(),
        nbinsx=50,
        marker_color='cyan',
        opacity=0.7
    )])

    fig.update_layout(
        title=dict(
            text='<b>相干系数分布</b>',
            font=dict(color='white', size=13),
            x=0.5
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title='相干系数',
            color='white',
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            title='像素数',
            color='white',
            gridcolor='rgba(255,255,255,0.1)'
        ),
        margin=dict(l=50, r=20, t=40, b=40),
        height=200
    )

    return fig


def create_cross_correlation_figure(cross_corr):
    cc_ds = downsample_for_plot(cross_corr, max_size=200)

    fig = go.Figure(data=go.Heatmap(
        z=cc_ds,
        colorscale='Hot',
        showscale=True,
        colorbar=dict(
            title='互相关',
            thickness=10,
            len=0.6,
            title_font=dict(color='white'),
            tickfont=dict(color='white')
        )
    ))

    fig.update_layout(
        title=dict(
            text='<b>频域互相关</b>',
            font=dict(color='white', size=13),
            x=0.5
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor='x', scaleratio=1),
        margin=dict(l=10, r=10, t=40, b=10),
        height=200
    )

    return fig


def create_residue_figure(residues, title='残差点分布'):
    res_ds = downsample_for_plot(residues, max_size=400)

    pos_y, pos_x = np.where(res_ds == 1)
    neg_y, neg_x = np.where(res_ds == -1)

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=np.zeros_like(res_ds, dtype=float),
        colorscale='Greys',
        showscale=False,
        opacity=0.3
    ))

    fig.add_trace(go.Scatter(
        x=pos_x,
        y=pos_y,
        mode='markers',
        marker=dict(color='red', size=3, symbol='circle'),
        name='正残差 (+)',
        showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=neg_x,
        y=neg_y,
        mode='markers',
        marker=dict(color='blue', size=3, symbol='circle'),
        name='负残差 (-)',
        showlegend=True
    ))

    fig.update_layout(
        title=dict(
            text=f'<b>{title}</b><br><span style="font-size:10px;color:#888">正: {len(pos_x)} | 负: {len(neg_x)} | 总计: {len(pos_x)+len(neg_x)}</span>',
            font=dict(color='white', size=14),
            x=0.5
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor='x', scaleratio=1, autorange='reversed'),
        legend=dict(
            font=dict(color='white', size=10),
            bgcolor='rgba(0,0,0,0.5)',
            x=0.02,
            y=0.98
        ),
        margin=dict(l=10, r=10, t=60, b=10),
        height=280
    )

    return fig


def create_unwrapped_phase_figure(unwrapped_phase, coherence, title='解缠相位图', threshold=0.3):
    phase_ds = downsample_for_plot(unwrapped_phase, max_size=400)
    coh_ds = downsample_for_plot(coherence, max_size=400)

    phase_masked = phase_ds.copy()
    phase_masked[coh_ds < threshold] = np.nan

    fig = go.Figure(data=go.Heatmap(
        z=phase_masked,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(
            title='相位 (rad)',
            thickness=15,
            len=0.7,
            title_font=dict(color='white'),
            tickfont=dict(color='white')
        )
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color='white', size=14),
            x=0.5
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor='x', scaleratio=1),
        margin=dict(l=10, r=10, t=40, b=10),
        height=280
    )

    return fig


def create_branch_cut_figure(branch_cuts, coherence=None, title='枝切线分布'):
    bc_ds = downsample_for_plot(branch_cuts.astype(float), max_size=400)

    fig = go.Figure()

    if coherence is not None:
        coh_ds = downsample_for_plot(coherence, max_size=400)
        fig.add_trace(go.Heatmap(
            z=coh_ds,
            colorscale='Gray',
            showscale=False,
            opacity=0.3
        ))

    fig.add_trace(go.Heatmap(
        z=bc_ds,
        colorscale=[[0, 'rgba(255,0,0,0)'], [1, 'rgba(255,50,50,0.9)']],
        showscale=False
    ))

    num_bc = int(np.sum(branch_cuts))

    fig.update_layout(
        title=dict(
            text=f'<b>{title}</b><br><span style="font-size:10px;color:#888">枝切线长度: {num_bc} 像素</span>',
            font=dict(color='white', size=14),
            x=0.5
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor='x', scaleratio=1),
        margin=dict(l=10, r=10, t=60, b=10),
        height=280
    )

    return fig


def create_unwrap_quality_figure(unwrap_mask, title='解缠有效区域'):
    mask_ds = downsample_for_plot(unwrap_mask.astype(float), max_size=400)
    valid_ratio = np.sum(unwrap_mask) / unwrap_mask.size

    fig = go.Figure(data=go.Heatmap(
        z=mask_ds,
        colorscale=[[0, 'rgba(50,50,80,0.8)'], [1, 'rgba(0,255,150,0.8)']],
        showscale=False
    ))

    fig.update_layout(
        title=dict(
            text=f'<b>{title}</b><br><span style="font-size:10px;color:#888">解缠率: {valid_ratio*100:.1f}%</span>',
            font=dict(color='white', size=14),
            x=0.5
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor='x', scaleratio=1),
        margin=dict(l=10, r=10, t=60, b=10),
        height=280
    )

    return fig


def create_aps_correction_figure(uncorrected, corrected, aps_estimated, title='大气校正前后对比'):
    unc_ds = downsample_for_plot(uncorrected, max_size=300)
    cor_ds = downsample_for_plot(corrected, max_size=300)
    aps_ds = downsample_for_plot(aps_estimated, max_size=300)

    vmax = max(np.nanmax(np.abs(unc_ds)), np.nanmax(np.abs(cor_ds)))

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('校正前相位', '估计APS', '校正后相位'),
        horizontal_spacing=0.06
    )

    fig.add_trace(go.Heatmap(
        z=unc_ds, colorscale='RdBu_r', zmin=-vmax, zmax=vmax, showscale=False
    ), row=1, col=1)
    fig.add_trace(go.Heatmap(
        z=aps_ds, colorscale='RdBu_r', showscale=False
    ), row=1, col=2)
    fig.add_trace(go.Heatmap(
        z=cor_ds, colorscale='RdBu_r', zmin=-vmax, zmax=vmax,
        showscale=True,
        colorbar=dict(
            title='rad', thickness=12, len=0.7,
            title_font=dict(color='white'),
            tickfont=dict(color='white')
        )
    ), row=1, col=3)

    fig.update_layout(
        title=dict(text=f'<b>{title}</b>', font=dict(color='white', size=14), x=0.5),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=280,
        margin=dict(l=10, r=30, t=60, b=10)
    )
    for i in range(1, 4):
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=i)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, scaleanchor='x', scaleratio=1, row=1, col=i)
        fig.update_annotations(font=dict(color='white', size=11))

    return fig


def create_vertical_displacement_figure(vz, coherence, title='垂直沉降图 (向下为负)'):
    vz_ds = downsample_for_plot(vz, max_size=400)
    coh_ds = downsample_for_plot(coherence, max_size=400)
    vz_ds[coh_ds < 0.3] = np.nan

    vmax = np.nanpercentile(np.abs(vz_ds), 98)

    fig = go.Figure(data=go.Heatmap(
        z=vz_ds,
        colorscale='RdBu_r',
        zmin=-vmax, zmax=vmax,
        showscale=True,
        colorbar=dict(
            title='位移 (m)', thickness=15, len=0.7,
            title_font=dict(color='white'), tickfont=dict(color='white')
        )
    ))

    fig.update_layout(
        title=dict(
            text=f'<b>{title}</b><br><span style="font-size:10px;color:#888">蓝:沉降 | 红:抬升</span>',
            font=dict(color='white', size=14), x=0.5
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor='x', scaleratio=1),
        margin=dict(l=10, r=10, t=60, b=10),
        height=280
    )
    return fig


def create_3d_terrain_vectors(elev, vx, vy, vz, coherence, arrows_data, title='3D 冰川地形 + 位移矢量场'):
    elev_ds = downsample_for_plot(elev, max_size=120)
    H, W = elev_ds.shape

    coh_ds = downsample_for_plot(coherence, max_size=120)
    elev_masked = elev_ds.copy()
    elev_masked[coh_ds < 0.15] = np.nan

    x = np.arange(W)
    y = np.arange(H)

    fig = go.Figure()

    fig.add_trace(go.Surface(
        z=elev_masked,
        x=x,
        y=y,
        colorscale='Gray',
        cmin=np.nanpercentile(elev_ds, 5),
        cmax=np.nanpercentile(elev_ds, 95),
        showscale=True,
        opacity=0.85,
        colorbar=dict(
            title='高程 (m)', thickness=15, len=0.6,
            title_font=dict(color='white'), tickfont=dict(color='white'),
            x=0.02
        ),
        contours=dict(
            z=dict(show=True, usecolormap=True, project=dict(z=True), size=200)
        ),
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2),
        lightposition=dict(x=100, y=100, z=1000)
    ))

    H_full, W_full = elev.shape
    sx = W / W_full
    sy = H / H_full
    sz = 1.0

    ax = arrows_data['X'] * sx
    ay = arrows_data['Y'] * sy
    az = arrows_data['Z'] * sz

    ds_factor = max(H_full, W_full) / max(W, H)
    scale_horiz = sx * 60.0 / (np.nanmax(np.sqrt(vx ** 2 + vy ** 2)) + 1e-6)
    scale_vert = sz * 500.0 / (np.nanmax(np.abs(vz)) + 1e-6)

    au = arrows_data['U'] * scale_horiz
    av = arrows_data['V'] * scale_horiz
    aw = arrows_data['W'] * scale_vert
    mag = arrows_data['magnitude']

    cone_sizes = 1.0 + 4.0 * (mag - mag.min()) / (mag.max() - mag.min() + 1e-9)

    fig.add_trace(go.Cone(
        x=ax + au,
        y=ay + av,
        z=az + aw,
        u=au,
        v=av,
        w=aw,
        colorscale=[[0, '#ff2222'], [1, '#ff6644']],
        showscale=True,
        colorbar=dict(
            title='位移 (m)', thickness=15, len=0.6,
            title_font=dict(color='white'), tickfont=dict(color='white'),
            x=0.98
        ),
        sizemode='absolute',
        sizeref=float(np.mean(cone_sizes)),
        opacity=0.95,
        anchor='tip',
        name='位移矢量'
    ))

    mid_idx = np.argsort(mag)[-min(40, len(mag)):]
    line_x, line_y, line_z = [], [], []
    for i in mid_idx:
        line_x.extend([ax[i], ax[i] + au[i], None])
        line_y.extend([ay[i], ay[i] + av[i], None])
        line_z.extend([az[i], az[i] + aw[i], None])

    fig.add_trace(go.Scatter3d(
        x=line_x, y=line_y, z=line_z,
        mode='lines',
        line=dict(color='#ff0000', width=3),
        showlegend=False,
        opacity=0.8
    ))

    fig.update_layout(
        title=dict(
            text=f'<b>{title}</b><br><span style="font-size:10px;color:#aaa">红色箭头:冰川三维流动方向 | 锥头大小:位移强度</span>',
            font=dict(color='cyan', size=15), x=0.5
        ),
        scene=dict(
            xaxis_title='方位向',
            yaxis_title='距离向',
            zaxis_title='高程 (m)',
            xaxis=dict(showbackground=True, backgroundcolor='rgba(0,20,40,0.5)', color='white', showticklabels=False),
            yaxis=dict(showbackground=True, backgroundcolor='rgba(0,20,40,0.5)', color='white', showticklabels=False),
            zaxis=dict(showbackground=True, backgroundcolor='rgba(0,20,40,0.8)', color='white'),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.6),
            camera=dict(eye=dict(x=1.4, y=1.6, z=0.9))
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=60, b=10),
        height=620
    )
    return fig


def create_stat_card(title, value, unit, color='cyan', icon='📊'):
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Span(icon, className='me-2', style={'fontSize': '24px'}),
                html.H6(title, className='text-muted mb-1', style={'fontSize': '12px'})
            ], className='d-flex align-items-center'),
            html.H3(
                f'{value}',
                className='mb-0',
                style={'color': color, 'fontWeight': 'bold'}
            ),
            html.Small(unit, className='text-muted')
        ]),
        className='bg-dark border-secondary h-100',
        style={'borderRadius': '8px'}
    )


header = html.Div([
    dbc.Row([
        dbc.Col([
            html.H2(
                '❄️ 南极冰川漂移监测系统',
                className='text-center text-cyan mb-0',
                style={
                    'color': '#00ffff',
                    'textShadow': '0 0 10px rgba(0, 255, 255, 0.5)',
                    'fontWeight': 'bold',
                    'letterSpacing': '2px'
                }
            ),
            html.P(
                '高维地球物理数据分析大屏 | 合成孔径雷达干涉测量 (InSAR)',
                className='text-center text-secondary mb-0 mt-1',
                style={'fontSize': '14px'}
            )
        ], width=12)
    ]),
    html.Hr(style={'borderColor': 'rgba(0, 255, 255, 0.3)', 'margin': '10px 0'})
])


stats_row = dbc.Row([
    dbc.Col(create_stat_card('X方向偏移', f'{coreg_result.offset_x:.2f}', '像素', '#00ffff', '↔️'), width=2),
    dbc.Col(create_stat_card('Y方向偏移', f'{coreg_result.offset_y:.2f}', '像素', '#00ff00', '↕️'), width=2),
    dbc.Col(create_stat_card('配准精度', f'{coreg_result.correlation_peak:.3f}', '相关峰值', '#ffff00', '🎯'), width=2),
    dbc.Col(create_stat_card('平均相干系数', f'{ifg_result.mean_coherence:.3f}', 'γ', '#ff8800', '📐'), width=2),
    dbc.Col(create_stat_card('平均漂移速率', f'{drift_analysis["mean_drift"]:.2f}', 'm/月', '#ff4444', '🧊'), width=2),
    dbc.Col(create_stat_card('有效像素比', f'{ifg_result.valid_pixel_ratio*100:.1f}', '%', '#aa44ff', '✅'), width=2),
], className='mb-2 g-2')

unwrap_stats_row = dbc.Row([
    dbc.Col(create_stat_card('残差点总数', f'{unwrap_result.num_residues}', '个', '#ff4444', '⚡'), width=2),
    dbc.Col(create_stat_card('解缠相位跨度', f'{np.ptp(unwrap_result.unwrapped_phase):.1f}', 'rad', '#00ffff', '📏'), width=2),
    dbc.Col(create_stat_card('相位周期数', f'{np.ptp(unwrap_result.unwrapped_phase)/(2*np.pi):.1f}', '个 2π', '#00ff88', '🔄'), width=2),
    dbc.Col(create_stat_card('解缠有效率', f'{np.sum(unwrap_result.unwrap_mask)/unwrap_result.unwrap_mask.size*100:.1f}', '%', '#ffff00', '🟢'), width=2),
    dbc.Col(create_stat_card('解缠算法', unwrap_result.algorithm, '', '#ff88ff', '🧮'), width=2),
    dbc.Col(create_stat_card('枝切线长度', f'{unwrap_result.num_branch_cuts}', '像素', '#ff6644', '✂️'), width=2),
], className='mb-2 g-2')

geo_stats_row = dbc.Row([
    dbc.Col(create_stat_card('APS RMS', f'{geo_summary["aps_rms_rad"]:.3f}', 'rad', '#ff44aa', '🌫️'), width=2),
    dbc.Col(create_stat_card('年均流速', f'{geo_summary["yearly_speed_m"]:.2f}', 'm/yr', '#00ffff', '🧊'), width=2),
    dbc.Col(create_stat_card('日流速', f'{geo_summary["daily_speed_cm"]:.2f}', 'cm/天', '#00ff88', '📏'), width=2),
    dbc.Col(create_stat_card('最大沉降', f'{geo_summary["max_subsidence_m"]:.3f}', 'm', '#4488ff', '⬇️'), width=2),
    dbc.Col(create_stat_card('最大抬升', f'{geo_summary["max_uplift_m"]:.3f}', 'm', '#ffaa00', '⬆️'), width=2),
    dbc.Col(create_stat_card('净体积变化', f'{geo_summary["net_volume_change_km3"]:.4f}', 'km³', '#ff6644', '📦'), width=2),
], className='mb-3 g-2')


main_content = dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.CardBody([
                dcc.Graph(
                    id='3d-terrain-vectors-graph',
                    figure=create_3d_terrain_vectors(
                        dem['elev'],
                        vectors_3d['vx'], vectors_3d['vy'], vectors_3d['vz'],
                        ifg_result.coherence,
                        vec_arrows
                    ),
                    config={'displayModeBar': True, 'responsive': True}
                )
            ])
        ], className='bg-dark border-secondary mb-3', style={'borderRadius': '8px'}),

        dbc.Card([
            dbc.CardBody([
                html.H6('🌫️ 大气校正 & 物理解译', className='mb-3', style={'color': '#ff88ff'}),
                dbc.Tabs([
                    dbc.Tab(label='大气校正对比', tab_id='aps_correction'),
                    dbc.Tab(label='垂直沉降场', tab_id='vertical_disp'),
                    dbc.Tab(label='LOS位移场', tab_id='los_disp'),
                    dbc.Tab(label='水平流速场', tab_id='horizontal_speed'),
                ], id='geo-tabs', active_tab='aps_correction'),
                dcc.Graph(
                    id='geo-graph',
                    figure=create_aps_correction_figure(
                        unwrap_result.unwrapped_phase,
                        aps_result['corrected'],
                        aps_result['aps_estimated']
                    ),
                    config={'displayModeBar': False, 'responsive': True}
                )
            ])
        ], className='bg-dark border-secondary mb-3', style={'borderRadius': '8px'}),

        dbc.Card([
            dbc.CardBody([
                html.H6('🧩 相位解缠分析', className='text-cyan mb-3', style={'color': '#00ffff'}),
                dbc.Tabs([
                    dbc.Tab(label='解缠相位', tab_id='unwrapped'),
                    dbc.Tab(label='残差点分布', tab_id='residues'),
                    dbc.Tab(label='枝切线', tab_id='branchcuts'),
                    dbc.Tab(label='解缠质量', tab_id='unwrapquality'),
                ], id='unwrap-tabs', active_tab='unwrapped'),
                dcc.Graph(
                    id='unwrap-graph',
                    figure=create_unwrapped_phase_figure(
                        unwrap_result.unwrapped_phase,
                        ifg_result.coherence,
                        '解缠相位图',
                        0.3
                    ),
                    config={'displayModeBar': False, 'responsive': True}
                )
            ])
        ], className='bg-dark border-secondary', style={'borderRadius': '8px'})
    ], width=8),

    dbc.Col([
        dbc.Card([
            dbc.CardBody([
                dbc.Tabs([
                    dbc.Tab(label='主影像', tab_id='master'),
                    dbc.Tab(label='辅影像', tab_id='slave'),
                    dbc.Tab(label='配准后', tab_id='registered'),
                ], id='amplitude-tabs', active_tab='master'),
                dcc.Graph(
                    id='amplitude-graph',
                    figure=create_amplitude_figure(master_amp, '主影像 - 振幅图'),
                    config={'displayModeBar': False, 'responsive': True}
                )
            ])
        ], className='bg-dark border-secondary mb-3', style={'borderRadius': '8px'}),

        dbc.Card([
            dbc.CardBody([
                dbc.Tabs([
                    dbc.Tab(label='干涉相位', tab_id='phase'),
                    dbc.Tab(label='相干系数', tab_id='coherence'),
                    dbc.Tab(label='互相关', tab_id='crosscorr'),
                ], id='analysis-tabs', active_tab='phase'),
                dcc.Graph(
                    id='analysis-graph',
                    figure=create_phase_figure(
                        ifg_result.wrapped_phase,
                        '干涉相位图',
                        ifg_result.coherence
                    ),
                    config={'displayModeBar': False, 'responsive': True}
                )
            ])
        ], className='bg-dark border-secondary mb-3', style={'borderRadius': '8px'}),

        dbc.Card([
            dbc.CardBody([
                dcc.Graph(
                    id='coherence-histogram',
                    figure=create_coherence_histogram(ifg_result.coherence),
                    config={'displayModeBar': False, 'responsive': True}
                )
            ])
        ], className='bg-dark border-secondary', style={'borderRadius': '8px'})
    ], width=4)
])


control_panel = dbc.Card([
    dbc.CardHeader([
        html.H6('⚙️ 参数控制面板', className='mb-0 text-cyan', style={'color': '#00ffff'})
    ], className='bg-dark border-secondary'),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Label('多视窗口大小', className='text-light small'),
                dcc.Slider(
                    id='multilook-slider',
                    min=1,
                    max=15,
                    step=2,
                    value=7,
                    marks={i: str(i) for i in range(1, 16, 2)},
                    className='mb-3'
                ),
            ], width=6),
            dbc.Col([
                html.Label('相干系数阈值', className='text-light small'),
                dcc.Slider(
                    id='coherence-threshold-slider',
                    min=0.1,
                    max=0.8,
                    step=0.05,
                    value=0.3,
                    marks={0.1: '0.1', 0.3: '0.3', 0.5: '0.5', 0.7: '0.7'},
                    className='mb-3'
                ),
            ], width=6),
        ]),
        dbc.Row([
            dbc.Col([
                html.Label('形变波长 (cm)', className='text-light small'),
                dcc.Input(
                    id='wavelength-input',
                    type='number',
                    value=5.6,
                    min=1,
                    max=100,
                    step=0.1,
                    className='form-control form-control-sm bg-dark text-light border-secondary'
                ),
            ], width=4),
            dbc.Col([
                html.Label('入射角 (°)', className='text-light small'),
                dcc.Input(
                    id='incidence-input',
                    type='number',
                    value=39.0,
                    min=10,
                    max=80,
                    step=1,
                    className='form-control form-control-sm bg-dark text-light border-secondary'
                ),
            ], width=4),
            dbc.Col([
                html.Label('像素大小 (m)', className='text-light small'),
                dcc.Input(
                    id='pixel-size-input',
                    type='number',
                    value=10.0,
                    min=1,
                    max=100,
                    step=1,
                    className='form-control form-control-sm bg-dark text-light border-secondary'
                ),
            ], width=4),
        ]),
        html.Hr(style={'borderColor': 'rgba(255,255,255,0.1)', 'margin': '15px 0'}),
        html.H6('🧩 相位解缠算法', className='text-light small mb-2', style={'color': '#ff88ff'}),
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='unwrap-algo-dropdown',
                    options=[
                        {'label': '质量引导法', 'value': 'quality'},
                        {'label': 'Goldstein 枝切法', 'value': 'branch_cut'},
                        {'label': '最小费用网络流', 'value': 'network_flow'},
                    ],
                    value='quality',
                    className='mb-2',
                    style={'fontSize': '12px', 'backgroundColor': '#1a1a2e'},
                    searchable=False
                ),
            ], width=12),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Button(
                    '🔬 执行相位解缠',
                    id='unwrap-btn',
                    color='warning',
                    size='sm',
                    className='w-100',
                    outline=True
                ),
            ], width=12),
        ]),
        html.Hr(style={'borderColor': 'rgba(255,255,255,0.1)', 'margin': '15px 0'}),
        dbc.Row([
            dbc.Col([
                dbc.Button(
                    '🔄 重新生成数据',
                    id='regenerate-btn',
                    color='primary',
                    size='sm',
                    className='w-100',
                    outline=True
                ),
            ], width=6),
            dbc.Col([
                dbc.Button(
                    '📊 重新计算干涉',
                    id='recalc-btn',
                    color='success',
                    size='sm',
                    className='w-100',
                    outline=True
                ),
            ], width=6),
        ]),
    ])
], className='bg-dark border-secondary mb-3', style={'borderRadius': '8px'})


info_panel = dbc.Card([
    dbc.CardHeader([
        html.H6('📡 监测信息', className='mb-0 text-cyan', style={'color': '#00ffff'})
    ], className='bg-dark border-secondary'),
    dbc.CardBody([
        html.Table([
            html.Tr([
                html.Td('卫星:', className='text-muted'),
                html.Td('Sentinel-1A', className='text-light')
            ]),
            html.Tr([
                html.Td('成像模式:', className='text-muted'),
                html.Td('IW 干涉宽幅', className='text-light')
            ]),
            html.Tr([
                html.Td('波段:', className='text-muted'),
                html.Td('C 波段 (5.6 cm)', className='text-light')
            ]),
            html.Tr([
                html.Td('分辨率:', className='text-muted'),
                html.Td('10 m × 10 m', className='text-light')
            ]),
            html.Tr([
                html.Td('影像尺寸:', className='text-muted'),
                html.Td(f'{processor.width} × {processor.height}', className='text-light')
            ]),
            html.Tr([
                html.Td('时间基线:', className='text-muted'),
                html.Td('30 天', className='text-light')
            ]),
            html.Tr([
                html.Td('监测区域:', className='text-muted'),
                html.Td('南极半岛', className='text-light')
            ]),
            html.Tr([
                html.Td('处理时间:', className='text-muted'),
                html.Td('实时', className='text-success')
            ]),
        ], className='table table-sm table-borderless mb-0'),
    ])
], className='bg-dark border-secondary', style={'borderRadius': '8px'})


sidebar = html.Div([
    control_panel,
    info_panel
])


app.layout = dbc.Container([
    header,
    stats_row,
    unwrap_stats_row,
    geo_stats_row,
    dbc.Row([
        dbc.Col(main_content, width=10),
        dbc.Col(sidebar, width=2),
    ]),
    html.Div(id='hidden-storage', style={'display': 'none'}),
    dcc.Interval(
        id='update-interval',
        interval=5000,
        n_intervals=0
    )
], fluid=True, style={'backgroundColor': '#0a0a0a', 'minHeight': '100vh', 'padding': '15px'})


@app.callback(
    Output('amplitude-graph', 'figure'),
    Input('amplitude-tabs', 'active_tab')
)
def update_amplitude_tab(active_tab):
    if active_tab == 'master':
        return create_amplitude_figure(master_amp, '主影像 - 振幅图')
    elif active_tab == 'slave':
        return create_amplitude_figure(np.abs(slave_slc), '辅影像 - 振幅图')
    else:
        return create_amplitude_figure(np.abs(coreg_result.registered_image), '配准后辅影像 - 振幅图')


@app.callback(
    Output('analysis-graph', 'figure'),
    Input('analysis-tabs', 'active_tab')
)
def update_analysis_tab(active_tab):
    if active_tab == 'phase':
        return create_phase_figure(
            ifg_result.wrapped_phase,
            '干涉相位图 (包裹)',
            ifg_result.coherence
        )
    elif active_tab == 'coherence':
        return create_coherence_figure(
            ifg_result.coherence,
            '干涉相干系数图'
        )
    else:
        return create_cross_correlation_figure(coreg_result.cross_correlation)


@app.callback(
    Output('unwrap-graph', 'figure'),
    Input('unwrap-tabs', 'active_tab')
)
def update_unwrap_tab(active_tab):
    global unwrap_result, residues, ifg_result
    if active_tab == 'unwrapped':
        return create_unwrapped_phase_figure(
            unwrap_result.unwrapped_phase,
            ifg_result.coherence,
            '解缠相位图',
            unwrapper.quality_threshold
        )
    elif active_tab == 'residues':
        return create_residue_figure(residues, '残差点分布')
    elif active_tab == 'branchcuts':
        bc = unwrap_result.branch_cuts
        if bc is None:
            bc = np.zeros_like(residues, dtype=bool)
        return create_branch_cut_figure(bc, ifg_result.coherence, '枝切线分布')
    else:
        mask = unwrap_result.unwrap_mask
        if mask is None:
            mask = np.ones_like(ifg_result.wrapped_phase, dtype=bool)
        return create_unwrap_quality_figure(mask, '解缠有效区域')


@app.callback(
    Output('geo-graph', 'figure'),
    Input('geo-tabs', 'active_tab')
)
def update_geo_tab(active_tab):
    global unwrap_result, aps_result, d_los, vectors_3d, ifg_result
    if active_tab == 'aps_correction':
        return create_aps_correction_figure(
            unwrap_result.unwrapped_phase,
            aps_result['corrected'],
            aps_result['aps_estimated'],
            '基于高程相关性的大气相位屏幕(APS)校正'
        )
    elif active_tab == 'vertical_disp':
        return create_vertical_displacement_figure(
            vectors_3d['vz'],
            ifg_result.coherence,
            '冰川垂直沉降/抬升场 (负=沉降, 正=抬升)'
        )
    elif active_tab == 'los_disp':
        fig = go.Figure()
        los_ds = downsample_for_plot(d_los, max_size=400)
        coh_ds = downsample_for_plot(ifg_result.coherence, max_size=400)
        los_ds[coh_ds < 0.3] = np.nan
        vmax = np.nanpercentile(np.abs(los_ds), 98)
        fig.add_trace(go.Heatmap(
            z=los_ds, colorscale='RdBu_r', zmin=-vmax, zmax=vmax,
            colorbar=dict(title='LOS (m)', thickness=15, len=0.7,
                          title_font=dict(color='white'), tickfont=dict(color='white'))
        ))
        fig.update_layout(
            title=dict(text='<b>雷达视线向(LOS)位移</b><br><span style="font-size:10px;color:#888">正=朝向卫星 | 负=远离卫星</span>',
                       font=dict(color='white', size=14), x=0.5),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor='x', scaleratio=1),
            margin=dict(l=10, r=10, t=60, b=10), height=280
        )
        return fig
    else:
        fig = go.Figure()
        spd_ds = downsample_for_plot(vectors_3d['speed'], max_size=400)
        coh_ds = downsample_for_plot(ifg_result.coherence, max_size=400)
        spd_ds[coh_ds < 0.3] = np.nan
        vmax = np.nanpercentile(spd_ds, 95)
        fig.add_trace(go.Heatmap(
            z=spd_ds, colorscale='Jet', zmin=0, zmax=vmax,
            colorbar=dict(title='速度 (m)', thickness=15, len=0.7,
                          title_font=dict(color='white'), tickfont=dict(color='white'))
        ))
        fig.update_layout(
            title=dict(text='<b>冰川水平流速场</b><br><span style="font-size:10px;color:#888">30天累计水平位移</span>',
                       font=dict(color='white', size=14), x=0.5),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor='x', scaleratio=1),
            margin=dict(l=10, r=10, t=60, b=10), height=280
        )
        return fig


def _recompute_geophysics_after_unwrap():
    """相位解缠后，重新进行大气校正、LOS分解和3D矢量计算"""
    global aps_result, corrected_phase, d_los, vectors_3d, vec_arrows, geo_summary
    aps_result = correct_aps_elevation_correlated(
        unwrap_result.unwrapped_phase,
        dem['elev'],
        coherence=ifg_result.coherence,
        mask=(ifg_result.coherence > 0.3) & unwrap_result.unwrap_mask,
        poly_order=3
    )
    corrected_phase = aps_result['corrected']
    d_los = phase_to_los_displacement(corrected_phase, wavelength=DEFAULT_WAVELENGTH_M)
    d_los[ifg_result.coherence < 0.3] = np.nan
    vectors_3d = los_to_3d_vectors(
        d_los,
        incidence_angle_deg=DEFAULT_INCIDENCE_DEG,
        heading_angle_deg=DEFAULT_HEADING_DEG,
        pixel_size=DEFAULT_PIXEL_SIZE
    )
    vec_arrows = downsample_vectors(
        vectors_3d['vx'], vectors_3d['vy'], vectors_3d['vz'],
        dem['elev'], factor=16, threshold_ratio=0.15
    )
    geo_summary = compute_geophysical_summary(
        d_los, vectors_3d, aps_result, ifg_result.coherence,
        pixel_size=DEFAULT_PIXEL_SIZE, days_between=30
    )


@app.callback(
    Output('hidden-storage', 'children'),
    Input('unwrap-btn', 'n_clicks'),
    State('unwrap-algo-dropdown', 'value'),
    prevent_initial_call=True
)
def run_phase_unwrap(n_clicks, algorithm):
    global unwrap_result, current_unwrap_algorithm
    print(f"\n=== 用户触发相位解缠: {algorithm} ===")

    if algorithm == 'quality':
        unwrap_result = unwrapper.quality_guided_unwrap(
            ifg_result.wrapped_phase,
            ifg_result.coherence
        )
    elif algorithm == 'branch_cut':
        unwrap_result = unwrapper.branch_cut_unwrap(
            ifg_result.wrapped_phase,
            ifg_result.coherence
        )
    elif algorithm == 'network_flow':
        unwrap_result = unwrapper.network_flow_unwrap(
            ifg_result.wrapped_phase,
            ifg_result.coherence
        )

    current_unwrap_algorithm = algorithm
    print(f"=== 解缠完成: {unwrap_result.algorithm} ===")
    print("=== 重新计算大气校正与3D矢量分解 ===")
    _recompute_geophysics_after_unwrap()
    print(f"=== 完成 | 年均流速 {geo_summary['yearly_speed_m']:.2f} m/yr | 最大沉降 {geo_summary['max_subsidence_m']:.3f} m ===\n")
    return str(n_clicks)


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print(" 启动 Dash 服务器: http://127.0.0.1:8050")
    print("=" * 60 + "\n")
    app.run(debug=False, host='0.0.0.0', port=8050)
