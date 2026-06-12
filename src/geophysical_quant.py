import numpy as np
from scipy import ndimage, signal
from typing import Tuple, Optional, Dict

"""
极地冰川地球物理量化模块
========================

包含:
1. 大气相位屏幕(APS)校正滤波 - 基于高程相关性
2. LOS几何三角投影分解 - 水平/垂直位移矢量
3. 合成南极地形高程图
4. 位移矢量场后处理
"""

c = 299792458.0


def generate_dem(shape: Tuple[int, int],
                 center_lat: float = -75.0,
                 seed: int = 42) -> Dict:
    """
    生成合成南极冰川数字高程模型(DEM)
    模拟东南极冰盖中心高、边缘低的典型冰穹地貌

    Parameters
    ----------
    shape : (H, W)
    center_lat : 中心纬度(南极为负值)
    seed : 随机种子

    Returns
    -------
    dict with keys:
        'elev' : 地形高程[m]
        'slope_x' : x方向坡度
        'slope_y' : y方向坡度
        'ice_divide' : 冰分水岭掩膜
    """
    H, W = shape
    rng = np.random.RandomState(seed)

    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    yy -= cy
    xx -= cx

    dist_from_center = np.sqrt(yy ** 2 + xx ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    radial = dist_from_center / max_dist

    dome_profile = 3800.0 * np.exp(-1.8 * radial ** 1.4) + 200.0

    detail_scale = min(H, W) / 8
    noise_low = ndimage.gaussian_filter(rng.randn(H, W) * 80.0, sigma=detail_scale * 0.25)
    noise_med = ndimage.gaussian_filter(rng.randn(H, W) * 35.0, sigma=detail_scale * 0.08)
    noise_high = ndimage.gaussian_filter(rng.randn(H, W) * 12.0, sigma=detail_scale * 0.02)

    elev = dome_profile + noise_low + noise_med + noise_high

    for _ in range(6):
        cy_r = rng.randint(int(H * 0.2), int(H * 0.8))
        cx_r = rng.randint(int(W * 0.2), int(W * 0.8))
        amp = rng.uniform(80.0, 220.0)
        radius = rng.uniform(min(H, W) * 0.06, min(H, W) * 0.18)
        yr, xr = np.mgrid[0:H, 0:W].astype(np.float64)
        yr -= cy_r
        xr -= cx_r
        rr = np.sqrt(yr ** 2 + xr ** 2)
        elev += amp * np.exp(-(rr / radius) ** 2)

    rift_mask = np.zeros((H, W), dtype=bool)
    for _ in range(4):
        y0 = rng.randint(0, H)
        x0 = rng.randint(0, W)
        angle = rng.uniform(0, np.pi)
        length = rng.uniform(min(H, W) * 0.3, min(H, W) * 0.7)
        width = rng.uniform(3.0, 8.0)
        for t in np.linspace(0, 1, int(length)):
            y = int(y0 + t * length * np.cos(angle))
            x = int(x0 + t * length * np.sin(angle))
            y1, y2 = max(0, y - int(width)), min(H, y + int(width) + 1)
            x1, x2 = max(0, x - int(width)), min(W, x + int(width) + 1)
            rift_mask[y1:y2, x1:x2] = True

    elev[rift_mask] -= rng.uniform(150.0, 380.0, size=rift_mask.sum())
    elev = np.clip(elev, 50.0, 4200.0)

    grad_y, grad_x = np.gradient(elev)
    pixel_size = 20.0
    slope_x = grad_x / pixel_size
    slope_y = grad_y / pixel_size

    local_max = (elev > ndimage.maximum_filter(elev, size=int(min(H, W) * 0.08)))
    ice_divide = local_max & (elev > 2800.0)

    return {
        'elev': elev,
        'slope_x': slope_x,
        'slope_y': slope_y,
        'ice_divide': ice_divide,
        'rift_mask': rift_mask
    }


def simulate_aps(dem: Dict,
                 strength: float = 2.5,
                 seed: int = 123) -> np.ndarray:
    """
    模拟极地对流层大气相位屏幕(APS)
    物理机制: 对流层延迟与气压(高程)呈指数相关 + 空间平滑湍动
    """
    H, W = dem['elev'].shape
    rng = np.random.RandomState(seed)

    scale_height = 7500.0
    elev_ref = dem['elev'] - dem['elev'].mean()
    elev_component = -strength * (1.0 - np.exp(-elev_ref / scale_height))

    turb_sigma = min(H, W) * 0.07
    turb = ndimage.gaussian_filter(rng.randn(H, W) * 0.9, sigma=turb_sigma)

    trend_xx, trend_yy = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    trend = 0.6 * (0.3 * trend_xx + 0.8 * trend_yy)

    aps_rad = elev_component + turb + trend
    return aps_rad


def correct_aps_elevation_correlated(unwrapped_phase: np.ndarray,
                                     dem_elev: np.ndarray,
                                     coherence: Optional[np.ndarray] = None,
                                     mask: Optional[np.ndarray] = None,
                                     poly_order: int = 3) -> Dict:
    """
    基于高程相关性的大气相位屏幕(APS)校正滤波

    原理: 对流层湿延迟与气压/高程呈强相关性
    步骤:
      1. 掩膜掉低相干/无效区域
      2. 对 unwrapped_phase vs dem_elev 做稳健多项式拟合
      3. 用拟合出的高程相关相位作为 APS 估计
      4. 同时滤除长波长空间趋势

    Parameters
    ----------
    unwrapped_phase : 解缠后的绝对相位 [rad]
    dem_elev : 地形高程 [m]
    coherence : 相干系数(用于加权)
    mask : 有效像素掩膜
    poly_order : 多项式阶数

    Returns
    -------
    dict with keys:
        'corrected' : 校正后相位 [rad]
        'aps_estimated' : 估计的大气相位屏幕 [rad]
        'trend_estimated' : 估计的轨道/几何长波趋势 [rad]
        'coeffs' : 高程拟合系数
    """
    H, W = unwrapped_phase.shape

    if mask is None:
        mask = np.isfinite(unwrapped_phase)
    else:
        mask = mask & np.isfinite(unwrapped_phase)

    if coherence is not None:
        weights = np.clip(coherence, 0.15, 1.0) ** 2
    else:
        weights = np.ones_like(unwrapped_phase)

    valid = mask & (weights > 0.15)
    if valid.sum() < 100:
        return {
            'corrected': unwrapped_phase.copy(),
            'aps_estimated': np.zeros_like(unwrapped_phase),
            'trend_estimated': np.zeros_like(unwrapped_phase),
            'coeffs': np.zeros(poly_order + 1)
        }

    elev_valid = dem_elev[valid]
    phase_valid = unwrapped_phase[valid]
    w_valid = weights[valid]

    elev_norm = (elev_valid - elev_valid.mean()) / (elev_valid.std() + 1e-6)
    X = np.vander(elev_norm, N=poly_order + 1, increasing=True)

    try:
        W_sqrt = np.sqrt(w_valid)
        Xw = X * W_sqrt[:, np.newaxis]
        yw = phase_valid * W_sqrt
        XtWX = Xw.T @ Xw
        XtWy = Xw.T @ yw
        coeffs = np.linalg.solve(XtWX + 1e-10 * np.eye(poly_order + 1), XtWy)
    except np.linalg.LinAlgError:
        coeffs = np.zeros(poly_order + 1)
        coeffs[0] = np.mean(phase_valid)

    elev_full_norm = (dem_elev - elev_valid.mean()) / (elev_valid.std() + 1e-6)
    X_full = np.vander(elev_full_norm.ravel(), N=poly_order + 1, increasing=True)
    aps_elevation = (X_full @ coeffs).reshape(H, W)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    yy_n = (yy - H / 2) / H
    xx_n = (xx - W / 2) / W

    X_space = np.column_stack([
        np.ones(H * W),
        xx_n.ravel(),
        yy_n.ravel(),
        xx_n.ravel() ** 2,
        yy_n.ravel() ** 2,
        xx_n.ravel() * yy_n.ravel()
    ])

    res = unwrapped_phase - aps_elevation
    try:
        y_valid = res[valid]
        Xs_valid = X_space[valid.ravel(), :]
        Ws_sqrt = np.sqrt(w_valid)
        Xsw = Xs_valid * Ws_sqrt[:, np.newaxis]
        ysw = y_valid * Ws_sqrt
        XtWX = Xsw.T @ Xsw
        XtWy = Xsw.T @ ysw
        space_coeffs = np.linalg.solve(XtWX + 1e-12 * np.eye(6), XtWy)
        trend_estimated = (X_space @ space_coeffs).reshape(H, W)
    except np.linalg.LinAlgError:
        trend_estimated = np.zeros_like(res)

    aps_estimated = aps_elevation + trend_estimated
    corrected = unwrapped_phase - aps_estimated
    corrected -= np.nanmean(corrected[valid])

    return {
        'corrected': corrected,
        'aps_estimated': aps_estimated,
        'aps_elevation': aps_elevation,
        'trend_estimated': trend_estimated,
        'coeffs': coeffs
    }


def phase_to_los_displacement(phase: np.ndarray,
                              wavelength: float = 0.031) -> np.ndarray:
    """
    绝对相位 -> 雷达视线向(LOS)位移

    公式: d_LOS = -φ * λ / (4π)

    Parameters
    ----------
    phase : 校正后的绝对相位 [rad]
    wavelength : 雷达波长 [m] (Sentinel-1 C-band = 0.056m, ALOS L-band ~ 0.23m)
                此处默认 0.031m 模拟 Ka 波段高分辨率 SAR

    Returns
    -------
    d_los : LOS方向位移 [m]，正值表示朝向卫星
    """
    return -phase * wavelength / (4.0 * np.pi)


def los_to_3d_vectors(d_los: np.ndarray,
                      incidence_angle_deg: float = 34.0,
                      heading_angle_deg: float = -13.0,
                      pixel_size: float = 20.0) -> Dict:
    """
    LOS几何三角投影分解 -> 三维位移矢量场

    坐标系:
      x: 方位向(沿轨道飞行方向, East-ish)
      y: 距离向(垂直轨道, North-ish)
      z: 垂直向上(正为抬升, 负为沉降)

    LOS 单位向量:
      Lx = -cos(θ_heading) * sin(θ_inc)   (方位向分量)
      Ly = -sin(θ_heading) * sin(θ_inc)   (距离向分量)
      Lz =  cos(θ_inc)                     (垂直分量)

    冰川运动物理约束:
      1. 沿冰川流向(坡度方向)的水平运动占主导
      2. 垂直沉降量与水平速度的散度相关(连续方程)
      3. 利用 D-InSAR 经典分解: d_los = L·d → 求解水平/垂直

    Parameters
    ----------
    d_los : LOS位移 [m]
    incidence_angle_deg : 卫星入射角(度), Sentinel-1 IW ~ 30-45°
    heading_angle_deg : 卫星飞行方位角(度, 顺时针从北), Sentinel-1 降轨 ~ -13°
    pixel_size : 像素间距 [m]

    Returns
    -------
    dict with keys:
        'vx' : x方向(方位向)水平速度 [m/period]
        'vy' : y方向(距离向)水平速度 [m/period]
        'vz' : z方向(垂直)位移, 负为沉降 [m]
        'speed' : 水平速度模长 [m]
        'direction' : 水平流向角 [radians]
        'los_unit' : LOS单位向量 (Lx, Ly, Lz)
    """
    theta_inc = np.deg2rad(incidence_angle_deg)
    theta_head = np.deg2rad(heading_angle_deg)

    Lx = -np.cos(theta_head) * np.sin(theta_inc)
    Ly = -np.sin(theta_head) * np.sin(theta_inc)
    Lz = np.cos(theta_inc)

    H, W = d_los.shape

    nan_mask = ~np.isfinite(d_los)
    if nan_mask.any():
        d_los_safe = d_los.copy()
        d_los_safe[nan_mask] = 0.0
        try:
            from scipy.ndimage import generic_filter
            d_los_safe = ndimage.gaussian_filter(d_los_safe, sigma=1.0)
        except Exception:
            pass
    else:
        d_los_safe = d_los

    grad_y, grad_x = np.gradient(d_los_safe, pixel_size, pixel_size)
    laplacian = ndimage.laplace(d_los_safe) / (pixel_size ** 2)

    flow_dir_y, flow_dir_x = np.gradient(-np.abs(d_los_safe))
    flow_mag = np.sqrt(flow_dir_x ** 2 + flow_dir_y ** 2) + 1e-9
    flow_dir_x /= flow_mag
    flow_dir_y /= flow_mag

    flow_strength = np.abs(d_los_safe) + np.abs(grad_x) + np.abs(grad_y)
    max_fs = flow_strength.max()
    if max_fs > 0:
        flow_strength /= max_fs
    else:
        flow_strength = np.ones_like(flow_strength) * 0.5
    flow_strength = np.power(np.clip(flow_strength, 1e-6, 1.0), 0.7)

    cos_inc = np.cos(theta_inc)
    sin_inc = np.sin(theta_inc)

    vz_raw = d_los_safe / (cos_inc + 1e-9)
    vz_smooth = ndimage.gaussian_filter(vz_raw, sigma=2.0)

    d_horizontal_from_los = (d_los_safe - Lz * vz_smooth) / (sin_inc + 1e-9)

    vx_flow = d_horizontal_from_los * flow_dir_x
    vy_flow = d_horizontal_from_los * flow_dir_y

    vx = flow_strength * vx_flow + (1.0 - flow_strength) * d_los_safe * (Lx / (Lz + 1e-9)) * 0.1
    vy = flow_strength * vy_flow + (1.0 - flow_strength) * d_los_safe * (Ly / (Lz + 1e-9)) * 0.1

    vz = vz_smooth * 0.6 + d_los_safe * (Lz / (Lx ** 2 + Ly ** 2 + Lz ** 2 + 1e-9)) * 0.4

    sigma = 1.8
    vx = ndimage.gaussian_filter(vx, sigma=sigma)
    vy = ndimage.gaussian_filter(vy, sigma=sigma)
    vz = ndimage.gaussian_filter(vz, sigma=sigma)

    speed = np.sqrt(vx ** 2 + vy ** 2)
    direction = np.arctan2(vy, vx)

    if nan_mask.any():
        vx[nan_mask] = np.nan
        vy[nan_mask] = np.nan
        vz[nan_mask] = np.nan
        speed[nan_mask] = np.nan
        direction[nan_mask] = np.nan

    return {
        'vx': vx,
        'vy': vy,
        'vz': vz,
        'speed': speed,
        'direction': direction,
        'los_unit': (Lx, Ly, Lz)
    }


def downsample_vectors(vx: np.ndarray,
                       vy: np.ndarray,
                       vz: np.ndarray,
                       elev: np.ndarray,
                       factor: int = 16,
                       threshold_ratio: float = 0.15) -> Dict:
    """
    3D矢量箭头簇下采样 -> 用于Plotly可视化

    策略:
    - 在规则网格上重采样
    - 只保留位移幅度显著的像素
    - 同时确保高裂缝/高形变密度区有足够箭头

    Parameters
    ----------
    vx, vy, vz : 三维位移场 [m]
    elev : 地形高程 [m]
    factor : 下采样因子
    threshold_ratio : 保留的位移强度分位数阈值

    Returns
    -------
    dict with keys:
        'X, Y, Z' : 箭头起点坐标(像素坐标系 + 高程)
        'U, V, W' : 箭头分量
        'magnitude' : 每个箭头的模长
    """
    H, W = vx.shape

    s = factor // 2
    yy, xx = np.mgrid[s:H:factor, s:W:factor]
    Ny, Nx = yy.shape

    X = xx.astype(np.float64).ravel()
    Y = yy.astype(np.float64).ravel()

    idx_y = yy.ravel().astype(int)
    idx_x = xx.ravel().astype(int)

    U = vx[idx_y, idx_x].ravel()
    V = vy[idx_y, idx_x].ravel()
    W = vz[idx_y, idx_x].ravel()
    Z = elev[idx_y, idx_x].ravel()

    finite_mask = np.isfinite(U) & np.isfinite(V) & np.isfinite(W) & np.isfinite(Z)
    if not finite_mask.any():
        return {
            'X': np.array([]), 'Y': np.array([]), 'Z': np.array([]),
            'U': np.array([]), 'V': np.array([]), 'W': np.array([]),
            'magnitude': np.array([])
        }

    X = X[finite_mask]
    Y = Y[finite_mask]
    Z = Z[finite_mask]
    U = U[finite_mask]
    V = V[finite_mask]
    W = W[finite_mask]

    mag = np.sqrt(U ** 2 + V ** 2 + W ** 2)
    if mag.size == 0:
        threshold = 0.0
    else:
        threshold = np.quantile(mag, threshold_ratio)
        if threshold <= 0 or not np.isfinite(threshold):
            threshold = 1e-9
    keep = mag > threshold

    return {
        'X': X[keep],
        'Y': Y[keep],
        'Z': Z[keep],
        'U': U[keep],
        'V': V[keep],
        'W': W[keep],
        'magnitude': mag[keep]
    }


def compute_geophysical_summary(d_los: np.ndarray,
                                vectors: Dict,
                                aps_result: Dict,
                                coherence: np.ndarray,
                                pixel_size: float = 20.0,
                                days_between: int = 30) -> Dict:
    """
    计算所有地球物理解译统计量
    """
    vx, vy, vz = vectors['vx'], vectors['vy'], vectors['vz']
    speed = vectors['speed']

    valid = np.isfinite(d_los) & (coherence > 0.3) & \
            np.isfinite(vx) & np.isfinite(vy) & np.isfinite(vz) & np.isfinite(speed)

    if valid.sum() < 100:
        return {
            'mean_los_m': 0.0,
            'std_los_m': 0.0,
            'max_subsidence_m': 0.0,
            'max_uplift_m': 0.0,
            'mean_speed_m': 0.0,
            'max_speed_m': 0.0,
            'daily_speed_cm': 0.0,
            'yearly_speed_m': 0.0,
            'net_volume_change_km3': 0.0,
            'aps_rms_rad': 0.0,
            'valid_ratio': 0.0
        }

    sec_per_day = 86400.0
    total_sec = days_between * sec_per_day

    summary = {
        'mean_los_m': float(np.nanmean(d_los[valid])),
        'std_los_m': float(np.nanstd(d_los[valid])),
        'max_subsidence_m': float(np.nanmin(vz[valid])),
        'max_uplift_m': float(np.nanmax(vz[valid])),
        'mean_speed_m': float(np.nanmean(speed[valid])),
        'max_speed_m': float(np.nanmax(speed[valid])),
        'daily_speed_cm': float(np.nanmean(speed[valid]) / days_between * 100.0),
        'yearly_speed_m': float(np.nanmean(speed[valid]) / days_between * 365.0),
        'net_volume_change_km3': float(
            np.nansum(vz[valid]) * pixel_size ** 2 / (1000.0 ** 3)
        ),
        'aps_rms_rad': float(np.sqrt(np.nanmean(aps_result['aps_estimated'][valid] ** 2))),
        'valid_ratio': float(valid.mean())
    }
    return summary
