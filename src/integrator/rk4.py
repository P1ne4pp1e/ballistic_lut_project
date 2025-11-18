# ============================================================================
# 文件: src/integrator/rk4.py (修改版)
# 新增: custom_dt参数，支持实时运算时使用更粗的时间步长
# ============================================================================

"""
GPU加速的RK4轨迹积分器 - 支持自定义时间步长
"""

import torch
import numpy as np
from typing import Tuple


class GPUTrajectoryIntegrator:
    """GPU加速的RK4轨迹积分器"""

    def __init__(self, physics_config, lut_config, device: str = 'cuda', custom_dt: float = None):
        """
        初始化积分器

        Args:
            physics_config: 物理参数配置
            lut_config: 查表配置
            device: 'cuda' or 'cpu'
            custom_dt: 自定义时间步长(s)，如果为None则使用lut_config.dt
                      建议:
                        - 表生成(离线): None (使用默认0.0001s，高精度)
                        - 实时查询: 0.001s (速度提升10倍，精度仍可接受)
        """
        self.physics = physics_config
        self.lut = lut_config
        self.device = device if torch.cuda.is_available() else 'cpu'

        self.g = torch.tensor(physics_config.g, device=self.device, dtype=torch.float32)
        self.k = torch.tensor(physics_config.k, device=self.device, dtype=torch.float32)

        # 使用自定义时间步长（如果提供）
        dt_value = custom_dt if custom_dt is not None else lut_config.dt
        self.dt = torch.tensor(dt_value, device=self.device, dtype=torch.float32)

        # 记录实际使用的dt
        self._actual_dt = dt_value

    def compute_acceleration(self, vd: torch.Tensor, vh: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算加速度"""
        v = torch.sqrt(vd ** 2 + vh ** 2 + 1e-10)
        a_d = -self.k * v * vd
        a_h = -self.k * v * vh - self.g
        return a_d, a_h

    def rk4_step(self, d: torch.Tensor, h: torch.Tensor,
                 vd: torch.Tensor, vh: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """单步RK4积分"""
        k1_vd, k1_vh = self.compute_acceleration(vd, vh)

        vd_k2 = vd + 0.5 * self.dt * k1_vd
        vh_k2 = vh + 0.5 * self.dt * k1_vh
        k2_vd, k2_vh = self.compute_acceleration(vd_k2, vh_k2)

        vd_k3 = vd + 0.5 * self.dt * k2_vd
        vh_k3 = vh + 0.5 * self.dt * k2_vh
        k3_vd, k3_vh = self.compute_acceleration(vd_k3, vh_k3)

        vd_k4 = vd + self.dt * k3_vd
        vh_k4 = vh + self.dt * k3_vh
        k4_vd, k4_vh = self.compute_acceleration(vd_k4, vh_k4)

        vd_new = vd + (self.dt / 6) * (k1_vd + 2 * k2_vd + 2 * k3_vd + k4_vd)
        vh_new = vh + (self.dt / 6) * (k1_vh + 2 * k2_vh + 2 * k3_vh + k4_vh)
        d_new = d + vd * self.dt
        h_new = h + vh * self.dt

        return d_new, h_new, vd_new, vh_new

    def integrate_trajectory(self, v0: float, theta: float) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        """
        积分单条轨迹

        Args:
            v0: 初始速度 (m/s)
            theta: 仰角 (rad)

        Returns:
            (points, times, landing_d, landing_h, flight_time)

            points: [(d, h), ...] 轨迹采样点
            times: [t0, t1, ...] 对应的飞行时间
            landing_d: 最终水平距离 (m)
            landing_h: 最终高度 (m)
            flight_time: 总飞行时间 (s)
        """
        vd0 = v0 * np.cos(theta)
        vh0 = v0 * np.sin(theta)

        d = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        h = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        vd = torch.tensor(vd0, device=self.device, dtype=torch.float32)
        vh = torch.tensor(vh0, device=self.device, dtype=torch.float32)

        points = [(0.0, 0.0)]
        times = [0.0]
        last_sample_d = 0.0
        t = 0.0

        # 积分到 max_time，不做落地检测
        while t < self.lut.max_time:
            d, h, vd, vh = self.rk4_step(d, h, vd, vh)
            h_val = h.item()
            d_val = d.item()
            t += self._actual_dt

            # 采样点
            if d_val - last_sample_d >= self.lut.distance_sample_interval:
                points.append((d_val, h_val))
                times.append(t)
                last_sample_d = d_val

        # 添加最后一个点
        final_d = d.item()
        final_h = h.item()
        if len(points) == 0 or (points[-1][0] != final_d or points[-1][1] != final_h):
            points.append((final_d, final_h))
            times.append(t)

        return np.array(points), np.array(times), final_d, final_h, t

    @property
    def dt_value(self) -> float:
        """返回当前使用的时间步长"""
        return self._actual_dt