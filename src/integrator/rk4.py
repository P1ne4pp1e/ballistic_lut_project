# ============================================================================
# 文件: src/integrator/rk4.py (修改)
# 直接修改 integrate_trajectory 返回飞行时间
# ============================================================================

"""
修改后的RK4积分器 - integrate_trajectory 直接返回飞行时间
"""

import torch
import numpy as np
from typing import Tuple


class GPUTrajectoryIntegrator:
    """GPU加速的RK4轨迹积分器"""

    def __init__(self, physics_config, lut_config, device: str = 'cuda'):
        self.physics = physics_config
        self.lut = lut_config
        self.device = device if torch.cuda.is_available() else 'cpu'

        self.g = torch.tensor(physics_config.g, device=self.device, dtype=torch.float32)
        self.k = torch.tensor(physics_config.k, device=self.device, dtype=torch.float32)
        self.dt = torch.tensor(lut_config.dt, device=self.device, dtype=torch.float32)

    def compute_acceleration(self, vd: torch.Tensor, vh: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算加速度"""
        v = torch.sqrt(vd**2 + vh**2 + 1e-10)
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

        vd_new = vd + (self.dt / 6) * (k1_vd + 2*k2_vd + 2*k3_vd + k4_vd)
        vh_new = vh + (self.dt / 6) * (k1_vh + 2*k2_vh + 2*k3_vh + k4_vh)
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
            landing_d: 落地水平距离 (m)
            landing_h: 落地高度 (m)
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

        while t < self.lut.max_time:
            d, h, vd, vh = self.rk4_step(d, h, vd, vh)
            h_val = h.item()
            d_val = d.item()
            t += self.lut.dt

            # 采样点
            if d_val - last_sample_d >= self.lut.distance_sample_interval:
                points.append((d_val, h_val))
                times.append(t)
                last_sample_d = d_val

            # 落地检测
            if h_val < 0:
                points.append((d_val, h_val))
                times.append(t)
                landing_d = d_val
                landing_h = h_val
                flight_time = t
                break
        else:
            landing_d = d.item()
            landing_h = h.item()
            flight_time = t

        return np.array(points), np.array(times), landing_d, landing_h, flight_time