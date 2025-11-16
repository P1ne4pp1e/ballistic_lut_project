"""配置管理"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple
import yaml
from pathlib import Path


@dataclass
class PhysicsConfig:
    """物理参数配置"""

    # 球体参数
    radius: float = 0.0085  # 17mm弹丸
    mass: float = 0.005  # 5g

    # 环境参数
    g: float = 9.8  # 重力加速度
    rho: float = 1.2  # 空气密度
    cd: float = 0.47  # 阻力系数

    def __post_init__(self):
        """计算派生参数"""
        self.area = np.pi * self.radius ** 2
        self.k = (self.cd * self.rho * self.area) / (2 * self.mass)

    def __repr__(self) -> str:
        return (f"PhysicsConfig(\n"
                f"  radius={self.radius * 1000:.1f}mm, mass={self.mass * 1000:.1f}g\n"
                f"  g={self.g:.1f}m/s², rho={self.rho:.2f}kg/m³\n"
                f"  cd={self.cd:.2f}, k={self.k:.6f}s⁻¹\n"
                f")")


@dataclass
class LUTConfig:
    """查表配置"""

    # 速度参数：10~30 m/s，间隔0.05
    v0_min: float = 10.0
    v0_max: float = 30.0
    dv0: float = 0.05

    # 角度参数：-10°~60°，间隔0.1°
    theta_min: float = -10.0
    theta_max: float = 60.0
    dtheta: float = 0.1

    # 数值求解参数
    dt: float = 0.0001  # RK4时间步长
    max_time: float = 5.0  # 最大飞行时间
    distance_sample_interval: float = 0.1  # 采样点间隔

    def __post_init__(self):
        """计算派生参数"""
        self.v0_list = np.arange(self.v0_min, self.v0_max + self.dv0 / 2, self.dv0)
        self.theta_list = np.arange(self.theta_min, self.theta_max + self.dtheta / 2, self.dtheta)
        self.theta_rad_list = np.radians(self.theta_list)

        self.n_v0 = len(self.v0_list)
        self.n_theta = len(self.theta_list)
        self.total_trajectories = self.n_v0 * self.n_theta

    def __repr__(self) -> str:
        return (f"LUTConfig(\n"
                f"  v0: {self.n_v0} points ({self.v0_min}~{self.v0_max} m/s, Δ={self.dv0})\n"
                f"  θ: {self.n_theta} points ({self.theta_min}~{self.theta_max}°, Δ={self.dtheta}°)\n"
                f"  Total trajectories: {self.total_trajectories:,}\n"
                f")")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'LUTConfig':
        """从YAML文件加载配置"""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)['lut']
        return cls(**data)