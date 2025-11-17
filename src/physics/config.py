"""配置管理 - 修复版"""

import numpy as np
from dataclasses import dataclass
import yaml
from pathlib import Path


@dataclass
class PhysicsConfig:
    """物理参数配置"""

    # 球体参数
    radius: float = 0.0085
    mass: float = 0.005

    # 环境参数
    g: float = 9.8
    rho: float = 1.2
    cd: float = 0.47

    def __post_init__(self):
        """计算派生参数"""
        self.area = np.pi * self.radius ** 2
        self.k = (self.cd * self.rho * self.area) / (2 * self.mass)

    def __repr__(self) -> str:
        return (f"PhysicsConfig(\n"
                f"  radius={self.radius * 1000:.1f}mm, mass={self.mass * 1000:.1f}g\n"
                f"  g={self.g:.2f}m/s², rho={self.rho:.2f}kg/m³\n"
                f"  cd={self.cd:.2f}, k={self.k:.6f}s⁻¹\n"
                f")")

    @classmethod
    def from_yaml(cls, yaml_path: str = 'config.yaml') -> 'PhysicsConfig':
        """从YAML文件加载配置"""
        # 自动查找项目根目录的config.yaml
        if not Path(yaml_path).is_absolute():
            # 尝试多个可能的路径
            candidates = [
                Path(yaml_path),  # 当前目录
                Path(__file__).parent.parent.parent / yaml_path,  # 项目根目录
                Path.cwd() / yaml_path,  # 工作目录
            ]

            yaml_path_obj = None
            for candidate in candidates:
                if candidate.exists():
                    yaml_path_obj = candidate
                    break

            if yaml_path_obj is None:
                print(f"警告: 无法找到 {yaml_path},使用默认配置")
                return cls()

            yaml_path = str(yaml_path_obj)

        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        if 'physics' not in config_data:
            print("警告: YAML中缺少physics配置,使用默认值")
            return cls()

        physics_data = config_data['physics']
        return cls(**physics_data)


@dataclass
class LUTConfig:
    """查表配置"""

    # 速度参数
    v0_min: float = 10.0
    v0_max: float = 30.0
    dv0: float = 0.05

    # 角度参数
    theta_min: float = -10.0
    theta_max: float = 60.0
    dtheta: float = 0.1

    # 数值求解参数
    dt: float = 0.0001
    max_time: float = 5.0
    distance_sample_interval: float = 0.1

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
    def from_yaml(cls, yaml_path: str = 'config.yaml') -> 'LUTConfig':
        """从YAML文件加载配置"""
        # 自动查找项目根目录的config.yaml
        if not Path(yaml_path).is_absolute():
            candidates = [
                Path(yaml_path),
                Path(__file__).parent.parent.parent / yaml_path,
                Path.cwd() / yaml_path,
            ]

            yaml_path_obj = None
            for candidate in candidates:
                if candidate.exists():
                    yaml_path_obj = candidate
                    break

            if yaml_path_obj is None:
                print(f"警告: 无法找到 {yaml_path},使用默认配置")
                return cls()

            yaml_path = str(yaml_path_obj)

        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        if 'lut' not in config_data:
            print("警告: YAML中缺少lut配置,使用默认值")
            return cls()

        lut_data = config_data['lut']
        return cls(**lut_data)