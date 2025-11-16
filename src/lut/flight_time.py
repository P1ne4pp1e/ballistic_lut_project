"""
飞行时间计算和存储

在查表生成时额外存储飞行时间，查询时可获取。
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class TrajectoryWithTime:
    """包含时间信息的轨迹数据"""
    points: np.ndarray  # [(d, h), ...] 轨迹点
    times: np.ndarray  # 对应的时间
    landing_d: float  # 落地水平距离
    landing_h: float  # 落地高度
    flight_time: float  # 总飞行时间
    v0: float  # 初速度
    theta_deg: float  # 仰角


class FlightTimeCalculator:
    """飞行时间计算器"""

    @staticmethod
    def calculate_flight_time(points: np.ndarray, dt: float) -> float:
        """
        计算飞行时间

        Args:
            points: 轨迹采样点，每点采样间隔为dt
            dt: 单位时间步长 (s)

        Returns:
            flight_time: 总飞行时间 (s)
        """
        # 轨迹点数 = 时间步数 + 1
        flight_time = (len(points) - 1) * dt
        return flight_time

    @staticmethod
    def interpolate_time_at_distance(points: np.ndarray, dt: float,
                                     d_query: float) -> Optional[float]:
        """
        根据水平距离插值获取对应时间

        Args:
            points: 轨迹采样点 [(d0, h0), (d1, h1), ...]
            dt: 单位时间步长
            d_query: 查询的距离

        Returns:
            time_at_distance: 到达该距离时的时间，未找到返回None
        """
        distances = points[:, 0]

        # 检查距离是否在轨迹范围内
        if d_query < distances[0] or d_query > distances[-1]:
            return None

        # 找到最近的两个点
        idx = np.searchsorted(distances, d_query)

        if idx == 0:
            return 0.0
        if idx == len(distances):
            return (len(distances) - 1) * dt

        # 线性插值
        d0, d1 = distances[idx - 1], distances[idx]
        t0, t1 = (idx - 1) * dt, idx * dt

        t_query = t0 + (d_query - d0) / (d1 - d0) * (t1 - t0)
        return t_query