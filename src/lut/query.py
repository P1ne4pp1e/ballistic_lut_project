# ============================================================================
# 文件: src/lut/query.py (完全重写)
# 新算法: 零点搜索法 - 固定v0和distance，找pitch最小的解
# ============================================================================

"""
查表查询引擎 - 零点搜索算法

核心思路:
1. 弹速精确匹配(无需插值)
2. 固定distance，计算所有θ的 err = h_table - h_target
3. 线性插值找err的零点
4. 选择θ最小的解(优先直射，避免攻顶)

性能: ~0.5-0.8ms/次，满足300Hz要求
"""

import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm


class TrajectoryLUT:
    """查表查询引擎 - 零点搜索法"""

    def __init__(self, lut_path: str = '../data/trajectory_lut_full.h5'):
        print(f"加载查表: {lut_path}")
        start_time = __import__('time').time()

        self.f = h5py.File(lut_path, 'r')

        # 读取元数据
        self.v0_list = np.array(self.f['v0_list'])
        self.theta_list = np.array(self.f['theta_list'])

        self.n_v0 = int(self.f.attrs['n_v0'])
        self.n_theta = int(self.f.attrs['n_theta'])
        self.dv0 = float(self.f.attrs['dv0'])
        self.dtheta = float(self.f.attrs['dtheta'])
        self.v0_min = float(self.f.attrs['v0_min'])
        self.v0_max = float(self.f.attrs['v0_max'])

        # 缓存轨迹到内存
        self.trajectories = {}
        traj_group = self.f['trajectories']

        pbar = tqdm(total=self.n_v0 * self.n_theta, desc="加载轨迹")
        for i_v in range(self.n_v0):
            for i_theta in range(self.n_theta):
                key = f'{i_v:04d}_{i_theta:04d}'
                grp = traj_group[key]
                self.trajectories[key] = {
                    'points': np.array(grp['points']),
                    'times': np.array(grp['times']),
                    'v0': float(grp.attrs['v0']),
                    'theta_deg': float(grp.attrs['theta_deg']),
                    'flight_time': float(grp.attrs['flight_time'])
                }
                pbar.update(1)
        pbar.close()

        elapsed = __import__('time').time() - start_time
        print(f"✓ 查表加载完成！耗时 {elapsed:.1f}s\n")

    def _linear_interp_h_at_d(self, points: np.ndarray, d_target: float) -> tuple:
        """
        在轨迹points中，在d=d_target处线性插值h

        Args:
            points: [N, 2] 轨迹点 [(d0,h0), (d1,h1), ...]
            d_target: 目标水平距离

        Returns:
            idx: 差值位置的index向上取整
            h_interp: 插值得到的高度

        Raises:
            ValueError: 如果d_target超出轨迹范围
        """
        distances = points[:, 0]
        heights = points[:, 1]

        # 严格边界检查 - 不允许外推
        if d_target < distances[0]:
            raise ValueError(f"d_target={d_target:.3f}m 小于轨迹最小距离 {distances[0]:.3f}m")
        if d_target > distances[-1]:
            raise ValueError(f"d_target={d_target:.3f}m 大于轨迹最大距离 {distances[-1]:.3f}m (可能超出射程)")

        # 二分查找d_target的位置
        idx = np.searchsorted(distances, d_target)

        # 如果正好在采样点上
        if idx < len(distances) and abs(distances[idx] - d_target) < 1e-6:
            return idx, heights[idx]

        # 线性插值
        d0, d1 = distances[idx - 1], distances[idx]
        h0, h1 = heights[idx - 1], heights[idx]

        h_interp = h0 + (h1 - h0) * (d_target - d0) / (d1 - d0)
        return idx, h_interp

    def query(self, d_target: float, h_target: float, v0_query: float,
              theta_precision: float = 0.01,
              debug: bool = False) -> tuple:
        """
        零点搜索法查询 - 早停优化版

        算法流程:
        1. 定位v0索引(精确匹配)
        2. 从小到大扫描θ，找第一个零点就返回
        3. 可选：检查斜率确保是直射解（err斜率>0）

        关键优化: 早停 - 找到第一个零点立即返回
        - 典型扫描: 50-100个点（原600个）
        - 性能提升: 3-6倍
        - 预期耗时: 0.1-0.3ms

        Args:
            d_target: 目标水平距离 (m)
            h_target: 目标高度 (m)
            v0_query: 当前初速度 (m/s) - 必须在表中且最多一位小数
            theta_precision: 预留参数，当前未使用
            debug: 是否打印调试信息

        Returns:
            (theta_deg, flight_time, error_mm)

            theta_deg: 最优仰角 (度)
            flight_time: 飞行时间 (s)
            error_mm: 最终误差 (mm) - 理论上为0

        性能: ~0.1-0.3ms/次 (早停优化)
        """
        # === Step 0: 参数检查 ===
        if v0_query < self.v0_min or v0_query > self.v0_max:
            raise ValueError(f"v0={v0_query:.1f} 超出范围 [{self.v0_min}, {self.v0_max}]")

        # === Step 1: 定位v0索引(精确匹配) ===
        i_v = int(round((v0_query - self.v0_min) / self.dv0))
        i_v = np.clip(i_v, 0, self.n_v0 - 1)

        actual_v0 = self.v0_list[i_v]

        if debug:
            print(f"\n{'=' * 80}")
            print(f"查询参数: d={d_target:.3f}m, h={h_target:.3f}m, v0={v0_query:.1f}m/s")
            print(f"匹配到表中速度: v0={actual_v0:.1f}m/s (索引 i_v={i_v})")
            print(f"{'=' * 80}\n")

        # === Step 2: 从小到大扫描θ，找第一个零点(早停) ===
        err_prev = None
        i_min_err = 0
        min_abs_err = float('inf')

        for i_theta in range(self.n_theta):
            key = f'{i_v:04d}_{i_theta:04d}'
            points = self.trajectories[key]['points']
            times = self.trajectories[key]['times']

            # 在d_target处插值h
            try:
                i_min_d, h_interp = self._linear_interp_h_at_d(points, d_target)
            except ValueError as e:
                # d_target超出此轨迹范围，跳过
                if debug:
                    print(f"  θ={self.theta_list[i_theta]:.1f}°: 超出射程，跳过")
                err_prev = None  # 重置，避免错误的零点检测
                continue

            # err = h_table - h_target
            err_curr = h_interp - h_target

            # 记录最小误差（备用）
            abs_err = abs(err_curr)
            if abs_err < min_abs_err:
                min_abs_err = abs_err
                i_min_err = i_theta

            # 检查零点
            if err_prev is not None and err_prev * err_curr < 0:
                # 找到零点！

                # 计算斜率
                slope = (err_curr - err_prev) / self.dtheta

                if debug:
                    print(
                        f"找到第一个零点 @ θ区间 [{self.theta_list[i_theta - 1]:.1f}°, {self.theta_list[i_theta]:.1f}°]")
                    print(f"  err变化: {err_prev * 1000:.1f}mm -> {err_curr * 1000:.1f}mm")
                    print(f"  斜率: {slope:.4f} (>0 表示直射解)")

                # 斜率检查（可选，增强鲁棒性）
                if slope > 0:
                    # 这是直射解，线性插值求精确θ
                    theta_i = self.theta_list[i_theta - 1]
                    theta_i1 = self.theta_list[i_theta]

                    # 零点θ
                    theta_zero = theta_i - err_prev * self.dtheta / (err_curr - err_prev)

                    # 插值飞行时间
                    key_i = f'{i_v:04d}_{i_theta - 1:04d}'
                    key_i1 = f'{i_v:04d}_{i_theta:04d}'
                    t_i = self.trajectories[key_i]['times'][i_min_d - 1]
                    t_i1 = self.trajectories[key_i1]['times'][i_min_d]

                    weight = (theta_zero - theta_i) / self.dtheta
                    t_zero = t_i + weight * (t_i1 - t_i)

                    if debug:
                        print(f"\n✓ 最优解(直射): θ={theta_zero:.2f}°, 飞行时间={t_zero:.4f}s")
                        print(f"  扫描了 {i_theta + 1}/{self.n_theta} 个角度")
                        print(f"  早停加速比: {self.n_theta / (i_theta + 1):.1f}x")
                        print(f"{'=' * 80}\n")

                    return theta_zero, t_zero, 0.0

                else:
                    # 斜率 < 0，这是攻顶解，继续搜索
                    if debug:
                        print(f"  ⚠️  斜率<0，这是攻顶解，跳过，继续搜索...\n")

            err_prev = err_curr

        # === Step 3: 未找到零点，返回误差最小的 ===
        if debug:
            print(f"⚠️  未找到零点，使用误差最小解")
            print(f"  扫描了全部 {self.n_theta} 个角度\n")

        theta_best = self.theta_list[i_min_err]
        key_best = f'{i_v:04d}_{i_min_err:04d}'
        t_best = self.trajectories[key_best]['flight_time']
        error_mm = min_abs_err * 1000

        if debug:
            print(f"最优解: θ={theta_best:.2f}°, 飞行时间={t_best:.4f}s, 误差={error_mm:.2f}mm")
            print(f"{'=' * 80}\n")

        return theta_best, t_best, error_mm

    def query_batch(self, targets: list, v0_query: float,
                    theta_precision: float = 0.01,
                    debug: bool = False) -> list:
        """
        批量查询

        Args:
            targets: [(d1, h1), (d2, h2), ...] 目标列表
            v0_query: 当前初速度
            theta_precision: 角度精度
            debug: 是否调试

        Returns:
            results: [(theta1, t1, err1), (theta2, t2, err2), ...]
        """
        results = []
        for d_target, h_target in targets:
            result = self.query(d_target, h_target, v0_query, theta_precision, debug)
            results.append(result)
        return results

    def get_trajectory_info(self, v0: float, theta_deg: float) -> dict:
        """
        获取指定(v0, θ)的轨迹信息(用于可视化、调试)

        Args:
            v0: 初速度 (m/s)
            theta_deg: 仰角 (度)

        Returns:
            {
                'points': [(d, h), ...],
                'times': [t, ...],
                'flight_time': float,
                'max_height': float,
                'max_distance': float
            }
        """
        # 找最近的索引
        i_v = int(round((v0 - self.v0_min) / self.dv0))
        i_v = np.clip(i_v, 0, self.n_v0 - 1)

        i_theta = np.argmin(np.abs(self.theta_list - theta_deg))

        key = f'{i_v:04d}_{i_theta:04d}'
        traj_data = self.trajectories[key]

        points = traj_data['points']

        return {
            'points': points,
            'times': traj_data['times'],
            'flight_time': traj_data['flight_time'],
            'max_height': np.max(points[:, 1]),
            'max_distance': np.max(points[:, 0]),
            'actual_v0': self.v0_list[i_v],
            'actual_theta': self.theta_list[i_theta]
        }

    def close(self):
        """关闭文件"""
        self.f.close()