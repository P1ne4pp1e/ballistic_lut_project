import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm


class TrajectoryLUT:
    """查表查询引擎"""

    def __init__(self, lut_path: str = 'data/trajectory_lut.h5'):
        print(f"加载查表: {lut_path}")
        start_time = __import__('time').time()

        self.f = h5py.File(lut_path, 'r')

        # 读取元数据 - 明确转换numpy标量到Python类型
        self.v0_list = np.array(self.f['v0_list'])
        self.theta_list = np.array(self.f['theta_list'])

        # HDF5 attrs返回numpy标量,需要用item()提取
        self.n_v0 = int(self.f.attrs['n_v0'])
        self.n_theta = int(self.f.attrs['n_theta'])
        self.dv0 = float(self.f.attrs['dv0'])
        self.v0_min = float(self.f.attrs['v0_min'])

        # print(f"DEBUG: n_v0={self.n_v0} (type={type(self.n_v0)})")
        # print(f"DEBUG: dv0={self.dv0} (type={type(self.dv0)})")
        # print(f"DEBUG: v0_min={self.v0_min} (type={type(self.v0_min)})")

        # 检查完整性
        expected_trajectories = self.n_v0 * self.n_theta
        actual_trajectories = len(self.f['trajectories'].keys())

        if actual_trajectories != expected_trajectories:
            print(f"⚠ 警告: 查表不完整!")
            print(f"  预期: {expected_trajectories:,} 条轨迹")
            print(f"  实际: {actual_trajectories:,} 条轨迹")
            print(f"  建议重新生成查表: python scripts/generate_lut.py\n")

        # 缓存轨迹
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

    def _find_v0_index(self, v0):
        """找到v0最近的索引 - 返回标量int"""
        # DEBUG
        # print(f"  v0={v0} (type={type(v0)}, shape={getattr(v0, 'shape', 'N/A')})")
        # print(f"  v0_min={self.v0_min} (type={type(self.v0_min)})")
        # print(f"  dv0={self.dv0} (type={type(self.dv0)})")

        # 计算索引
        idx = (v0 - self.v0_min) / self.dv0
        # print(f"  raw idx={idx} (type={type(idx)}, shape={getattr(idx, 'shape', 'N/A')})")

        idx = np.round(idx)
        # print(f"  rounded idx={idx} (type={type(idx)}, shape={getattr(idx, 'shape', 'N/A')})")

        idx = np.clip(idx, 0, self.n_v0 - 1)
        # print(f"  clipped idx={idx} (type={type(idx)}, shape={getattr(idx, 'shape', 'N/A')})")

        # 转为Python int
        return int(idx)

    def _distance_to_trajectory(self, points, d_target, h_target):
        """计算目标点到轨迹的最小距离"""
        distances = np.sqrt((points[:, 0] - d_target) ** 2 + (points[:, 1] - h_target) ** 2)
        min_dist = np.min(distances)
        min_idx = np.argmin(distances)
        return min_dist, points[min_idx]

    def query(self, d_target: float, h_target: float, v0: float) -> tuple:
        """
        查表查询

        Args:
            d_target: 目标水平距离 (m)
            h_target: 目标高度 (m)
            v0: 当前初速度 (m/s)

        Returns:
            (theta_deg, error_m, flight_time)

            theta_deg: 最优仰角 (度)
            error_m: 预测误差 (m)
            flight_time: 飞行时间 (s)
        """
        i_v = self._find_v0_index(v0)

        # 对所有θ，计算误差
        candidates = []

        for i_theta in range(self.n_theta):
            key = f'{i_v:04d}_{i_theta:04d}'
            traj_data = self.trajectories[key]
            points = traj_data['points']

            # 找轨迹上最近的点
            min_dist, closest_point = self._distance_to_trajectory(
                points, d_target, h_target
            )

            # 记录候选
            candidates.append({
                'theta_deg': traj_data['theta_deg'],
                'i_theta': i_theta,
                'error': min_dist,
                'flight_time': traj_data['flight_time']
            })

        # 按误差排序
        candidates.sort(key=lambda x: x['error'])

        # 筛选有效候选（误差<5cm）
        valid_candidates = [c for c in candidates if c['error'] < 0.05]

        if not valid_candidates:
            # 无有效解,返回最近的
            best = candidates[0]
        else:
            # 在有效候选中选择pitch最小的
            valid_candidates.sort(key=lambda x: x['theta_deg'])
            best = valid_candidates[0]

        return best['theta_deg'], best['error'], best['flight_time']

    def get_time_at_distance(self, theta_deg: float, v0: float,
                             d_query: float) -> float:
        """
        获取指定仰角和速度下，到达某水平距离时的飞行时间

        Args:
            theta_deg: 仰角 (度)
            v0: 初速度 (m/s)
            d_query: 查询的水平距离 (m)

        Returns:
            flight_time: 飞行时间 (s)
        """
        i_v = self._find_v0_index(v0)
        i_theta = int(np.argmin(np.abs(self.theta_list - theta_deg)))

        key = f'{i_v:04d}_{i_theta:04d}'
        traj_data = self.trajectories[key]

        points = traj_data['points']
        times = traj_data['times']

        distances = points[:, 0]

        if d_query < distances[0] or d_query > distances[-1]:
            return None

        # 二分查找
        idx = np.searchsorted(distances, d_query)

        if idx == 0:
            return 0.0
        if idx == len(distances):
            return times[-1]

        # 线性插值
        d0, d1 = distances[idx - 1], distances[idx]
        t0, t1 = times[idx - 1], times[idx]

        t_query = t0 + (d_query - d0) / (d1 - d0) * (t1 - t0)
        return t_query

    def close(self):
        """关闭文件"""
        self.f.close()