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

        # 读取元数据
        self.v0_list = np.array(self.f['v0_list'])
        self.theta_list = np.array(self.f['theta_list'])
        self.n_v0 = int(self.f.attrs['n_v0'])
        self.n_theta = int(self.f.attrs['n_theta'])
        self.dv0 = float(self.f.attrs['dv0'])

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
        """找到v0最近的索引"""
        idx = np.round((v0 - self.v0_list[0]) / self.dv0).astype(int)
        idx = np.clip(idx, 0, self.n_v0 - 1)
        return idx

    def _distance_to_trajectory(self, points, d_target, h_target):
        """计算目标点到轨迹的最小距离"""
        distances = np.sqrt((points[:, 0] - d_target)**2 + (points[:, 1] - h_target)**2)
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

        示例:
            >>> theta, error, t_flight = lut.query(20.0, 2.5, 21.5)
            >>> print(f"仰角: {theta:.1f}°, 误差: {error*1000:.1f}mm, 飞行时间: {t_flight:.3f}s")
            仰角: 24.5°, 误差: 1.2mm, 飞行时间: 2.345s
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
            print(f"警告：目标超出射程或无有效解")
            print(f"最近的候选：θ={candidates[0]['theta_deg']:.1f}°, 误差={candidates[0]['error']:.3f}m")
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

        示例:
            >>> t = lut.get_time_at_distance(25.0, 21.0, 15.0)
            >>> print(f"到达15m的飞行时间: {t:.3f}s")
            到达15m的飞行时间: 0.847s
        """
        i_v = np.clip(int(np.round((v0 - self.v0_list[0]) / self.dv0)), 0, self.n_v0 - 1)
        i_theta = np.argmin(np.abs(self.theta_list - theta_deg))

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
        d0, d1 = distances[idx-1], distances[idx]
        t0, t1 = times[idx-1], times[idx]

        t_query = t0 + (d_query - d0) / (d1 - d0) * (t1 - t0)
        return t_query

    def close(self):
        """关闭文件"""
        self.f.close()