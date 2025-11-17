import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm


class TrajectoryLUT:
    """查表查询引擎 - 分离查询和误差验证"""

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
        self.dtheta = float(self.f.attrs['dtheta'])
        self.v0_min = float(self.f.attrs['v0_min'])
        self.v0_max = float(self.f.attrs['v0_max'])

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

        # 初始化积分器（延迟加载，仅在需要误差验证时使用）
        self._integrator = None

    def _get_integrator(self):
        """延迟加载积分器（避免import循环）"""
        if self._integrator is None:
            from src.integrator.rk4 import GPUTrajectoryIntegrator
            from src.physics.config import PhysicsConfig, LUTConfig

            physics_config = PhysicsConfig.from_yaml('config.yaml')
            lut_config = LUTConfig.from_yaml('config.yaml')
            self._integrator = GPUTrajectoryIntegrator(physics_config, lut_config)

        return self._integrator

    def _compute_error_for_traj(self, points: np.ndarray, d_target: float, h_target: float) -> float:
        """
        计算单条轨迹到目标点的最小距离

        Args:
            points: 轨迹点 [n, 2]
            d_target: 目标水平距离
            h_target: 目标高度

        Returns:
            min_error: 最小误差 (m)
        """
        distances = np.sqrt(
            (points[:, 0] - d_target) ** 2 +
            (points[:, 1] - h_target) ** 2
        )
        return np.min(distances)

    def _compute_errors_for_v0(self, i_v: int, d_target: float, h_target: float) -> np.ndarray:
        """
        计算给定v0索引下所有θ的误差（基于查表）

        Args:
            i_v: 速度索引
            d_target: 目标水平距离
            h_target: 目标高度

        Returns:
            errors: [n_theta] 每个角度的最小误差
        """
        errors = np.zeros(self.n_theta)

        for i_theta in range(self.n_theta):
            key = f'{i_v:04d}_{i_theta:04d}'
            points = self.trajectories[key]['points']
            errors[i_theta] = self._compute_error_for_traj(points, d_target, h_target)

        return errors

    def _refine_theta(self, i_theta_best: int, errors: np.ndarray) -> float:
        """
        使用二次抛物线插值精化角度

        Args:
            i_theta_best: 最优角度索引
            errors: 所有角度的误差

        Returns:
            theta_refined: 精化后的角度(度)
        """
        if i_theta_best == 0 or i_theta_best == self.n_theta - 1:
            return self.theta_list[i_theta_best]

        e_prev = errors[i_theta_best - 1]
        e_curr = errors[i_theta_best]
        e_next = errors[i_theta_best + 1]

        dtheta = self.dtheta
        a = (e_next + e_prev - 2 * e_curr) / (2 * dtheta ** 2)
        b = (e_next - e_prev) / (2 * dtheta)

        if abs(a) < 1e-10 or a < 0:
            return self.theta_list[i_theta_best]

        delta_theta = -b / (2 * a)
        delta_theta = np.clip(delta_theta, -dtheta, dtheta)

        theta_refined = self.theta_list[i_theta_best] + delta_theta
        return theta_refined

    def query(self, d_target: float, h_target: float, v0_query: float,
              error_threshold: float = 0.1,
              use_theta_refine: bool = True,
              debug: bool = False) -> tuple:
        """
        查表查询 - 快速返回最优解（不计算真实误差）

        Args:
            d_target: 目标水平距离 (m)
            h_target: 目标高度 (m)
            v0_query: 当前初速度 (m/s)
            error_threshold: 误差阈值 (m)
            use_theta_refine: 是否使用角度二次插值精化
            debug: 是否打印调试信息

        Returns:
            (theta_deg, estimated_error, flight_time)

            theta_deg: 最优仰角 (度)
            estimated_error: 基于查表的估计误差 (m) - 不是真实误差！
            flight_time: 估计飞行时间 (s)
        """
        # 边界检查
        if v0_query < self.v0_min or v0_query > self.v0_max:
            if debug:
                print(f"⚠ 警告: v0={v0_query:.2f} 超出范围 [{self.v0_min}, {self.v0_max}]")
            v0_query = np.clip(v0_query, self.v0_min, self.v0_max)

        # === Step 1: 找v0的两个邻居 ===
        i_v_low = int(np.floor((v0_query - self.v0_min) / self.dv0))
        i_v_high = min(i_v_low + 1, self.n_v0 - 1)

        # 如果正好在网格点上
        if i_v_low == i_v_high or abs(v0_query - self.v0_list[i_v_low]) < 1e-6:
            return self._query_single_v0(d_target, h_target, i_v_low,
                                         error_threshold, use_theta_refine, debug)

        # === Step 2: 计算插值权重 ===
        v0_low = self.v0_list[i_v_low]
        v0_high = self.v0_list[i_v_high]

        w_low = (v0_high - v0_query) / self.dv0
        w_high = (v0_query - v0_low) / self.dv0

        if debug:
            print(f"\n{'=' * 80}")
            print(f"查询参数: d={d_target:.2f}m, h={h_target:.2f}m, v0={v0_query:.2f}m/s")
            print(f"速度插值: v0_low={v0_low:.2f}, v0_high={v0_high:.2f}")
            print(f"插值权重: w_low={w_low:.3f}, w_high={w_high:.3f}")
            print(f"{'=' * 80}\n")

        # === Step 3: 计算两个速度的误差分布 ===
        errors_low = self._compute_errors_for_v0(i_v_low, d_target, h_target)
        errors_high = self._compute_errors_for_v0(i_v_high, d_target, h_target)

        # === Step 4: 插值误差 ===
        errors_interp = w_low * errors_low + w_high * errors_high

        # === Step 5: 收集所有候选解信息 ===
        candidates = []

        for i_theta in range(self.n_theta):
            theta = self.theta_list[i_theta]
            error = errors_interp[i_theta]

            t_low = self.trajectories[f'{i_v_low:04d}_{i_theta:04d}']['flight_time']
            t_high = self.trajectories[f'{i_v_high:04d}_{i_theta:04d}']['flight_time']
            t_interp = w_low * t_low + w_high * t_high

            candidates.append({
                'i_theta': i_theta,
                'theta': theta,
                'error': error,
                'flight_time': t_interp
            })

        # === Step 6: 筛选有效候选 ===
        valid_candidates = [c for c in candidates if c['error'] < error_threshold]

        if debug and valid_candidates:
            print(f"找到 {len(valid_candidates)} 个有效候选解\n")

        # === Step 7: 选择飞行时间最短的有效解 ===
        if valid_candidates:
            best = min(valid_candidates, key=lambda x: x['flight_time'])
            best_idx = best['i_theta']
        else:
            relaxed_candidates = [c for c in candidates if c['error'] < error_threshold * 2]

            if relaxed_candidates:
                best = min(relaxed_candidates, key=lambda x: x['flight_time'])
                best_idx = best['i_theta']
            else:
                best = min(candidates, key=lambda x: x['error'])
                best_idx = best['i_theta']

        best_theta = self.theta_list[best_idx]
        best_error = errors_interp[best_idx]

        # === Step 8: 角度精化 ===
        if use_theta_refine:
            best_theta = self._refine_theta(best_idx, errors_interp)

        # === Step 9: 插值飞行时间 ===
        flight_time_low = self.trajectories[f'{i_v_low:04d}_{best_idx:04d}']['flight_time']
        flight_time_high = self.trajectories[f'{i_v_high:04d}_{best_idx:04d}']['flight_time']
        flight_time = w_low * flight_time_low + w_high * flight_time_high

        if debug:
            print(f"最优解: θ={best_theta:.2f}°, 估计误差={best_error * 1000:.2f}mm, "
                  f"飞行时间={flight_time:.4f}s\n")

        return best_theta, best_error, flight_time

    def _query_single_v0(self, d_target: float, h_target: float, i_v: int,
                         error_threshold: float = 0.05,
                         use_theta_refine: bool = False,
                         debug: bool = False) -> tuple:
        """单个v0的查询（无需速度插值）"""
        errors = self._compute_errors_for_v0(i_v, d_target, h_target)

        candidates = []
        for i_theta in range(self.n_theta):
            theta = self.theta_list[i_theta]
            error = errors[i_theta]
            flight_time = self.trajectories[f'{i_v:04d}_{i_theta:04d}']['flight_time']

            candidates.append({
                'i_theta': i_theta,
                'theta': theta,
                'error': error,
                'flight_time': flight_time
            })

        valid_candidates = [c for c in candidates if c['error'] < error_threshold]

        if valid_candidates:
            best = min(valid_candidates, key=lambda x: x['flight_time'])
            best_idx = best['i_theta']
        else:
            relaxed_candidates = [c for c in candidates if c['error'] < error_threshold * 2]
            if relaxed_candidates:
                best = min(relaxed_candidates, key=lambda x: x['flight_time'])
                best_idx = best['i_theta']
            else:
                best = min(candidates, key=lambda x: x['error'])
                best_idx = best['i_theta']

        best_theta = self.theta_list[best_idx]
        best_error = errors[best_idx]

        if use_theta_refine:
            best_theta = self._refine_theta(best_idx, errors)

        flight_time = self.trajectories[f'{i_v:04d}_{best_idx:04d}']['flight_time']

        return best_theta, best_error, flight_time

    def compute_real_error(self, d_target: float, h_target: float,
                           v0_query: float, theta_deg: float) -> tuple:
        """
        计算真实误差 - 用查询条件重新积分轨迹

        ⚠️ 警告: 此函数会调用RK4积分，耗时约1-2ms，不应在高频查询中使用！

        使用场景:
        1. 离线验证查表精度
        2. 调试时检查误差
        3. 生成测试报告

        Args:
            d_target: 目标水平距离 (m)
            h_target: 目标高度 (m)
            v0_query: 查询时使用的初速度 (m/s)
            theta_deg: query()返回的最优仰角 (度)

        Returns:
            (real_error, closest_d, closest_h, flight_time_to_target)

            real_error: 真实误差 (m)
            closest_d: 轨迹上最接近目标的点的水平距离 (m)
            closest_h: 轨迹上最接近目标的点的高度 (m)
            flight_time_to_target: 到达最近点的飞行时间 (s)
        """
        integrator = self._get_integrator()

        # 用查询条件积分真实轨迹
        theta_rad = np.radians(theta_deg)
        points, times, _, _, _ = integrator.integrate_trajectory(v0_query, theta_rad)

        # 计算到目标点的距离
        distances = np.sqrt(
            (points[:, 0] - d_target) ** 2 +
            (points[:, 1] - h_target) ** 2
        )

        # 找最小距离
        min_idx = np.argmin(distances)
        real_error = distances[min_idx]
        closest_d = points[min_idx, 0]
        closest_h = points[min_idx, 1]
        flight_time_to_target = times[min_idx]

        return real_error, closest_d, closest_h, flight_time_to_target

    def query_with_validation(self, d_target: float, h_target: float, v0_query: float,
                              error_threshold: float = 0.05,
                              use_theta_refine: bool = False,
                              debug: bool = False) -> dict:
        """
        查询并验证 - 返回完整信息（包括真实误差）

        ⚠️ 警告: 此函数会调用RK4积分，总耗时约2-3ms，不适合高频查询！

        适用场景: 离线测试、调试、验证

        Args:
            d_target: 目标水平距离 (m)
            h_target: 目标高度 (m)
            v0_query: 当前初速度 (m/s)
            error_threshold: 误差阈值 (m)
            use_theta_refine: 是否角度精化
            debug: 是否打印调试信息

        Returns:
            {
                'theta_deg': 最优仰角 (度),
                'estimated_error': 基于查表的估计误差 (m),
                'real_error': 基于RK4积分的真实误差 (m),
                'flight_time': 估计飞行时间 (s),
                'closest_point': (d, h) 真实轨迹最接近目标的点,
                'time_to_target': 到达最近点的时间 (s)
            }
        """
        # Step 1: 快速查询
        theta_deg, estimated_error, flight_time = self.query(
            d_target, h_target, v0_query,
            error_threshold, use_theta_refine, debug
        )

        # Step 2: 计算真实误差
        real_error, closest_d, closest_h, time_to_target = self.compute_real_error(
            d_target, h_target, v0_query, theta_deg
        )

        if debug:
            print(f"\n{'=' * 80}")
            print(f"误差对比:")
            print(f"  估计误差 (查表): {estimated_error * 1000:.2f}mm")
            print(f"  真实误差 (RK4):  {real_error * 1000:.2f}mm")
            print(f"  误差偏差: {abs(real_error - estimated_error) * 1000:.2f}mm")
            print(f"最近点: ({closest_d:.3f}m, {closest_h:.3f}m)")
            print(f"到达时间: {time_to_target:.4f}s")
            print(f"{'=' * 80}\n")

        return {
            'theta_deg': theta_deg,
            'estimated_error': estimated_error,
            'real_error': real_error,
            'flight_time': flight_time,
            'closest_point': (closest_d, closest_h),
            'time_to_target': time_to_target
        }

    def get_time_at_distance(self, theta_deg: float, v0: float, d_query: float) -> float:
        """获取到达指定距离的飞行时间"""
        i_v = int(np.round((v0 - self.v0_min) / self.dv0))
        i_v = np.clip(i_v, 0, self.n_v0 - 1)

        i_theta = int(np.argmin(np.abs(self.theta_list - theta_deg)))

        key = f'{i_v:04d}_{i_theta:04d}'
        traj_data = self.trajectories[key]

        points = traj_data['points']
        times = traj_data['times']
        distances = points[:, 0]

        if d_query < distances[0] or d_query > distances[-1]:
            return None

        idx = np.searchsorted(distances, d_query)

        if idx == 0:
            return times[0]
        if idx == len(distances):
            return times[-1]

        d0, d1 = distances[idx - 1], distances[idx]
        t0, t1 = times[idx - 1], times[idx]

        t_query = t0 + (d_query - d0) / (d1 - d0) * (t1 - t0)
        return t_query

    def close(self):
        """关闭文件"""
        self.f.close()