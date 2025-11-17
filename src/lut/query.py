import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm


class TrajectoryLUT:
    """查表查询引擎 - 优先选择飞行时间最短的解"""

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
        计算给定v0索引下所有θ的误差

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
        # 边界检查
        if i_theta_best == 0 or i_theta_best == self.n_theta - 1:
            return self.theta_list[i_theta_best]

        # 三个点的误差
        e_prev = errors[i_theta_best - 1]
        e_curr = errors[i_theta_best]
        e_next = errors[i_theta_best + 1]

        # 二次拟合
        dtheta = self.dtheta

        a = (e_next + e_prev - 2 * e_curr) / (2 * dtheta ** 2)
        b = (e_next - e_prev) / (2 * dtheta)

        # 检查是否为极小值
        if abs(a) < 1e-10 or a < 0:
            return self.theta_list[i_theta_best]

        # 计算偏移
        delta_theta = -b / (2 * a)

        # 限制在邻域内
        delta_theta = np.clip(delta_theta, -dtheta, dtheta)

        theta_refined = self.theta_list[i_theta_best] + delta_theta

        return theta_refined

    def query(self, d_target: float, h_target: float, v0_query: float,
              error_threshold: float = 0.05,
              use_theta_refine: bool = False,
              debug: bool = False) -> tuple:
        """
        查表查询 - 优先选择飞行时间最短的有效解

        Args:
            d_target: 目标水平距离 (m)
            h_target: 目标高度 (m)
            v0_query: 当前初速度 (m/s) - 固定输入
            error_threshold: 误差阈值 (m), 默认5cm
            use_theta_refine: 是否使用角度二次插值精化
            debug: 是否打印调试信息

        Returns:
            (theta_deg, error_m, flight_time)

            theta_deg: 最优仰角 (度)
            error_m: 预测误差 (m)
            flight_time: 飞行时间 (s)
        """
        # 边界检查
        if v0_query < self.v0_min or v0_query > self.v0_max:
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
            print(f"误差阈值: {error_threshold * 1000:.1f}mm")
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

            # 获取飞行时间(也需要插值)
            t_low = self.trajectories[f'{i_v_low:04d}_{i_theta:04d}']['flight_time']
            t_high = self.trajectories[f'{i_v_high:04d}_{i_theta:04d}']['flight_time']
            t_interp = w_low * t_low + w_high * t_high

            candidates.append({
                'i_theta': i_theta,
                'theta': theta,
                'error': error,
                'flight_time': t_interp
            })

        # === Step 6: 筛选有效候选(误差<阈值) ===
        valid_candidates = [c for c in candidates if c['error'] < error_threshold]

        if debug:
            print(f"{'序号':<6} {'角度(°)':<10} {'误差(mm)':<12} {'飞行时间(s)':<15} {'有效?':<8}")
            print("-" * 80)

            for cand in candidates:
                is_valid = cand['error'] < error_threshold

                if is_valid:
                    print(f"{cand['i_theta']:<6} {cand['theta']:<10.2f} "
                          f"{cand['error'] * 1000:<12.2f} {cand['flight_time']:<15.4f} "
                          f"{'✓':<8}")

            print(f"\n共找到 {len(valid_candidates)} 个有效候选解 (误差<{error_threshold * 1000:.1f}mm)\n")

        # === Step 7: 选择飞行时间最短的有效解 ===
        if valid_candidates:
            # 关键: 在所有有效解中选择飞行时间最短的
            best = min(valid_candidates, key=lambda x: x['flight_time'])
            best_idx = best['i_theta']

            if debug:
                # 对比误差最小的
                min_error_cand = min(valid_candidates, key=lambda x: x['error'])

                if min_error_cand['i_theta'] != best_idx:
                    print(f"选择策略对比:")
                    print(f"  误差最小: θ={min_error_cand['theta']:.2f}°, "
                          f"误差={min_error_cand['error'] * 1000:.2f}mm, "
                          f"时间={min_error_cand['flight_time']:.4f}s")
                    print(f"  时间最短: θ={best['theta']:.2f}°, "
                          f"误差={best['error'] * 1000:.2f}mm, "
                          f"时间={best['flight_time']:.4f}s")
                    print(f"  → 选择时间最短解 ✓\n")
        else:
            # 无有效解,放宽阈值
            if debug:
                print(f"⚠ 无有效解,尝试放宽阈值到 {error_threshold * 2 * 1000:.1f}mm\n")

            relaxed_candidates = [c for c in candidates if c['error'] < error_threshold * 2]

            if relaxed_candidates:
                best = min(relaxed_candidates, key=lambda x: x['flight_time'])
                best_idx = best['i_theta']
            else:
                # 实在找不到,返回误差最小的
                best = min(candidates, key=lambda x: x['error'])
                best_idx = best['i_theta']

                if debug:
                    print(f"⚠ 警告: 所有解误差均>阈值, 返回误差最小解\n")

        best_theta = self.theta_list[best_idx]
        best_error = errors_interp[best_idx]

        # === Step 8: (可选) θ方向二次插值精化 ===
        if use_theta_refine:
            theta_before_refine = best_theta
            best_theta = self._refine_theta(best_idx, errors_interp)
            if debug:
                print(f"角度精化: {theta_before_refine:.3f}° -> {best_theta:.3f}°\n")

        # === Step 9: 插值飞行时间 ===
        flight_time_low = self.trajectories[f'{i_v_low:04d}_{best_idx:04d}']['flight_time']
        flight_time_high = self.trajectories[f'{i_v_high:04d}_{best_idx:04d}']['flight_time']

        flight_time = w_low * flight_time_low + w_high * flight_time_high

        if debug:
            print(f"{'=' * 80}")
            print(f"最优解: θ={best_theta:.2f}°, 误差={best_error * 1000:.2f}mm, "
                  f"飞行时间={flight_time:.4f}s")
            print(f"{'=' * 80}\n")

        return best_theta, best_error, flight_time

    def _query_single_v0(self, d_target: float, h_target: float, i_v: int,
                         error_threshold: float = 0.05,
                         use_theta_refine: bool = False,
                         debug: bool = False) -> tuple:
        """
        单个v0的查询(无需速度插值)

        Args:
            d_target: 目标水平距离
            h_target: 目标高度
            i_v: 速度索引
            error_threshold: 误差阈值
            use_theta_refine: 是否精化角度
            debug: 是否打印调试信息

        Returns:
            (theta_deg, error_m, flight_time)
        """
        if debug:
            print(f"\n{'=' * 80}")
            print(f"查询参数: d={d_target:.2f}m, h={h_target:.2f}m, v0={self.v0_list[i_v]:.2f}m/s")
            print(f"(正好在网格点上,无需速度插值)")
            print(f"误差阈值: {error_threshold * 1000:.1f}mm")
            print(f"{'=' * 80}\n")

        errors = self._compute_errors_for_v0(i_v, d_target, h_target)

        # 收集候选信息
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

        # 筛选有效候选
        valid_candidates = [c for c in candidates if c['error'] < error_threshold]

        if debug:
            print(f"{'序号':<6} {'角度(°)':<10} {'误差(mm)':<12} {'飞行时间(s)':<15} {'有效?':<8}")
            print("-" * 80)

            for cand in candidates:
                is_valid = cand['error'] < error_threshold

                if is_valid:
                    print(f"{cand['i_theta']:<6} {cand['theta']:<10.2f} "
                          f"{cand['error'] * 1000:<12.2f} {cand['flight_time']:<15.4f} "
                          f"{'✓':<8}")

            print(f"\n共找到 {len(valid_candidates)} 个有效候选解\n")

        # 选择飞行时间最短的有效解
        if valid_candidates:
            best = min(valid_candidates, key=lambda x: x['flight_time'])
            best_idx = best['i_theta']
        else:
            # 放宽阈值
            relaxed_candidates = [c for c in candidates if c['error'] < error_threshold * 2]

            if relaxed_candidates:
                best = min(relaxed_candidates, key=lambda x: x['flight_time'])
                best_idx = best['i_theta']
            else:
                best = min(candidates, key=lambda x: x['error'])
                best_idx = best['i_theta']

                if debug:
                    print(f"⚠ 警告: 无有效解, 返回误差最小解\n")

        best_theta = self.theta_list[best_idx]
        best_error = errors[best_idx]

        # (可选) 角度精化
        if use_theta_refine:
            theta_before = best_theta
            best_theta = self._refine_theta(best_idx, errors)
            if debug:
                print(f"角度精化: {theta_before:.3f}° -> {best_theta:.3f}°\n")

        flight_time = self.trajectories[f'{i_v:04d}_{best_idx:04d}']['flight_time']

        if debug:
            print(f"{'=' * 80}")
            print(f"最优解: θ={best_theta:.2f}°, 误差={best_error * 1000:.2f}mm, "
                  f"飞行时间={flight_time:.4f}s")
            print(f"{'=' * 80}\n")

        return best_theta, best_error, flight_time

    def get_time_at_distance(self, theta_deg: float, v0: float,
                             d_query: float) -> float:
        """
        获取指定仰角和速度下，到达某水平距离时的飞行时间

        Args:
            theta_deg: 仰角 (度)
            v0: 初速度 (m/s)
            d_query: 查询的水平距离 (m)

        Returns:
            flight_time: 飞行时间 (s), 如果超出范围返回None
        """
        # 找v0索引
        i_v = int(np.round((v0 - self.v0_min) / self.dv0))
        i_v = np.clip(i_v, 0, self.n_v0 - 1)

        # 找θ索引
        i_theta = int(np.argmin(np.abs(self.theta_list - theta_deg)))

        key = f'{i_v:04d}_{i_theta:04d}'
        traj_data = self.trajectories[key]

        points = traj_data['points']
        times = traj_data['times']

        distances = points[:, 0]

        # 检查范围
        if d_query < distances[0] or d_query > distances[-1]:
            return None

        # 二分查找
        idx = np.searchsorted(distances, d_query)

        if idx == 0:
            return times[0]
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