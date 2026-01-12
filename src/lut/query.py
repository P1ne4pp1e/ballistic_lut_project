"""
优化版查询算法 - 三分法找峰 + 二分法找零点 (带有效角度过滤)

核心改进:
1. 预先过滤无效角度(超时返回None的角度)
2. 在有效角度范围内执行三分+二分
3. 线性插值得到精确角度

性能: O(log n) ≈ 10-15次查询
"""

import numpy as np
from typing import Tuple, Optional, List


class TrajectoryLUT:
    """查表查询引擎 - 三分+二分优化版 (带有效角度过滤)"""

    def __init__(self, lut_path: str = '../data/trajectory_lut_full.h5'):
        import h5py
        from tqdm import tqdm

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
        """在轨迹points中，在d=d_target处线性插值h"""
        distances = points[:, 0]
        heights = points[:, 1]

        if d_target < distances[0]:
            raise ValueError(f"d_target={d_target:.3f}m 小于轨迹最小距离")
        if d_target > distances[-1]:
            raise ValueError(f"d_target={d_target:.3f}m 超出射程")

        idx = np.searchsorted(distances, d_target)

        if idx < len(distances) and abs(distances[idx] - d_target) < 1e-6:
            return idx, heights[idx]

        d0, d1 = distances[idx - 1], distances[idx]
        h0, h1 = heights[idx - 1], heights[idx]

        h_interp = h0 + (h1 - h0) * (d_target - d0) / (d1 - d0)
        return idx, h_interp

    def _get_error_at_theta(self, i_v: int, i_theta: int,
                            d_target: float, h_target: float) -> Optional[float]:
        """获取指定角度下的高度误差 (返回None表示超出射程或超时)"""
        key = f'{i_v:04d}_{i_theta:04d}'
        points = self.trajectories[key]['points']

        try:
            _, h_interp = self._linear_interp_h_at_d(points, d_target)
            return h_interp - h_target
        except ValueError:
            return None

    def _get_valid_theta_indices(self, i_v: int, d_target: float, h_target: float) -> List[int]:
        """
        获取有效角度索引列表 (过滤掉返回None的角度)

        Returns:
            valid_indices: 有效角度的索引列表
        """
        valid_indices = []
        for i_theta in range(self.n_theta):
            err = self._get_error_at_theta(i_v, i_theta, d_target, h_target)
            if err is not None:
                valid_indices.append(i_theta)
        return valid_indices

    def _ternary_search_peak(self, i_v: int, d_target: float, h_target: float,
                             valid_indices: List[int]) -> Tuple[int, int]:
        """
        三分法找error最大值 (仅在有效角度范围内)

        Args:
            valid_indices: 有效角度索引列表

        Returns:
            (peak_index_in_valid, iter_count): 峰值在valid_indices中的位置和迭代次数
        """
        if not valid_indices:
            raise ValueError("没有有效角度")

        left, right = 0, len(valid_indices) - 1
        iter_count = 0

        while right - left > 2:
            iter_count += 1

            mid1 = left + (right - left) // 3
            mid2 = right - (right - left) // 3

            i_theta1 = valid_indices[mid1]
            i_theta2 = valid_indices[mid2]

            err1 = self._get_error_at_theta(i_v, i_theta1, d_target, h_target)
            err2 = self._get_error_at_theta(i_v, i_theta2, d_target, h_target)

            if err1 < err2:
                left = mid1
            else:
                right = mid2

        # 在最后3个点中找最大error
        max_err = -np.inf
        peak_idx = left

        for i in range(left, right + 1):
            i_theta = valid_indices[i]
            err = self._get_error_at_theta(i_v, i_theta, d_target, h_target)
            if err > max_err:
                max_err = err
                peak_idx = i

        return peak_idx, iter_count

    def _binary_search_zero(self, i_v: int, d_target: float, h_target: float,
                            valid_indices: List[int], left: int, right: int,
                            debug: bool = False) -> Optional[Tuple[int, int, int]]:
        """
        二分法查找零点 (在有效角度的上升段)

        Args:
            valid_indices: 有效角度索引列表
            left, right: 在valid_indices中的索引范围

        Returns:
            (i_prev, i_curr, iter_count): 零点所在区间(在valid_indices中的索引)和迭代次数
            None: 未找到零点
        """
        i_theta_left = valid_indices[left]
        i_theta_right = valid_indices[right]

        err_left = self._get_error_at_theta(i_v, i_theta_left, d_target, h_target)
        err_right = self._get_error_at_theta(i_v, i_theta_right, d_target, h_target)

        if debug:
            print(f"  left={left}(θ={self.theta_list[i_theta_left]:.1f}°), err={err_left*1000:.1f}mm")
            print(f"  right={right}(θ={self.theta_list[i_theta_right]:.1f}°), err={err_right*1000:.1f}mm")

        # 检查是否有零点
        if err_left * err_right > 0:
            return None

        # 二分查找
        iter_count = 0
        while right - left > 1:
            iter_count += 1
            mid = (left + right) // 2
            i_theta_mid = valid_indices[mid]
            err_mid = self._get_error_at_theta(i_v, i_theta_mid, d_target, h_target)

            if abs(err_mid) < 1e-6:
                return (mid, mid, iter_count)

            if err_mid * err_left < 0:
                right = mid
                err_right = err_mid
            else:
                left = mid
                err_left = err_mid

        return (left, right, iter_count)

    def query(self, d_target: float, h_target: float, v0_query: float,
              theta_precision: float = 0.01, debug: bool = False) -> tuple:
        """
        查询最优仰角 - 三分+二分优化版 (带有效角度过滤)

        算法流程:
        1. 过滤无效角度(超时/超出射程) O(n)
        2. 在有效角度内三分法找峰值 O(log m)
        3. 在左侧上升沿二分找零点 O(log m)
        4. 线性插值精确角度 O(1)

        Args:
            d_target: 目标水平距离 (m)
            h_target: 目标高度 (m)
            v0_query: 当前初速度 (m/s)
            theta_precision: 预留参数
            debug: 是否打印调试信息

        Returns:
            (theta_deg, flight_time, error_mm)
        """
        # Step 1: 定位v0索引
        i_v = int(round((v0_query - self.v0_min) / self.dv0))
        i_v = np.clip(i_v, 0, self.n_v0 - 1)

        if debug:
            print(f"\n{'=' * 80}")
            print(f"查询: d={d_target:.3f}m, h={h_target:.3f}m, v0={v0_query:.1f}m/s")
            print(f"{'=' * 80}")

        # Step 2: 获取有效角度范围
        valid_indices = self._get_valid_theta_indices(i_v, d_target, h_target)

        if not valid_indices:
            raise ValueError(f"目标(d={d_target:.3f}m, h={h_target:.3f}m)无法到达")

        if debug:
            valid_thetas = [self.theta_list[i] for i in valid_indices]
            print(f"有效角度: {len(valid_indices)}个 ({valid_thetas[0]:.1f}°~{valid_thetas[-1]:.1f}°)")

        # Step 3: 在有效角度内三分法找峰值
        peak_idx_in_valid, ternary_iters = self._ternary_search_peak(
            i_v, d_target, h_target, valid_indices
        )

        peak_i_theta = valid_indices[peak_idx_in_valid]

        if debug:
            peak_theta = self.theta_list[peak_i_theta]
            peak_err = self._get_error_at_theta(i_v, peak_i_theta, d_target, h_target)
            print(f"三分法: {ternary_iters}次迭代")
            print(f"  峰值: θ={peak_theta:.1f}°, err={peak_err * 1000:.1f}mm")

        # Step 4: 在左侧上升沿二分找零点
        zero_result = self._binary_search_zero(
            i_v, d_target, h_target, valid_indices, 0, peak_idx_in_valid, debug
        )

        if zero_result is None:
            # 未找到零点,返回峰值点(误差最小)
            if debug:
                print("⚠️  未找到零点,使用峰值点")

            theta_best = self.theta_list[peak_i_theta]
            key_best = f'{i_v:04d}_{peak_i_theta:04d}'
            t_best = self.trajectories[key_best]['flight_time']

            peak_err = self._get_error_at_theta(i_v, peak_i_theta, d_target, h_target)
            error_mm = abs(peak_err) * 1000

            if debug:
                print(f"结果: θ={theta_best:.2f}°, err={error_mm:.1f}mm")
                print(f"{'=' * 80}\n")

            return theta_best, t_best, error_mm

        # Step 5: 线性插值精确零点
        i_prev_in_valid, i_curr_in_valid, binary_iters = zero_result

        i_prev = valid_indices[i_prev_in_valid]
        i_curr = valid_indices[i_curr_in_valid]

        if debug:
            print(f"二分法: {binary_iters}次迭代")

        if i_prev == i_curr:
            # 精确零点
            theta_zero = self.theta_list[i_prev]
            key = f'{i_v:04d}_{i_prev:04d}'
            t_zero = self.trajectories[key]['flight_time']

            if debug:
                print(f"✓ 精确零点: θ={theta_zero:.2f}°")
                print(f"总查询次数: {ternary_iters + binary_iters}")
                print(f"{'=' * 80}\n")

            return theta_zero, t_zero, 0.0

        # 线性插值
        err_prev = self._get_error_at_theta(i_v, i_prev, d_target, h_target)
        err_curr = self._get_error_at_theta(i_v, i_curr, d_target, h_target)

        theta_prev = self.theta_list[i_prev]
        theta_curr = self.theta_list[i_curr]

        theta_zero = theta_prev - err_prev * (theta_curr - theta_prev) / (err_curr - err_prev)

        # 插值飞行时间
        key_prev = f'{i_v:04d}_{i_prev:04d}'
        key_curr = f'{i_v:04d}_{i_curr:04d}'

        points_prev = self.trajectories[key_prev]['points']
        idx_prev, _ = self._linear_interp_h_at_d(points_prev, d_target)
        t_prev = self.trajectories[key_prev]['times'][idx_prev]

        points_curr = self.trajectories[key_curr]['points']
        idx_curr, _ = self._linear_interp_h_at_d(points_curr, d_target)
        t_curr = self.trajectories[key_curr]['times'][idx_curr]

        weight = (theta_zero - theta_prev) / (theta_curr - theta_prev)
        t_zero = t_prev + weight * (t_curr - t_prev)

        if debug:
            print(f"✓ 插值零点: θ={theta_zero:.2f}°, t={t_zero:.4f}s")
            print(f"  区间: [{theta_prev:.1f}°, {theta_curr:.1f}°]")
            print(f"  误差: [{err_prev * 1000:.1f}mm, {err_curr * 1000:.1f}mm]")
            print(f"总查询次数: {ternary_iters + binary_iters}")
            print(f"{'=' * 80}\n")

        return theta_zero, t_zero, 0.0

    def query_batch(self, targets: list, v0_query: float,
                    theta_precision: float = 0.01,
                    debug: bool = False) -> list:
        """批量查询"""
        results = []
        for d_target, h_target in targets:
            result = self.query(d_target, h_target, v0_query, theta_precision, debug)
            results.append(result)
        return results

    def get_trajectory_info(self, v0: float, theta_deg: float) -> dict:
        """获取指定(v0, θ)的轨迹信息"""
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


# ============================================================================
# 性能测试
# ============================================================================

if __name__ == '__main__':
    import time

    print("\n" + "=" * 80)
    print("三分+二分优化算法测试 (带有效角度过滤)")
    print("=" * 80 + "\n")

    lut = TrajectoryLUT('../data/trajectory_lut_full.h5')

    # 测试用例
    test_cases = [
        (2.0, 0.1, 24.0, "近距离(凸曲线)"),
        (5.0, 0.2, 24.0, "中距离(凸曲线)"),
        (8.0, 0.0, 24.0, "远距离(可能单调)"),
        (1.5, -0.1, 24.0, "低目标"),
    ]

    print("单次查询测试:\n")
    for d, h, v0, desc in test_cases:
        print(f"场景: {desc}")
        lut.query(d, h, v0, debug=True)

    # 批量性能测试
    print("\n" + "=" * 80)
    print("批量性能测试 (1000次)")
    print("=" * 80 + "\n")

    num_queries = 1000
    v0_candidates = np.arange(23.0, 25.0, 0.1)
    v0_queries = np.random.choice(v0_candidates, num_queries)
    d_targets = np.random.uniform(1.0, 10.0, num_queries)
    h_targets = np.random.uniform(-0.5, 0.5, num_queries)

    start = time.time()
    for d, h, v0 in zip(d_targets, h_targets, v0_queries):
        lut.query(d, h, v0)
    elapsed = time.time() - start

    avg_time_ms = (elapsed / num_queries) * 1000
    frequency_hz = num_queries / elapsed

    print(f"总耗时: {elapsed:.2f}s")
    print(f"平均耗时: {avg_time_ms:.3f}ms")
    print(f"查询频率: {frequency_hz:.1f}Hz")
    print(f"满足300Hz: {'✓' if frequency_hz > 300 else '✗'}\n")

    lut.close()