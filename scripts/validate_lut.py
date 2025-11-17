import sys

sys.path.insert(0, '.')

import numpy as np
import time
from src.lut.query import TrajectoryLUT


def test_query(num_tests: int = 20):
    """测试查表查询"""
    print(f"\n运行{num_tests}次查询测试...\n")

    lut = TrajectoryLUT('../data/trajectory_lut_full.h5')

    errors = []
    flight_times = []

    for i in range(num_tests):
        # 每次循环生成一个标量
        d_target = float(np.random.uniform(1, 3))
        h_target = float(np.random.uniform(-0.1, 0.2))
        v0_query = float(np.random.uniform(23, 25))  # 使用你的v0范围

        theta_pred_deg, error, flight_time = lut.query(d_target, h_target, v0_query, error_threshold=0.1, use_theta_refine=True, debug=False)

        errors.append(error)
        flight_times.append(flight_time)

        print(f"测试 {i + 1:2d}: d={d_target:6.3f}m, h={h_target:5.3f}m, v₀={v0_query:6.3f} m/s "
              f"=> θ={theta_pred_deg:6.3f}°, 误差={error * 1000:6.1f}mm, 飞行时间={flight_time*1000:.1f}ms")

    errors = np.array(errors) * 1000  # 转为mm
    flight_times = np.array(flight_times)

    print(f"\n统计结果:")
    print(f"  平均误差: {np.mean(errors):.1f} mm")
    print(f"  最大误差: {np.max(errors):.1f} mm")
    print(f"  最小误差: {np.min(errors):.1f} mm")
    print(f"  标准差: {np.std(errors):.1f} mm")

    print(f"\n飞行时间统计:")
    print(f"  平均: {np.mean(flight_times):.3f}s")
    print(f"  最小: {np.min(flight_times):.3f}s")
    print(f"  最大: {np.max(flight_times):.3f}s")

    lut.close()


def test_performance(num_queries: int = 1000):
    """性能测试"""
    print(f"\n性能测试:{num_queries}次查询...\n")

    lut = TrajectoryLUT('../data/trajectory_lut.h5')

    d_targets = np.random.uniform(5, 25, num_queries)
    h_targets = np.random.uniform(-1, 5, num_queries)
    v0_queries = np.random.uniform(23, 25, num_queries)

    start_time = time.time()
    for d_t, h_t, v0 in zip(d_targets, h_targets, v0_queries):
        _ = lut.query(d_t, h_t, v0)

    elapsed = time.time() - start_time
    avg_time_ms = (elapsed / num_queries) * 1000
    frequency_hz = num_queries / elapsed

    print(f"✓ 总耗时: {elapsed:.2f} s")
    print(f"✓ 平均查询时间: {avg_time_ms:.3f} ms")
    print(f"✓ 查询频率: {frequency_hz:.1f} Hz")
    print(f"✓ 满足250Hz要求: {'✓ 是' if frequency_hz > 250 else '✗ 否'}")

    lut.close()


if __name__ == '__main__':
    test_query(20)
    test_performance(1000)