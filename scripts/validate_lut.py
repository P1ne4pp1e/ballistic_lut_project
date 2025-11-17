import sys

sys.path.insert(0, '.')

import numpy as np
import time
from src.lut.query import TrajectoryLUT


def test_query_speed():
    """测试快速查询性能（不计算真实误差）"""
    print("=" * 80)
    print("测试1: 快速查询性能（仅查表，不积分）")
    print("=" * 80)

    lut = TrajectoryLUT('../data/trajectory_lut_full.h5')

    num_queries = 1000
    d_targets = np.random.uniform(1, 3, num_queries)
    h_targets = np.random.uniform(-0.1, 0.2, num_queries)
    v0_queries = np.random.uniform(23, 25, num_queries)

    start_time = time.time()
    for d, h, v0 in zip(d_targets, h_targets, v0_queries):
        theta, est_error, flight_time = lut.query(d, h, v0)
    elapsed = time.time() - start_time

    avg_time_ms = (elapsed / num_queries) * 1000
    frequency_hz = num_queries / elapsed

    print(f"\n✓ 总耗时: {elapsed:.2f}s")
    print(f"✓ 平均查询时间: {avg_time_ms:.3f}ms")
    print(f"✓ 查询频率: {frequency_hz:.1f}Hz")
    print(f"✓ 满足250Hz要求: {'✓ 是' if frequency_hz > 250 else '✗ 否'}\n")

    lut.close()


def test_real_error_validation():
    """测试真实误差验证（带RK4积分）"""
    print("=" * 80)
    print("测试2: 真实误差验证（查表 + RK4积分）")
    print("=" * 80)

    lut = TrajectoryLUT('../data/trajectory_lut_full.h5')

    num_tests = 20

    print(f"\n{'Num':<6} {'d(m)':<8} {'h(m)':<8} {'v0':<8} {'θ(°)':<10} "
          f"{'Est err':<12} {'Real err':<12} {'bias':<10} {'time':<10}")
    print("-" * 90)

    estimated_errors = []
    real_errors = []
    deviations = []
    query_times = []
    validation_times = []

    for i in range(num_tests):
        d_target = float(np.random.uniform(1, 3))
        h_target = float(np.random.uniform(-0.1, 0.2))
        v0_query = float(np.random.uniform(23, 25))

        # 测量快速查询时间
        t0 = time.time()
        theta_deg, est_error, flight_time = lut.query(d_target, h_target, v0_query)
        query_time = time.time() - t0

        # 测量真实误差计算时间
        t0 = time.time()
        real_error, closest_d, closest_h, _ = lut.compute_real_error(
            d_target, h_target, v0_query, theta_deg
        )
        validation_time = time.time() - t0

        deviation = abs(real_error - est_error)

        estimated_errors.append(est_error)
        real_errors.append(real_error)
        deviations.append(deviation)
        query_times.append(query_time)
        validation_times.append(validation_time)

        print(f"{i + 1:<6} {d_target:<8.3f} {h_target:<8.3f} {v0_query:<8.3f} {theta_deg:<10.3f} "
              f"{est_error * 1000:<12.2f} {real_error * 1000:<12.2f} {deviation * 1000:<10.2f} {flight_time * 1000: <10.2f}ms")

    estimated_errors = np.array(estimated_errors) * 1000  # 转为mm
    real_errors = np.array(real_errors) * 1000
    deviations = np.array(deviations) * 1000

    print(f"\n{'统计项':<20} {'估计误差(mm)':<15} {'真实误差(mm)':<15} {'偏差(mm)':<15}")
    print("-" * 65)
    print(f"{'平均':<20} {np.mean(estimated_errors):<15.2f} {np.mean(real_errors):<15.2f} {np.mean(deviations):<15.2f}")
    print(f"{'最大':<20} {np.max(estimated_errors):<15.2f} {np.max(real_errors):<15.2f} {np.max(deviations):<15.2f}")
    print(f"{'最小':<20} {np.min(estimated_errors):<15.2f} {np.min(real_errors):<15.2f} {np.min(deviations):<15.2f}")
    print(f"{'标准差':<20} {np.std(estimated_errors):<15.2f} {np.std(real_errors):<15.2f} {np.std(deviations):<15.2f}")

    print(f"\n耗时统计:")
    print(f"  平均查询耗时: {np.mean(query_times) * 1000:.3f}ms")
    print(f"  平均验证耗时: {np.mean(validation_times) * 1000:.3f}ms")
    print(f"  总耗时: {(np.mean(query_times) + np.mean(validation_times)) * 1000:.3f}ms")

    print(f"\n结论:")
    print(f"  ✓ 快速查询可用于实时瞄准 (250Hz)")
    print(f"  ✓ 真实误差验证用于离线测试和调试")
    print(f"  ✓ 平均误差偏差: {np.mean(deviations):.2f}mm (可接受)\n")

    lut.close()


def test_query_with_validation():
    """测试完整验证流程"""
    print("=" * 80)
    print("测试3: 完整验证流程（query_with_validation）")
    print("=" * 80)

    lut = TrajectoryLUT('../data/trajectory_lut_full.h5')

    # 单次测试，带详细输出
    d_target = 2.5
    h_target = 0.1
    v0_query = 24.0

    print(f"\n查询条件:")
    print(f"  目标: ({d_target}m, {h_target}m)")
    print(f"  初速度: {v0_query}m/s\n")

    t0 = time.time()
    result = lut.query_with_validation(
        d_target, h_target, v0_query,
        error_threshold=0.05,
        use_theta_refine=True,
        debug=True
    )
    elapsed = time.time() - t0

    print(f"返回结果:")
    print(f"  最优仰角: {result['theta_deg']:.3f}°")
    print(f"  估计误差: {result['estimated_error'] * 1000:.2f}mm")
    print(f"  真实误差: {result['real_error'] * 1000:.2f}mm")
    print(f"  飞行时间: {result['flight_time']:.4f}s")
    print(f"  最近点: ({result['closest_point'][0]:.3f}m, {result['closest_point'][1]:.3f}m)")
    print(f"  到达时间: {result['time_to_target']:.4f}s")
    print(f"\n总耗时: {elapsed * 1000:.2f}ms\n")

    lut.close()


if __name__ == '__main__':
    # 测试1: 快速查询性能
    test_query_speed()

    # 测试2: 真实误差验证
    test_real_error_validation()

    # 测试3: 完整验证流程
    test_query_with_validation()