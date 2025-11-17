import sys

sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
from src.physics.config import PhysicsConfig, LUTConfig
from src.integrator.rk4 import GPUTrajectoryIntegrator

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_single_trajectory():
    """绘制单条轨迹 - 验证基本物理特性"""
    print("=" * 80)
    print("测试1: 单条轨迹可视化")
    print("=" * 80)

    physics = PhysicsConfig.from_yaml('config.yaml')
    lut_config = LUTConfig.from_yaml('config.yaml')
    integrator = GPUTrajectoryIntegrator(physics, lut_config)

    # 测试参数
    v0 = 24.0  # m/s
    theta_deg = 5.0  # 度
    theta_rad = np.radians(theta_deg)

    print(f"\n测试参数:")
    print(f"  初速度: {v0} m/s")
    print(f"  仰角: {theta_deg}°")
    print(f"  阻力系数k: {physics.k:.6f} s⁻¹")
    print(f"  重力加速度g: {physics.g} m/s²\n")

    # 积分轨迹
    points, times, landing_d, landing_h, flight_time = integrator.integrate_trajectory(v0, theta_rad)

    print(f"轨迹结果:")
    print(f"  落地距离: {landing_d:.3f} m")
    print(f"  落地高度: {landing_h:.3f} m")
    print(f"  飞行时间: {flight_time:.4f} s")
    print(f"  轨迹点数: {len(points)}")

    # 计算理论真空轨迹（对比用）
    t_vacuum = np.linspace(0, flight_time, 100)
    vx0 = v0 * np.cos(theta_rad)
    vy0 = v0 * np.sin(theta_rad)

    d_vacuum = vx0 * t_vacuum
    h_vacuum = vy0 * t_vacuum - 0.5 * physics.g * t_vacuum ** 2

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 子图1: 轨迹对比
    ax = axes[0, 0]
    ax.plot(points[:, 0], points[:, 1], 'b-', linewidth=2, label='RK4 (with drag)')
    ax.plot(d_vacuum, h_vacuum, 'r--', linewidth=1.5, alpha=0.7, label='Vacuum (no drag)')
    ax.scatter([landing_d], [landing_h], c='red', s=100, zorder=5, label='Landing point')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Horizontal distance (m)', fontsize=12)
    ax.set_ylabel('Height (m)', fontsize=12)
    ax.set_title(f'Trajectory: v0={v0}m/s, theta={theta_deg}deg', fontsize=13, fontweight='bold')
    ax.legend()

    # 子图2: 速度随时间变化
    ax = axes[0, 1]
    velocities = np.sqrt(np.diff(points[:, 0]) ** 2 + np.diff(points[:, 1]) ** 2) / np.diff(times)
    ax.plot(times[:-1], velocities, 'b-', linewidth=2)
    ax.axhline(v0, color='r', linestyle='--', alpha=0.5, label='Initial velocity')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Velocity (m/s)', fontsize=12)
    ax.set_title('Velocity vs Time (velocity decay)', fontsize=13, fontweight='bold')
    ax.legend()

    # 子图3: 高度随水平距离
    ax = axes[1, 0]
    ax.plot(points[:, 0], points[:, 1], 'b-', linewidth=2)

    # 标注关键点
    max_h_idx = np.argmax(points[:, 1])
    ax.scatter([points[max_h_idx, 0]], [points[max_h_idx, 1]],
               c='green', s=100, zorder=5, label=f'Max height: {points[max_h_idx, 1]:.3f}m')

    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Horizontal distance (m)', fontsize=12)
    ax.set_ylabel('Height (m)', fontsize=12)
    ax.set_title('Height profile', fontsize=13, fontweight='bold')
    ax.legend()

    # 子图4: 水平速度和竖直速度
    ax = axes[1, 1]
    vd = np.diff(points[:, 0]) / np.diff(times)
    vh = np.diff(points[:, 1]) / np.diff(times)

    ax.plot(times[:-1], vd, 'b-', linewidth=2, label='Horizontal velocity')
    ax.plot(times[:-1], vh, 'r-', linewidth=2, label='Vertical velocity')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Velocity (m/s)', fontsize=12)
    ax.set_title('Velocity components', fontsize=13, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig('trajectory_single.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: trajectory_single.png\n")
    plt.show()


def plot_multiple_angles():
    """不同仰角的轨迹对比"""
    print("=" * 80)
    print("Test 2: Multiple angles comparison")
    print("=" * 80)

    physics = PhysicsConfig.from_yaml('config.yaml')
    lut_config = LUTConfig.from_yaml('config.yaml')
    integrator = GPUTrajectoryIntegrator(physics, lut_config)

    v0 = 24.0
    angles_deg = [5, 10, 15, 20, 25, 30]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(angles_deg)))

    print(f"\nv0 = {v0} m/s\n")
    print(f"{'Angle(deg)':<12} {'Range(m)':<12} {'Max_h(m)':<12} {'Flight_time(s)':<15}")
    print("-" * 60)

    for i, theta_deg in enumerate(angles_deg):
        theta_rad = np.radians(theta_deg)
        points, times, landing_d, landing_h, flight_time = integrator.integrate_trajectory(v0, theta_rad)

        max_h = np.max(points[:, 1])

        print(f"{theta_deg:<12} {landing_d:<12.3f} {max_h:<12.3f} {flight_time:<15.4f}")

        # 轨迹图
        ax1.plot(points[:, 0], points[:, 1], color=colors[i],
                 linewidth=2, label=f'{theta_deg}deg')
        ax1.scatter([landing_d], [landing_h], color=colors[i], s=50, zorder=5)

    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Horizontal distance (m)', fontsize=12)
    ax1.set_ylabel('Height (m)', fontsize=12)
    ax1.set_title(f'Trajectories at different angles (v0={v0}m/s)', fontsize=13, fontweight='bold')
    ax1.legend()

    # 射程-仰角关系
    angles_fine = np.linspace(5, 45, 20)
    ranges = []

    for theta_deg in angles_fine:
        theta_rad = np.radians(theta_deg)
        points, _, landing_d, _, _ = integrator.integrate_trajectory(v0, theta_rad)
        ranges.append(landing_d)

    ax2.plot(angles_fine, ranges, 'b-', linewidth=2, marker='o', markersize=5)

    max_range_idx = np.argmax(ranges)
    max_range = ranges[max_range_idx]
    optimal_angle = angles_fine[max_range_idx]

    ax2.scatter([optimal_angle], [max_range], color='red', s=150, zorder=5,
                label=f'Max range: {max_range:.2f}m @ {optimal_angle:.1f}deg')

    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Launch angle (deg)', fontsize=12)
    ax2.set_ylabel('Range (m)', fontsize=12)
    ax2.set_title('Range vs Launch angle', fontsize=13, fontweight='bold')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('trajectory_angles.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Optimal angle: {optimal_angle:.1f}deg, Max range: {max_range:.2f}m")
    print(f"✓ Saved: trajectory_angles.png\n")
    plt.show()


def plot_velocity_vs_range():
    """不同初速度的射程对比"""
    print("=" * 80)
    print("Test 3: Velocity vs Range")
    print("=" * 80)

    physics = PhysicsConfig.from_yaml('config.yaml')
    lut_config = LUTConfig.from_yaml('config.yaml')
    integrator = GPUTrajectoryIntegrator(physics, lut_config)

    velocities = [20, 22, 24, 26, 28]
    theta_deg = 5  # 固定仰角
    theta_rad = np.radians(theta_deg)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.plasma(np.linspace(0, 1, len(velocities)))

    print(f"\nFixed angle: {theta_deg}deg\n")
    print(f"{'v0(m/s)':<12} {'Range(m)':<12} {'Max_h(m)':<12} {'Flight_time(s)':<15}")
    print("-" * 60)

    ranges = []

    for i, v0 in enumerate(velocities):
        points, times, landing_d, landing_h, flight_time = integrator.integrate_trajectory(v0, theta_rad)

        max_h = np.max(points[:, 1])
        ranges.append(landing_d)

        print(f"{v0:<12} {landing_d:<12.3f} {max_h:<12.3f} {flight_time:<15.4f}")

        ax1.plot(points[:, 0], points[:, 1], color=colors[i],
                 linewidth=2, label=f'v0={v0}m/s')
        ax1.scatter([landing_d], [landing_h], color=colors[i], s=50, zorder=5)

    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Horizontal distance (m)', fontsize=12)
    ax1.set_ylabel('Height (m)', fontsize=12)
    ax1.set_title(f'Trajectories at different velocities (theta={theta_deg}deg)',
                  fontsize=13, fontweight='bold')
    ax1.legend()

    # 速度-射程关系
    ax2.plot(velocities, ranges, 'b-', linewidth=2, marker='o', markersize=8)

    # 理论真空射程（对比）
    ranges_vacuum = [(v0 ** 2 * np.sin(2 * theta_rad)) / physics.g for v0 in velocities]
    ax2.plot(velocities, ranges_vacuum, 'r--', linewidth=2, marker='s',
             markersize=6, alpha=0.7, label='Vacuum (theoretical)')

    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Initial velocity (m/s)', fontsize=12)
    ax2.set_ylabel('Range (m)', fontsize=12)
    ax2.set_title('Range vs Initial velocity', fontsize=13, fontweight='bold')
    ax2.legend(['With drag (RK4)', 'Vacuum (theory)'])

    plt.tight_layout()
    plt.savefig('trajectory_velocities.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: trajectory_velocities.png\n")
    plt.show()


def plot_energy_conservation():
    """能量守恒验证（带空气阻力会损失能量）"""
    print("=" * 80)
    print("Test 4: Energy analysis (should decrease due to drag)")
    print("=" * 80)

    physics = PhysicsConfig.from_yaml('config.yaml')
    lut_config = LUTConfig.from_yaml('config.yaml')
    integrator = GPUTrajectoryIntegrator(physics, lut_config)

    v0 = 24.0
    theta_deg = 5
    theta_rad = np.radians(theta_deg)

    points, times, landing_d, landing_h, flight_time = integrator.integrate_trajectory(v0, theta_rad)

    # 计算能量
    velocities = np.sqrt(np.diff(points[:, 0]) ** 2 + np.diff(points[:, 1]) ** 2) / np.diff(times)
    heights = points[:-1, 1]  # 对应速度的高度

    m = physics.mass

    KE = 0.5 * m * velocities ** 2  # 动能
    PE = m * physics.g * heights  # 势能
    total_energy = KE + PE

    initial_energy = 0.5 * m * v0 ** 2
    energy_ratio = total_energy / initial_energy

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 子图1: 能量随时间变化
    ax = axes[0, 0]
    ax.plot(times[:-1], KE, 'b-', linewidth=2, label='Kinetic Energy')
    ax.plot(times[:-1], PE, 'r-', linewidth=2, label='Potential Energy')
    ax.plot(times[:-1], total_energy, 'k-', linewidth=2, label='Total Energy')
    ax.axhline(initial_energy, color='g', linestyle='--', alpha=0.5, label='Initial Energy')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Energy (J)', fontsize=12)
    ax.set_title('Energy vs Time', fontsize=13, fontweight='bold')
    ax.legend()

    # 子图2: 能量损失比例
    ax = axes[0, 1]
    energy_loss = (initial_energy - total_energy) / initial_energy * 100
    ax.plot(times[:-1], energy_loss, 'r-', linewidth=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Energy loss (%)', fontsize=12)
    ax.set_title('Energy loss due to drag', fontsize=13, fontweight='bold')

    # 子图3: 速度衰减
    ax = axes[1, 0]
    ax.plot(times[:-1], velocities, 'b-', linewidth=2)
    ax.axhline(v0, color='r', linestyle='--', alpha=0.5, label=f'v0={v0}m/s')

    # 理论指数衰减（近似）
    v_theory = v0 * np.exp(-physics.k * times[:-1])
    ax.plot(times[:-1], v_theory, 'g--', linewidth=1.5, alpha=0.7,
            label='Exponential decay (approx)')

    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Velocity (m/s)', fontsize=12)
    ax.set_title('Velocity decay', fontsize=13, fontweight='bold')
    ax.legend()

    # 子图4: 轨迹图
    ax = axes[1, 1]
    ax.plot(points[:, 0], points[:, 1], 'b-', linewidth=2)
    ax.scatter([landing_d], [landing_h], c='red', s=100, zorder=5)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Horizontal distance (m)', fontsize=12)
    ax.set_ylabel('Height (m)', fontsize=12)
    ax.set_title(f'Trajectory (v0={v0}m/s, theta={theta_deg}deg)',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig('trajectory_energy.png', dpi=150, bbox_inches='tight')

    final_energy = total_energy[-1]
    energy_loss_total = (initial_energy - final_energy) / initial_energy * 100

    print(f"\nEnergy analysis:")
    print(f"  Initial energy: {initial_energy:.6f} J")
    print(f"  Final energy: {final_energy:.6f} J")
    print(f"  Energy loss: {energy_loss_total:.2f}%")
    print(f"  Expected behavior: Energy should decrease (drag dissipation)")
    print(f"\n✓ Saved: trajectory_energy.png\n")
    plt.show()


def plot_drag_coefficient_comparison():
    """对比不同阻力系数的影响"""
    print("=" * 80)
    print("Test 5: Drag coefficient comparison")
    print("=" * 80)

    physics = PhysicsConfig.from_yaml('config.yaml')
    lut_config = LUTConfig.from_yaml('config.yaml')

    v0 = 24.0
    theta_deg = 10
    theta_rad = np.radians(theta_deg)

    # 不同阻力系数
    cd_values = [0.0, 0.2, 0.47, 0.8]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.coolwarm(np.linspace(0, 1, len(cd_values)))

    print(f"\nv0={v0}m/s, theta={theta_deg}deg\n")
    print(f"{'Cd':<10} {'k(s^-1)':<12} {'Range(m)':<12} {'Flight_time(s)':<15}")
    print("-" * 60)

    for i, cd in enumerate(cd_values):
        # 创建临时配置
        physics_temp = PhysicsConfig(
            radius=physics.radius,
            mass=physics.mass,
            g=physics.g,
            rho=physics.rho,
            cd=cd
        )

        integrator = GPUTrajectoryIntegrator(physics_temp, lut_config)
        points, times, landing_d, landing_h, flight_time = integrator.integrate_trajectory(v0, theta_rad)

        print(f"{cd:<10.2f} {physics_temp.k:<12.6f} {landing_d:<12.3f} {flight_time:<15.4f}")

        label = f'Cd={cd:.2f}' if cd > 0 else 'No drag (Cd=0)'
        ax1.plot(points[:, 0], points[:, 1], color=colors[i],
                 linewidth=2, label=label)
        ax1.scatter([landing_d], [landing_h], color=colors[i], s=50, zorder=5)

    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Horizontal distance (m)', fontsize=12)
    ax1.set_ylabel('Height (m)', fontsize=12)
    ax1.set_title(f'Effect of drag coefficient (v0={v0}m/s, theta={theta_deg}deg)',
                  fontsize=13, fontweight='bold')
    ax1.legend()

    # Cd vs Range
    cd_fine = np.linspace(0, 1.0, 20)
    ranges = []

    for cd in cd_fine:
        physics_temp = PhysicsConfig(
            radius=physics.radius,
            mass=physics.mass,
            g=physics.g,
            rho=physics.rho,
            cd=cd
        )
        integrator = GPUTrajectoryIntegrator(physics_temp, lut_config)
        points, _, landing_d, _, _ = integrator.integrate_trajectory(v0, theta_rad)
        ranges.append(landing_d)

    ax2.plot(cd_fine, ranges, 'b-', linewidth=2, marker='o', markersize=5)
    ax2.axvline(0.47, color='r', linestyle='--', alpha=0.5, label='Smooth sphere (Cd=0.47)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Drag coefficient Cd', fontsize=12)
    ax2.set_ylabel('Range (m)', fontsize=12)
    ax2.set_title('Range vs Drag coefficient', fontsize=13, fontweight='bold')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('trajectory_drag_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: trajectory_drag_comparison.png\n")
    plt.show()


def main():
    """运行所有测试"""
    print("\n")
    print("*" * 80)
    print(" " * 20 + "RK4 TRAJECTORY VALIDATION")
    print("*" * 80)
    print("\nThis script validates the RK4 integrator by:")
    print("  1. Visualizing single trajectory")
    print("  2. Comparing multiple launch angles")
    print("  3. Testing different initial velocities")
    print("  4. Checking energy conservation (should decrease due to drag)")
    print("  5. Comparing different drag coefficients")
    print("\n" + "*" * 80 + "\n")

    try:
        # Test 1
        plot_single_trajectory()

        # Test 2
        plot_multiple_angles()

        # Test 3
        plot_velocity_vs_range()

        # Test 4
        plot_energy_conservation()

        # Test 5
        plot_drag_coefficient_comparison()

        print("=" * 80)
        print("ALL TESTS COMPLETED!")
        print("=" * 80)
        print("\nGenerated files:")
        print("  ✓ trajectory_single.png")
        print("  ✓ trajectory_angles.png")
        print("  ✓ trajectory_velocities.png")
        print("  ✓ trajectory_energy.png")
        print("  ✓ trajectory_drag_comparison.png")
        print("\nValidation points:")
        print("  ✓ Trajectories should be parabolic")
        print("  ✓ Velocity should decay over time")
        print("  ✓ Energy should decrease (drag dissipation)")
        print("  ✓ Range increases with velocity")
        print("  ✓ Optimal angle should be around 30-35deg (with drag)")
        print("  ✓ Higher Cd = shorter range")
        print()

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()