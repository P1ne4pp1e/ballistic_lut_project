import h5py
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm


class TrajectoryLUTGenerator:
    """查表生成器"""

    def __init__(self, physics_config, lut_config):
        self.physics = physics_config
        self.lut = lut_config
        from src.integrator.rk4 import GPUTrajectoryIntegrator
        self.integrator = GPUTrajectoryIntegrator(physics_config, lut_config)

    def generate(self, save_path: str = 'data/trajectory_lut.h5') -> None:
        """
        生成完整的查表

        Args:
            save_path: 保存路径
        """
        print(f"\n生成 {self.lut.total_trajectories:,} 条轨迹...")
        start_time = time.time()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(save_path, 'w') as f:
            # 元数据
            f.attrs['v0_min'] = self.lut.v0_min
            f.attrs['v0_max'] = self.lut.v0_max
            f.attrs['dv0'] = self.lut.dv0
            f.attrs['theta_min'] = self.lut.theta_min
            f.attrs['theta_max'] = self.lut.theta_max
            f.attrs['dtheta'] = self.lut.dtheta
            f.attrs['n_v0'] = self.lut.n_v0
            f.attrs['n_theta'] = self.lut.n_theta

            # 参数列表
            f.create_dataset('v0_list', data=self.lut.v0_list)
            f.create_dataset('theta_list', data=self.lut.theta_list)
            f.create_dataset('theta_rad_list', data=self.lut.theta_rad_list)

            # 轨迹组
            traj_group = f.create_group('trajectories')

            # 生成轨迹
            pbar = tqdm(total=self.lut.total_trajectories, desc="积分进度")

            for i_v, v0 in enumerate(self.lut.v0_list):
                for i_theta, theta_rad in enumerate(self.lut.theta_rad_list):
                    # 积分轨迹（现在直接返回飞行时间）
                    points, times, landing_d, landing_h, flight_time = \
                        self.integrator.integrate_trajectory(v0, theta_rad)

                    key = f'{i_v:04d}_{i_theta:04d}'
                    grp = traj_group.create_group(key)

                    # 存储轨迹点和时间
                    grp.create_dataset('points', data=points, compression='gzip')
                    grp.create_dataset('times', data=times, compression='gzip')

                    # 属性
                    grp.attrs['v0'] = v0
                    grp.attrs['theta_deg'] = np.degrees(theta_rad)
                    grp.attrs['landing_d'] = landing_d
                    grp.attrs['landing_h'] = landing_h
                    grp.attrs['flight_time'] = flight_time

                    pbar.update(1)

            pbar.close()

        elapsed = time.time() - start_time
        file_size = Path(save_path).stat().st_size / 1024 / 1024

        print(f"\n✓ 查表生成完成！")
        print(f"  耗时: {elapsed:.1f}s")
        print(f"  保存到: {save_path}")
        print(f"  文件大小: {file_size:.1f} MB")