"""生成查表的主脚本"""

import sys

sys.path.insert(0, '.')

from src.physics.config import PhysicsConfig, LUTConfig
from src.lut.generator import TrajectoryLUTGenerator


def main():
    print("=" * 70)
    print("弹道查表生成系统")
    print("=" * 70)

    # 配置
    physics_config = PhysicsConfig()
    lut_config = LUTConfig()

    print(f"\n{physics_config}")
    print(f"\n{lut_config}")

    # 生成查表
    generator = TrajectoryLUTGenerator(physics_config, lut_config)
    generator.generate(save_path='data/trajectory_lut.h5')


if __name__ == '__main__':
    main()