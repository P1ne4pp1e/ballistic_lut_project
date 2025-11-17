import sys
from pathlib import Path

# 切换到项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
os.chdir(project_root)  # 确保工作目录在根目录

from src.physics.config import PhysicsConfig, LUTConfig
from src.lut.generator import TrajectoryLUTGenerator


def main():
    print("=" * 70)
    print("弹道查表生成系统")
    print("=" * 70)
    print(f"工作目录: {Path.cwd()}")

    # 从YAML加载配置
    physics_config = PhysicsConfig.from_yaml('config.yaml')
    lut_config = LUTConfig.from_yaml('config.yaml')

    print(f"\n{physics_config}")
    print(f"\n{lut_config}")

    # 使用绝对路径
    save_path = project_root / 'data' / 'trajectory_lut_full.h5'
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n保存路径: {save_path.absolute()}")

    # 生成查表
    generator = TrajectoryLUTGenerator(physics_config, lut_config)
    generator.generate(save_path=str(save_path))


if __name__ == '__main__':
    main()