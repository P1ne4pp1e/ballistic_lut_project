## 基本使用流程

### Step 1: 环境配置

```bash
# 克隆项目
git clone <repo-url>
cd ballistic_lut_project

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### Step 2: 生成查表

首次运行需要生成查表文件（281,101条轨迹）：

```bash
python scripts/generate_lut.py
```

输出示例：
```
======================================================================
弹道查表生成系统
======================================================================

PhysicsConfig(
  radius=8.5mm, mass=5.0g
  g=9.8m/s², rho=1.2kg/m³
  cd=0.47, k=0.254700s⁻¹
)

LUTConfig(
  v0: 401 points (10~30 m/s, Δ=0.05)
  θ: 701 points (-10~60°, Δ=0.1°)
  Total trajectories: 281,101
)

生成 281,101 条轨迹...
[████████████████████████████] 100%

✓ 查表生成完成！
  耗时: 45.3s
  保存到: data/trajectory_lut.h5
  文件大小: 125.4 MB
```

### Step 3: 验证查表

```bash
python scripts/validate_lut.py
```

输出示例：
```
运行20次查询测试...

测试  1: d=20.34m, h=2.15m, v₀=21.42 m/s => θ=25.1°, 误差= 1.2mm
测试  2: d=15.67m, h=1.83m, v₀=18.95 m/s => θ=18.9°, 误差= 0.8mm
...

统计结果:
  平均误差: 1.1 mm
  最大误差: 4.8 mm
  最小误差: 0.1 mm
  标准差: 1.5 mm

性能测试：1000次查询...

✓ 总耗时: 0.45 s
✓ 平均查询时间: 0.45 ms
✓ 查询频率: 2222.2 Hz
✓ 满足250Hz要求: ✓ 是
```

### Step 4: 在应用中使用

```python
from src.lut.query import TrajectoryLUT

# 初始化（加载查表）
lut = TrajectoryLUT('data/trajectory_lut.h5')

# 实时瞄准循环（250Hz）
while True:
    # 获取目标信息
    target_d, target_h = get_target_position()
    v0 = get_muzzle_velocity()
    
    # 查询最优仰角
    theta_deg, error = lut.query(target_d, target_h, v0)
    
    # 发送到电机控制系统
    send_to_motor(pitch=theta_deg)
    
    time.sleep(1/250)  # 250Hz

# 清理
lut.close()
```

## 常见用法示例

### 示例1: 单次查询

```python
from src.lut.query import TrajectoryLUT

lut = TrajectoryLUT()

# 查询参数
d_target = 20.0  # 20米
h_target = 2.0   # 2米高
v0 = 21.5        # 21.5 m/s

# 查询
theta, error = lut.query(d_target, h_target, v0)

print(f"发射仰角: {theta:.2f}°")
print(f"预测误差: {error*1000:.1f}mm")

lut.close()
```

### 示例2: 批量查询

```python
import numpy as np
from src.lut.query import TrajectoryLUT

lut = TrajectoryLUT()

# 批量查询
targets = [
    (10, 1.0, 15),
    (15, 1.5, 18),
    (20, 2.0, 21),
    (25, 2.5, 24),
]

for d, h, v0 in targets:
    theta, error = lut.query(d, h, v0)
    print(f"目标({d}m, {h}m), v0={v0}m/s => θ={theta:.1f}°, 误差={error*1000:.1f}mm")

lut.close()
```

### 示例3: 性能基准

```python
import time
import numpy as np
from src.lut.query import TrajectoryLUT

lut = TrajectoryLUT()

# 生成随机查询
num_queries = 10000
d_targets = np.random.uniform(5, 25, num_queries)
h_targets = np.random.uniform(-1, 5, num_queries)
v0_queries = np.random.uniform(10, 30, num_queries)

# 计时
start = time.time()
for d, h, v0 in zip(d_targets, h_targets, v0_queries):
    lut.query(d, h, v0)
elapsed = time.time() - start

print(f"查询{num_queries}次耗时: {elapsed:.2f}s")
print(f"平均单次耗时: {elapsed/num_queries*1000:.3f}ms")
print(f"查询频率: {num_queries/elapsed:.1f}Hz")

lut.close()
```

## 集成到ROS系统

```python
import rospy
from src.lut.query import TrajectoryLUT

class BalisticCalculator:
    def __init__(self):
        self.lut = TrajectoryLUT('data/trajectory_lut.h5')
        self.pub = rospy.Publisher('/gun_angle', AimAngle, queue_size=1)
        self.sub = rospy.Subscriber('/target_info', TargetInfo, self.callback)
    
    def callback(self, msg):
        # 解析目标信息
        d = msg.distance
        h = msg.height
        v0 = msg.velocity
        
        # 查询
        theta, error = self.lut.query(d, h, v0)
        
        # 发布
        angle_msg = AimAngle()
        angle_msg.pitch = theta
        self.pub.publish(angle_msg)
    
    def __del__(self):
        self.lut.close()
```

## 调试技巧

### 启用详细日志

编辑 `config.yaml`：
```yaml
logging:
  level: DEBUG
```

### 检查查表完整性

```python
from src.lut.query import TrajectoryLUT

lut = TrajectoryLUT()
print(f"速度点数: {lut.n_v0}")
print(f"角度点数: {lut.n_theta}")
print(f"总轨迹数: {lut.n_v0 * lut.n_theta}")
lut.close()
```

## 性能优化

### 1. 预热缓存

```python
# 第一次查询会较慢（IO），之后会快速
lut = TrajectoryLUT()
lut.query(15, 2, 20)  # 预热
```

### 2. 多进程查询

```python
from multiprocessing import Pool

def query_batch(args):
    lut = TrajectoryLUT()
    d, h, v0 = args
    result = lut.query(d, h, v0)
    lut.close()
    return result

with Pool(4) as p:
    results = p.map(query_batch, targets)
```

## 故障排除

### 问题: "CUDA out of memory"

解决: 编辑 `config.yaml`，改用CPU：
```yaml
system:
  device: cpu
```

### 问题: 查询精度不足

检查:
1. 物理参数是否准确
2. 表分辨率是否足够
3. 是否有其他外力（风等）

### 问题: 查表文件损坏

重新生成:
```bash
rm data/trajectory_lut.h5
python scripts/generate_lut.py
```