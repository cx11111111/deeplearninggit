import numpy as np
data = np.load(r'E:\personal_reserch\Unet_Radar_wpf\data\power_matrices.npy')
print(data.shape)
print(data.max())
print(data.min())
import numpy as np

# 假设数据存储在 data 变量中

# 检查是否有 NaN 值
has_nan = np.isnan(data).any()

# 找到前10个最大值的索引
top_10_indices = np.argsort(data, axis=None)[-10:]

# 通过索引找到对应的最大值
top_10_values = data.ravel()[top_10_indices]

# 获取这些最大值的多维索引
top_10_indices_multi_dim = np.unravel_index(top_10_indices, data.shape)

print("前十个最大值为:", top_10_values)
print("前十个最大值的位置为:", top_10_indices_multi_dim)
print("数组中是否存在 NaN 值:", has_nan)

import numpy as np

# 假设数据存储在 data 变量中

# 检查是否有 NaN 值
has_nan = np.isnan(data).any()

# 找到前10个最大值的索引
top_10_indices_max = np.argsort(data, axis=None)[-10:]

# 找到前10个最小值的索引
top_10_indices_min = np.argsort(data, axis=None)[:10]

# 通过索引找到对应的最大值
top_10_values_max = data.ravel()[top_10_indices_max]

# 通过索引找到对应的最小值
top_10_values_min = data.ravel()[top_10_indices_min]

# 获取这些最大值和最小值的多维索引
top_10_indices_multi_dim_max = np.unravel_index(top_10_indices_max, data.shape)
top_10_indices_multi_dim_min = np.unravel_index(top_10_indices_min, data.shape)

print("前十个最大值为:", top_10_values_max)
print("前十个最大值的位置为:", top_10_indices_multi_dim_max)
print("前十个最小值为:", top_10_values_min)
print("前十个最小值的位置为:", top_10_indices_multi_dim_min)
print("数组中是否存在 NaN 值:", has_nan)
