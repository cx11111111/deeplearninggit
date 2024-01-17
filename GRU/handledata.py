import pandas as pd

# 加载数据集
dataset = pd.read_csv("C:\\Users\\22279\\Desktop\\数据集\\风机功率数据\\turb_60.csv")

# 去掉缺失值和小于0的值
TARGET_COL = 'Patv'

dataset = dataset[dataset[TARGET_COL] >= 0]
dataset.dropna(inplace=True)


# 保存新的数据集到指定目录
new_directory = 'C:\\Users\\22279\\Desktop\\'
dataset.to_csv(new_directory + 'newturb_60.csv', index=False)