import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import rcParams

data=pd.read_csv(r'C:\Users\22279\Desktop\大论文数据\CNNASSA-Informer\真实值与预测值turb_4.csv')
true=data['Real']
pred=data['Predict']
y_lower_pred95=data['y_lower_pred95']
y_upper_pred95=data['y_upper_pred95']
y_lower_pred85=data['y_lower_pred85']
y_upper_pred85=data['y_upper_pred85']
y_lower_pred75=data['y_lower_pred75']
y_upper_pred75=data['y_upper_pred75']



# 计算 PICP
def calculate_picp(y_true, lower_bounds, upper_bounds):
    coverage = (y_true >= lower_bounds) & (y_true <= upper_bounds)
    return np.mean(coverage)

# 计算 PINAW
def calculate_pinaw(y_true, lower_bounds, upper_bounds):
    interval_width = upper_bounds - lower_bounds
    y_range = np.max(y_true) - np.min(y_true)
    return np.mean(interval_width) / y_range

# 计算 CWC
def calculate_cwc(y_true, lower_bounds, upper_bounds, alpha=0.95):
    picp = calculate_picp(y_true, lower_bounds, upper_bounds)
    pinaw = calculate_pinaw(y_true, lower_bounds, upper_bounds)
    c = (np.exp(-10 * (picp - alpha)) if picp < alpha else 0) + 1
    return pinaw * c

# 运行计算
picp_95 = calculate_picp(pred, y_lower_pred95, y_upper_pred95)
pinaw_95 = calculate_pinaw(pred, y_lower_pred95, y_upper_pred95)
cwc_95 = calculate_cwc(pred, y_lower_pred95, y_upper_pred95)
picp_75 = calculate_picp(pred, y_lower_pred75, y_upper_pred75)
pinaw_75 = calculate_pinaw(pred, y_lower_pred75, y_upper_pred75)
cwc_75 = calculate_cwc(pred, y_lower_pred75, y_upper_pred75)
picp_85 = calculate_picp(pred, y_lower_pred85, y_upper_pred85)
pinaw_85 = calculate_pinaw(pred, y_lower_pred85, y_upper_pred85)
cwc_85 = calculate_cwc(pred, y_lower_pred85, y_upper_pred85)


picp_95median = calculate_picp(true, y_lower_pred95, y_upper_pred95)
pinaw_95median = calculate_pinaw(true, y_lower_pred95, y_upper_pred95)
cwc_95median = calculate_cwc(true, y_lower_pred95, y_upper_pred95)
picp_75median = calculate_picp(true, y_lower_pred75, y_upper_pred75)
pinaw_75median = calculate_pinaw(true, y_lower_pred75, y_upper_pred75)
cwc_75median = calculate_cwc(true, y_lower_pred75, y_upper_pred75)
picp_85median = calculate_picp(true, y_lower_pred85, y_upper_pred85)
pinaw_85median = calculate_pinaw(true, y_lower_pred85, y_upper_pred85)
cwc_85median = calculate_cwc(true, y_lower_pred85, y_upper_pred85)


print("95% 75% 和85%置信区间的PICP分别是:", picp_95,picp_75,picp_85)
print("95% 75% 和85%置信区间的PINAW分别是:", pinaw_95,pinaw_75,pinaw_85)
print("95% 75% 和85%置信区间的CWC分别是：", cwc_95,cwc_75,cwc_85)

print("95% 75% 和85%置信区间的PICP分别是:", picp_95median,picp_75median,picp_85median)
print("95% 75% 和85%置信区间的PINAW分别是:", pinaw_95median,pinaw_75median,pinaw_85median)
print("95% 75% 和85%置信区间的CWC分别是：", cwc_95median,cwc_75median,cwc_85median)

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(12,8))
#plt.plot(range(200), true[100:300], color="red", label="真实值", alpha=0.6)
plt.plot(range(288),pred[1100:1388],color="blue", label="真实值", linewidth=2)
plt.fill_between(range(288), y_lower_pred95[1100:1388].squeeze(), y_upper_pred95[1100:1388].squeeze(), color="green", alpha=0.2, label="95%预测区间")
plt.fill_between(range(288), y_lower_pred85[1100:1388].squeeze(), y_upper_pred85[1100:1388].squeeze(), color="green", alpha=0.5, label="85%预测区间")
plt.fill_between(range(288), y_lower_pred75[1100:1388].squeeze(), y_upper_pred75[1100:1388].squeeze(), color="green", alpha=0.7, label="75%预测区间")
#plt.fill_between(range(200), y_lower_pred25[100:300].squeeze(), y_upper_pred25[100:300].squeeze(), color="green", alpha=0.7, label="25%预测区间")

plt.xlabel('样本点', fontsize=15)
plt.ylabel('发电功率（kW）', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(0,288)
#plt.ylim(-499,2200)
plt.legend(fontsize=13)
#plt.title("Prediction Interval with Quantile Regression")
plt.savefig(r'C:\Users\22279\Desktop\2.png')
plt.show()
