import math

from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

data=pd.read_csv(r'C:\Users\22279\Desktop\2.csv')
#data.dropna(inplace=True)
pred=data['llamapreds']
true=data['llamatrues']

plt.figure(figsize=(10,5))
plt.plot(pred,label='preds')
plt.plot(true,label='trues')
plt.legend()

save_path='C:/Users/22279/Desktop/my_plot.png'
plt.savefig(save_path,bbox_inches='tight')

mse=mean_squared_error(pred,true)
rmse=math.sqrt(mse)
mae=mean_absolute_error(pred,true)
r2=r2_score(pred,true)
print(f"MSE:{mse:.5f}    RMSE:{rmse:.5f}   MAE:{mae:.5f}    R2:{r2:.5f}")