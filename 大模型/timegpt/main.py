import pandas as pd
from nixtlats import TimeGPT
import matplotlib.pyplot as plt
df=pd.read_csv(r'C:\Users\22279\Desktop\数据集\风机功率数据\turb_1.csv')
df=df[['Day','Tmstamp', 'y']]
df["ds"]=pd.to_datetime(df["Day"].astype(str)+" "+df["Tmstamp"],format="%j %H:%M")
df=df[['ds','y']]
df=df.interpolate(method='linear',limit_direction='both')
plt.plot(df['ds'],df['y'])
plt.plot(title='Patv')
plt.xlabel('ds')
plt.ylabel('Patv')
plt.show()
print(df)

timegpt=TimeGPT(token='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')
timegpt.validate_token()
fcst_df=timegpt.forecast(df,h=24,level=[80,90])