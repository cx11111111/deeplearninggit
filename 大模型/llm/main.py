import pandas as pd
import matplotlib.pyplot as plt
import openai
from datetime import datetime
import os
import traceback
import json

dataset = pd.read_csv(r"C:\Users\22279\Desktop\数据集\风机功率数据\turb_1.1.csv")
# data1 = data1[['Day','Tmstamp', 'Patv']]
# data1['date'] = data1['Day'].astype(str)+" "+data1['Tmstamp']
# data1['date']=data1['date'].apply(lambda x:datetime.strptime(x,"%d %H:%M").strftime("%Y-%m-%d %H:%M"))
# data1.set_index('date', inplace=True)
# data = pd.Series(data1['Patv'])
dataset=dataset['Patv'].dropna()
dataset.index=pd.RangeIndex(start=0,stop=len(dataset),step=1)

print(type(dataset))
print(dataset.shape)
plt.plot(dataset)
plt.plot(title='Patv')
#plt.show()

lag=3
subseqs=list()
for i in range(len(dataset)-lag):
    input=dataset[i:i+lag]
    output=dataset[i+lag]
    subseqs.append((input,output))
print(len(subseqs))

train_size=int(0.7*len(subseqs))
train_seqs=subseqs[:train_size]
test_seqs=subseqs[train_size:]

train_x=[seq[0] for seq in train_seqs]
train_y=[seq[1] for seq in train_seqs]
test_x=[seq[0] for seq in test_seqs]
test_y=[seq[1] for seq in test_seqs]
print(train_y)
print(train_x)
print(len(test_y))
print(len(test_x))

openai.api_key = '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'
#openai.api_key = os.getenv('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')
horizon = 1


def chat_gpt_forecast(data,  forecast_col="Forecast", model="gpt-3.5-turbo", verbose=False):
    prompt = f""""
    Given a dataset separated by triple backticks, which is wind power data with a sampling frequency of 10 minutes, 
    gives you three ten-minute values to predict the next {horizon} ten minutes.
    Returns an answer in JSON format, which has a key: '{forecast_col}', and a list of values assigned to it.
    Only predictions are returned, not Python code.

    '''{data.to_string()}'''
    """
    if verbose:
        print(prompt)

    # 创建聊天消息列表
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(model=model, messages=messages, temperature=0)

    output = response.choices[0].message["content"]

    try:
        json_object = json.loads(output)
        #df = pd.DataFrame(json_object)
    except:
        df = output
        print(traceback.format_exc())

    return json_object

forecasts=[]
for i in range(len(train_seqs)):
     gpt_forecast = chat_gpt_forecast(train_x[i])
     # gpt_forecast = pd.Series(gpt_forecast.split(','), name='Forecast')
     forecast=gpt_forecast["Forecast"]
     forecasts.append(forecast)
     print("gpt_forecast:",gpt_forecast)
     print("forecast:",forecast)
     print("forecasts:",forecasts)

print(train_y)
print(forecasts)
plt.plot(train_y, label='Real')
plt.plot(forecasts, label='Forecast')
plt.ylabel("Patv")
plt.xlabel("date")
plt.legend()
# y.merge(gpt_forecast, how='outer').plot(x='date', y=['Patv', 'Forecast'])
plt.show()


# def chat_gpt_forecast(data, horizon, time_idx="date", forecast_col="Forecast", model="gpt-3.5-turbo", verbose=False):
#     prompt = f""""
#     Given the dataset delimited by the triple backticks,
#     forecast next {horizon} values of the time series.
#
#     Return the answer in JSON format, containing two keys: '{time_idx}'
#     and '{forecast_col}', and list of values assigned to them.
#     Return only the forecasts, not the Python code.
#
#     '''{data.to_string()}'''
#     """
#     if verbose:
#         print(prompt)
#
#     # 创建聊天消息列表
#     messages = [{"role": "user", "content": prompt}]
#     response = openai.ChatCompletion.create(model=model, messages=messages, temperature=0)
#
#     output = response.choices[0].message["content"]
#
#     try:
#         json_object = json.loads(output)
#         df = pd.DataFrame(json_object)
#         df[time_idx] = df[time_idx].astype(data.index.dtype)
#     except:
#         df = output
#         print(traceback.format_exc())
#
#     return df
#
#
# gpt_forecast = chat_gpt_forecast(y_train, horizon)
# gpt_forecast=pd.Series(gpt_forecast.split(','),name='Forecast')
# y = data.reset_index()
# print(y_test)
# print(gpt_forecast)
# plt.plot(y_test,label='Real')
# plt.plot(gpt_forecast,label='Forecast')
# plt.ylabel("Patv")
# plt.xlabel("date")
# plt.legend()
# #y.merge(gpt_forecast, how='outer').plot(x='date', y=['Patv', 'Forecast'])
# plt.show()