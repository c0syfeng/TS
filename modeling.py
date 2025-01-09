#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import datetime
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F


# In[107]:


data = pd.read_csv('windelectricity.csv')


# In[108]:


data


# In[109]:


dfhs = data.copy()
dfs = data.copy()


# # 构造数据

# In[110]:


dfhs['OTHS'] = dfhs['OT'].shift(-16)
dfs['OTS'] = dfs['OT'].shift(-288)


# In[111]:


dfhs=dfhs.dropna()
dfs=dfs.dropna()


# In[112]:


dfhs


# # 数据归一化

# In[113]:


def unit1(x):
    x_min = np.min(x,axis=0)
    x_max = np.max(x,axis=0)
    x = (2*x-(x_max-x_min))/(x_max-x_min)
    return x

def unit2(x):
    return np.sin(x)


# In[114]:


def reunit1(x,x_min,x_max):
    return 0.5*(x*(x_max-x_min)+(x_max-x_min))

def reunit2(x):
    return np.arcsin(x)


# In[115]:


col1 = dfhs.iloc[:,1:5].columns.tolist() + dfhs.iloc[:, -5:].columns.tolist()
print(col1)
col2=dfhs.iloc[:,5:9].columns.tolist()
print(col2)


# In[116]:


dfhs[col2] = dfhs[col2].apply(unit2)
dfhs[col1] = dfhs[col1].apply(unit1)


# In[117]:


col3 = dfs.iloc[:,1:5].columns.tolist() + dfs.iloc[:, -5:].columns.tolist()
print(col3)
col4=dfs.iloc[:,5:9].columns.tolist()
print(col4)


# In[118]:


dfs[col4] = dfs[col4].apply(unit2)
dfs[col3] = dfs[col3].apply(unit1)


# In[119]:


dfs


# In[120]:


xhs = dfhs.iloc[:,1:-2]
yhs = dfhs['OTHS']
xs = dfs.iloc[:,1:-2]
ys = dfs['OTS']


# In[121]:


xhs


# In[122]:


xs


# # 特征选择

# In[174]:


#计算特征重要性
lgb_data = lgb.Dataset(xs, label=ys)
params = {
    "objective": "regression",  # 选择适合你问题类型的目标函数
    "metric": "rmse",  # 选择适合你问题类型的评估指标
}

num_round = 500  # 树的数量，可以根据需要调整
model_ = lgb.train(params, lgb_data, num_round)

feature_importance = model_.feature_importance(importance_type="split")


# In[175]:


feature_names = xs.select_dtypes(exclude=['object']).columns  # 特征名称
feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
feature_importance_df=feature_importance_df.sort_values(by="Importance", ascending=False)


# In[176]:


feature_importance_df


# In[183]:


xs = xs.drop(['30m_ws','50m_ws'],axis=1)
xhs = xhs.drop(['30m_ws','50m_ws'],axis=1)


# # Lightgbm模型搭建

# In[123]:


train_size=int(len(dfs)*0.7)
train_xs=xs[:train_size]
test_xs=xs[train_size:]
train_ys=ys[:train_size]
test_ys=ys[train_size:]
my_models = lgb.LGBMRegressor(objective='regression', num_leaves=25, learning_rate=0.0001, n_estimators=900,
                         verbosity=2)
my_models.fit(train_xs, train_ys)
pred_ys = my_models.predict(test_xs)
RMSEs = np.sqrt(mean_squared_error(test_ys,pred_ys))
print("rmse=:",RMSEs)


# In[255]:


train_size=int(len(dfhs)*0.7)
train_xhs=xhs[:train_size]
test_xhs=xhs[train_size:]
train_yhs=yhs[:train_size]
test_yhs=yhs[train_size:]
my_modelhs = lgb.LGBMRegressor(objective='regression', num_leaves=2, learning_rate=0.0009, n_estimators=2750,
                         verbosity=3)
my_modelhs.fit(train_xhs, train_yhs)
pred_yhs = my_modelhs.predict(test_xhs)

RMSEhs = np.sqrt(mean_squared_error(test_yhs,pred_yhs))
print("rmse=:",RMSEhs)


# # LSTM 模型搭建

# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 设定随机种子，以便结果可复现
torch.manual_seed(42)


# In[3]:


df = pd.read_csv('windelectricity.csv')


# In[4]:


df


# In[5]:


col01 = df.iloc[:,1:5].columns.tolist() + df.iloc[:, -5:].columns.tolist()
print(col01)
col02=df.iloc[:,5:9].columns.tolist()
print(col02)


# In[10]:


df[col02] = df[col02].apply(unit2)
df[col01] = df[col01].apply(unit1)


# In[11]:


df0 = df.drop('date',axis=1)


# In[12]:


df0 = df0.reset_index(drop=True)


# In[13]:


df0


# In[14]:


def split_data(data,pred_len,time_step=96):
    dataX=[]
    datay=[]
    for i in range(len(data)-time_step-pred_len+1):
        dataX.append(data[i:i+time_step])
        datay.append(data[i+time_step:i+time_step+pred_len])
    dataX=np.array(dataX)
    datay=np.array(datay)
    return dataX,datay


# In[15]:


X,y = split_data(df0,16)
print(f"dataX.shape:{X.shape},datay.shape:{y.shape}")


# In[16]:


#划分训练集和测试集的函数
def train_test_split(dataX,datay,shuffle=True,percentage=0.8):
    """
    将训练数据X和标签y以numpy.array数组的形式传入
    划分的比例定为训练集:测试集=8:2
    """
    if shuffle:
        random_num=[index for index in range(len(dataX))]
        np.random.shuffle(random_num)
        dataX=dataX[random_num]
        datay=datay[random_num]
    split_num=int(len(dataX)*percentage)
    train_X=dataX[:split_num]
    train_y=datay[:split_num]
    test_X=dataX[split_num:]
    test_y=datay[split_num:]
    return train_X,train_y,test_X,test_y


# In[17]:


train_X,train_y,test_X,test_y=train_test_split(X,y,shuffle=False,percentage=0.8)
print(f"train_X.shape:{train_X.shape},test_X.shape:{test_X.shape}")


# In[18]:


X_train,y_train=train_X,train_y


# In[47]:


# 定义CNN+LSTM模型类
class CNN_LSTM(nn.Module):
    def __init__(self, conv_input,conv_output,input_size, hidden_size, num_layers, output_size):
        super(CNN_LSTM,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.conv_output = conv_output
        self.conv=nn.Conv1d(conv_input,conv_output,1)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x=self.conv(x)
        h0 = torch.zeros(self.num_layers,x.size(0), self.hidden_size) # 初始化隐藏状态h0
        c0 = torch.zeros(self.num_layers,x.size(0), self.hidden_size)  # 初始化记忆状态c0
        #print(f"x.shape:{x.shape},h0.shape:{h0.shape},c0.shape:{c0.shape}")
        out, _ = self.lstm(x, (h0, c0))# LSTM前向传播
        out = self.fc(out) #[b,s,d]
        return out


# In[65]:


test_X1=torch.Tensor(test_X)
test_y1=torch.Tensor(test_y)

# 定义输入、隐藏状态和输出维度
input_size = 12  # 输入特征维度
conv_input=96
conv_output=16
hidden_size = 64  # LSTM隐藏状态维度
num_layers = 10  # LSTM层数
output_size = 12  # 输出维度（预测目标维度）

# 创建CNN_LSTM模型实例
model =CNN_LSTM(conv_input,conv_output,input_size, hidden_size, num_layers, output_size)

#训练周期为500次
num_epochs=500
batch_size=64#一次训练的数量
#优化器
optimizer=torch.optim.Adam(model.parameters(),lr=0.05,betas=(0.5,0.999))
#损失函数
criterion=nn.MSELoss()

train_losses=[]
test_losses=[]

print(f"start")

for epoch in range(num_epochs):
    
    random_num=[i for i in range(len(train_X))]
    np.random.shuffle(random_num)
    
    train_X=train_X[random_num]
    train_y=train_y[random_num]
    
    train_X1=torch.Tensor(train_X[:batch_size])
    train_y1=torch.Tensor(train_y[:batch_size])
    
    #训练
    model.train()
    #将梯度清空
    optimizer.zero_grad()
    #将数据放进去训练
    output=model(train_X1)
    #计算每次的损失函数
    train_loss=criterion(output,train_y1)
    #反向传播
    train_loss.backward()
    
    #优化器进行优化(梯度下降,降低误差)
    optimizer.step()
    
    if epoch%10==0:
        model.eval()
        with torch.no_grad():
            output=model(test_X1)
            test_loss=criterion(output,test_y1)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f"epoch:{epoch},train_loss:{np.sqrt(train_loss.detach().numpy())},test_loss:{np.sqrt(test_loss.detach().numpy())}")


# # 短期预测

# In[84]:


X_,y_ = split_data(df0,288)
print(f"dataX.shape:{X_.shape},datay.shape:{y_.shape}")


# In[85]:


train_X_,train_y_,test_X_,test_y_=train_test_split(X_,y_,shuffle=False,percentage=0.8)
print(f"train_X_.shape:{train_X_.shape},test_X_.shape:{test_X_.shape}")


# In[86]:


X_train,y_train=train_X_,train_y_


# In[88]:


train_y1.shape


# In[91]:


test_X1=torch.Tensor(test_X_)
test_y1=torch.Tensor(test_y_)
# 定义输入、隐藏状态和输出维度
input_size = 12  # 输入特征维度
conv_input=96
conv_output=288
hidden_size = 64  # LSTM隐藏状态维度
num_layers = 10  # LSTM层数
output_size = 12  # 输出维度（预测目标维度）

# 创建CNN_LSTM模型实例
model =CNN_LSTM(conv_input,conv_output,input_size, hidden_size, num_layers, output_size)

#训练周期为500次
num_epochs=500
batch_size=64#一次训练的数量
#优化器
optimizer=torch.optim.Adam(model.parameters(),lr=0.05,betas=(0.5,0.999))
#损失函数
criterion=nn.MSELoss()

train_losses=[]
test_losses=[]

print(f"start")

for epoch in range(num_epochs):
    
    random_num=[i for i in range(len(train_X_))]
    np.random.shuffle(random_num)
    
    train_X_=train_X_[random_num]
    train_y_=train_y_[random_num]
    
    train_X1=torch.Tensor(train_X_[:batch_size])
    train_y1=torch.Tensor(train_y_[:batch_size])
    print(train_y1.shape)
    #训练
    model.train()
    #将梯度清空
    optimizer.zero_grad()
    #将数据放进去训练
    output=model(train_X1)
    print(output.shape)
    #计算每次的损失函数
    train_loss=criterion(output,train_y1)
    #反向传播
    train_loss.backward()
    
    #优化器进行优化(梯度下降,降低误差)
    optimizer.step()
    
    if epoch%10==0:
        model.eval()
        with torch.no_grad():
            output=model(test_X1)
            test_loss=criterion(output,test_y1)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f"epoch:{epoch},train_loss:{np.sqrt(train_loss.detach().numpy())},test_loss:{np.sqrt(test_loss.detach().numpy())}")


# In[6]:


import os
os.getcwd()


# In[11]:


true288 = np.load('result\\ITransFormer_pl288\\true.npy')
pred288 = np.load('result\\ITransFormer_pl288\\pred.npy')


# In[10]:


true288.shape


# In[12]:


pred288.shape


# In[13]:


import matplotlib.pyplot as plt

# 假设你已经有了真实值和预测值的列表或者numpy数组
true_values = true288[0,:,-1]
pred_values = pred288[0,:,-1]

# 创建一个新的图像
plt.figure()

# 绘制真实值，我们用蓝色线条表示
plt.plot(true_values, label='True', color='blue')

# 绘制预测值，我们用红色线条表示
plt.plot(pred_values, label='Pred', color='red')

# 添加图例
plt.legend()

# 显示图像
plt.show()


# In[14]:


true16 = np.load('result\\ITransFormer_pl16\\true.npy')
pred16 = np.load('result\\ITransFormer_pl16\\pred.npy')


# In[15]:


# 假设你已经有了真实值和预测值的列表或者numpy数组
true_values = true16[0,:,-1]
pred_values = pred16[0,:,-1]

# 创建一个新的图像
plt.figure()

# 绘制真实值，我们用蓝色线条表示
plt.plot(true_values, label='True', color='blue')

# 绘制预测值，我们用红色线条表示
plt.plot(pred_values, label='Pred', color='red')

# 添加图例
plt.legend()

# 显示图像
plt.show()


# In[16]:


true16.shape


# In[ ]:




