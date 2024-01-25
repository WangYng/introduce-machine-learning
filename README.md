## 研发AI共学会

### 必要的运行环境
 1. #### 安装 python 3
```script
brew install python
```
 2. #### 安装 venv python 虚拟环境， 防止第三方包版本冲突
```python
# 安装 venv python 虚拟环境
python3 -m venv ./.venv

# 激活并进入虚拟环境
source ./.venv/bin/activate
```
 3. #### 安装第三包
``` python
# 本机的python3版本是 3.8
pip3 install -r requirements.txt
```

### python 文件介绍
 1. #### train_ann.py
介绍 **神经网络** 的训练和使用，使用 **tensowflow** 框架演示
``` python
python3 train_ann.py
``` 

 2. #### train_cnn.py
介绍 **卷积神经网络** 的训练和使用，使用 **tensowflow** 框架演示
``` python
python3 train_cnn.py
``` 

 3. #### train_rnn.py
介绍 **循环神经网络** 的训练和使用，使用 **tensowflow** 框架演示
``` python
python3 train_rnn.py
``` 

 4. #### train_dnn_transformer.py
介绍 **Transform模型** 的使用，使用 **transformers** 框架演示
 ``` python
python3 train_dnn_transformer.py
``` 
