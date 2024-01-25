## 软件开发工程师应该了解的AI知识

### 开发使用的AI工具
 1. **GitHub Copilot：**
由 GitHub 和 OpenAI 合作开发的一款人工智能辅助编程工具，它能够集成在 Visual Studio Code、Visual Studio（包括Visual Studio 2019和Visual Studio 2022）、Neovim以及 JetBrains 系列的 IDE 中，为开发者提供自动补全代码、自动生成函数或方法、注释说明等服务。通过学习大量的公开代码库，Copilot 可以理解程序员正在编写的内容，并给出相应的建议，从而提高编程效率。每个月19美金，支持国际信用卡付款。 
 2. **Tabnine：**
是一款基于深度学习算法的代码补全工具，支持VSCode和JetBrains IDEs，它能理解当前编辑的代码上下文，并据此预测出可能的后续代码片段，适用于多种开发语言。每个月15美金，支持支付宝和国际信用卡付款。
 3. **DevChat：**
可以在VSCode中集成的AI助手，帮助开发者根据需求生成或验证代码段，使得AI能够更紧密地参与到实际编程过程中，通过用户自定义的提示词目录和灵活的Prompt模板管理，实现代码编写阶段的智能化辅助。这些插件旨在减少重复劳动，提高程序员的工作效率，同时也有助于降低错误率。可以选择GPT-3.5或者GPT-4模型生成结果。按量付费，支持微信支付。
 4. **IntelliCode：**
由微软推出的VSCode插件，适用于JavaScript和Java语言。它提供了基于AI的智能补全功能，能够根据大量的开源项目数据来预测并推荐最适合的代码片段。免费。必装。
 5. **ESLint：**
虽然不完全是AI工具，但这款插件在配合VSCode使用时能提供动态的自动修复建议，有助于遵循代码规范和优化代码结构。免费。必装。

### 测试使用的AI工具
 1. **Airtest：**
这套工具集通过图像识别技术辅助实现游戏和APP自动化测试，这在一定程度上利用了AI技术。
 2. **UICrawler：**
这类工具能够自动发现UI元素并生成对应的测试脚本，在这一过程中可能会利用到深度学习或机器学习来进行图像识别和处理。
 3. **Perfdog：**
性能测试工具，结合AI算法来分析和预测应用性能瓶颈，提供更为精准的性能测试报告。
 4. **Testim.io：**
运用了AI和机器学习技术来提升UI回归测试的效果，能自动生成和维护测试脚本，并对视觉变化做出智能判断。

### AI工具的不足
 1. GPT很多时候响应得不够快。虽然它已经快到秒出答案，但这对快速输出状态的程序员来说仍然是不够的。你会因为等它而暂时停下思路。
 2. GPT生成代码质量不足。它生成的代码可以满足大部分简单且重复的功能需求，但对于熟练的程序员，可能会额外浪费很多精力来校验它自动生成的代码是否正确。

### AI工具使用建议
 1. GPT能帮助初学者面对不那么熟悉的编程语言或开发框架时，快速学习常用的接口调用方式和简单的实现方案。这意味着我们可以不用为了某些基础问题反复翻找 API 手册，或体验 CSDN 这样的技术博客网站的层层传送门。
 2. GPT可以帮助我们在不熟悉的领域快速上手，只需要一些注释便可快速生成部分业务逻辑，然后进行测试。当然，最终代码的可靠性还是需要开发者人为辨别和控制。
 3. 寥寥几行代码就能实现的功能。这类任务显然是GPT擅长的。

### AI技术的应用前景
AI技术的应用前景非常广阔，正以前所未有的速度渗透到各行各业中，并且随着底层技术的不断突破和进步，未来应用场景将更加丰富多样。
以下是AI应用的部分前景：
 1. **智能制造：** 
AI将在智能工厂、自动化生产线和供应链管理中发挥关键作用，通过预测性维护、质量控制和优化生产流程来提高效率。
 2. **医疗健康：** 
AI将助力精准医疗、疾病诊断与预防、药物研发以及健康管理，通过大数据分析和机器学习改善医疗服务质量和患者预后。
 3. **自动驾驶：** 
在智能交通领域，AI是实现全自动驾驶的核心技术，可以大大减少交通事故并优化交通流量。
 4. **金融服务：** 
利用AI进行风险评估、信贷审批、投资策略制定及反欺诈监测，提供个性化的金融产品和服务。
 5. **教育行业：** 
AI能够个性化教学，智能辅导系统可以根据每个学生的学习进度和能力定制课程内容。
 6. **智能家居：** 
AI结合物联网技术实现家居设备智能化，提升居民生活品质和便利程度。
 7. **环境保护：** 
AI应用于环境监测、污染预测和生态保护等方面，帮助解决气候变化、资源短缺等全球性问题。
 8. **自然语言处理：** 
在客服、新闻生成、语音助手等领域，AI的自然语言处理能力使得人机交互更加自然流畅。

### AI的底层技术原理
 1. **机器学习（Machine Learning）：**
这是AI的核心技术之一，其基本思想是使计算机能够在没有明确编程的情况下从数据中学习规律，并能根据新数据自动调整模型参数以改进性能。
 2. **深度学习（Deep Learning）：**
作为机器学习的一个分支，深度学习主要通过构建多层非线性神经网络模型，在大量数据集上训练实现对复杂模式和特征的高效识别与提取。
 3. **自然语言处理（Natural Language Processing, NLP）：**
该领域技术让计算机理解、解释、生成人类自然语言文本。它包括词法分析、句法分析、语义分析等多个层次的技术，使AI能够与人类进行更有效的沟通。随着硬件计算能力的提升和算法优化，这些底层技术将继续发展和完善，为更多前沿领域的AI应用创造可能性。

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
 3. #### 安装第三方包
``` python
# 本机的python3版本是 3.8
pip3 install -r requirements.txt
```

