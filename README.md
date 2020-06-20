# 2048-api

基于在线学习、分层学习训练卷积神经网络控制2048小游戏

# Code structure
* [`game.py`](game.py): the core 2048 `Game` class.
* [`agents.py`](agents.py): the `Agent` class with instances.
* [`displays.py`](displays.py): the `Display` class with instances, to show the `Game` state.
* [`expectimax/`](game2048/expectimax): a powerful ExpectiMax agent by [here](https://github.com/nneonneo/2048-ai).

* [`static/`](static/): frontend assets (based on Vue.js) for web app.
* [`webapp.py`](webapp.py): run the web app (backend) demo.
* [`evaluate.py`](evaluate.py): evaluate your self-defined agent.
* [`train_my_model.py`](train_my_model.py): 边生成训练集边训练模型（需要在GPU环境下运行）
* [`train_my_model_dataSet.py`](train_my_model_dataSet.py):  使用我在上海交通大学学生创新中心资源池**已生成的训练集**训练模型（需要在GPU环境下运行）
* [`tools_for_model.py`](tools_for_model.py): 训练模型、预测结果需要使用到的工具函数
* [`my_model.py`](my_model.py): 定义了本项目使用到的神经网络
* [`picture_for_project/`](picture_for_project/): 包含了我的模型框架图以及运行1000次的成绩分布图

# Requirements
* 同时提供了Windows与Linux下编译生成的expectimax，请选择合适的expectimax运行代码（需要在代码中expectimax的导入部分进行调整）其中Windows下运行的expectimax，在\expectimax \bin文件夹中的可执行文件为2048.exe
* 本项目模型在pytorch-gpu 1.3.1版本下完成训练
* 本项目在云服务器（Linux）和本地（Windows）下均有运行，但可能需要进行一定代码调整

# To compile the pre-defined ExpectiMax agent

```bash
cd game2048/expectimax
bash configure
make
```

在Windows下同样可以实现，但需要安装Git Bash以及MinGW

# To run the evaluate.py

```bash
python evaluate.py >> EE228_evaluate.log
```
注意需要在pytorch-gpu、有GPU环境下运行；若无GPU，则需要将代码的PATH和score部分修改为

```python
PATH = './2048_CPU.pth'
score = single_run(GAME_SIZE, SCORE_TO_WIN,
                   AgentClass=TestAgent, model = model)
```
# LICENSE
The code is under Apache-2.0 License.

# For EE369 / EE228 students from SJTU
Please read course project [requirements](EE369.md) and [description](https://docs.qq.com/slide/DS05hVGVFY1BuRVp5). 

