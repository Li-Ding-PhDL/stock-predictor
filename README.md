# 基于 MATLAB 机器学习的橡胶混凝土性能预测研究

[![MATLAB](https://img.shields.io/badge/MATLAB-R2020b+-blue?logo=mathworks)](https://www.mathworks.com/products/matlab.html)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 项目简介

本项目利用 **MATLAB** 和 **机器学习技术** 对橡胶混凝土（Rubber Concrete）的力学性能进行预测研究。通过集成多种智能优化算法与机器学习模型，构建高精度的非线性预测系统，为橡胶混凝土的配合比设计和性能评估提供科学依据。

## 🔬 研究背景

橡胶混凝土是一种将废旧橡胶颗粒掺入传统混凝土中形成的新型环保建筑材料。然而，橡胶掺量对混凝土力学性能的影响具有高度非线性特征，传统经验公式难以准确预测。本研究采用机器学习方法，建立数据驱动的性能预测模型。

## 📁 文件结构

```
Research-on-Rubber-Concrete-Using-MATLAB-Machine-Learning/
├── 1_PSO-SVR.m                 # 粒子群优化支持向量回归模型
├── 2_FA_RF.m                   # 萤火虫算法优化随机森林模型
├── 3_PSO-LSBoost.m             # 粒子群优化提升树模型
├── 3_RC_V37_Forward_Expert_Ultimate.m  # 主程序/专家系统模型
├── 4_GA-BP.m                   # 遗传算法优化 BP 神经网络模型
├── 5_PSO_LSSVM.m               # 粒子群优化最小二乘支持向量机模型
├── 6_LSTM.m                    # 长短期记忆网络深度学习模型
├── 数据集 3.xlsx                # 原始实验数据集
├── 3_ConcreteModel_LSBoost.mat # 预训练 LSBoost 模型文件
└── README.md                   # 项目说明文档
```

## 🤖 机器学习模型

本项目实现了 **6 种** 主流机器学习算法及其优化变体：

| 编号 | 模型名称 | 优化算法 | 特点 |
|:---:|---------|---------|------|
| 1 | **PSO-SVR** | 粒子群优化 (PSO) | 支持向量回归 + 自适应参数寻优 |
| 2 | **FA-RF** | 萤火虫算法 (FA) | 随机森林 + 群体智能特征选择 |
| 3 | **PSO-LSBoost** | 粒子群优化 (PSO) | 提升树集成 + 学习率优化 |
| 4 | **GA-BP** | 遗传算法 (GA) | 反向传播神经网络 + 全局最优权重 |
| 5 | **PSO-LSSVM** | 粒子群优化 (PSO) | 最小二乘 SVM + 核参数优化 |
| 6 | **LSTM** | - | 长短期记忆深度学习网络 |

## 🎯 核心功能

- ✅ **多模型对比分析**：集成 6 种机器学习算法，支持性能横向对比
- ✅ **智能超参数优化**：PSO、FA、GA 三种启发式算法自动调参
- ✅ **数据预处理**：Excel 数据导入、归一化、训练集/测试集划分
- ✅ **模型持久化**：支持训练好的模型保存为 `.mat` 文件
- ✅ **可视化输出**：预测结果对比图、误差分析图、收敛曲线

## 📊 数据集说明

| 文件名 | 格式 | 说明 |
|:------:|:----:|:-----|
| `数据集 3.xlsx` | Excel | 橡胶混凝土实验数据（配合比、力学性能等） |

### 典型数据特征
- **输入特征**：水泥用量、水胶比、橡胶掺量、砂率、外加剂等
- **输出目标**：抗压强度、抗折强度、弹性模量等力学指标

## 🚀 快速开始

### 环境要求
- MATLAB R2020b 或更高版本
- MATLAB Statistics and Machine Learning Toolbox
- MATLAB Deep Learning Toolbox（用于 LSTM 模型）

### 运行步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/Li-Ding-PhDL/Research-on-Rubber-Concrete-Using-MATLAB-Machine-Learning.git
   cd Research-on-Rubber-Concrete-Using-MATLAB-Machine-Learning
   ```

2. **准备数据**
   - 确保 `数据集 3.xlsx` 位于项目根目录
   - 检查数据格式与完整性

3. **运行模型**
   ```matlab
   % 示例：运行 PSO-SVR 模型
   run('1_PSO-SVR.m')
   
   % 运行主程序（专家系统）
   run('3_RC_V37_Forward_Expert_Ultimate.m')
   ```

4. **查看结果**
   - 预测性能指标（R²、RMSE、MAE）
   - 可视化对比图表
   - 模型文件（`.mat`）

## 📈 预期输出

每个模型运行后将生成：
- 训练集/测试集预测对比图
- 预测值 vs 实际值散点图
- 相对误差分布图
- 优化算法收敛曲线
- 性能评估指标表格

## 🔍 模型选择建议

| 应用场景 | 推荐模型 | 理由 |
|:--------|:--------|:-----|
| 小样本数据 | PSO-SVR / PSO-LSSVM | 泛化能力强，适合有限数据 |
| 高维特征 | FA-RF | 内置特征重要性评估 |
| 高精度需求 | GA-BP / LSTM | 非线性拟合能力最优 |
| 快速部署 | PSO-LSBoost | 训练速度快，可解释性好 |

## 📝 引用格式

如果您在本研究中使用了本项目的代码或数据，请引用：

```bibtex
@software{Li_RubberConcreteML2026,
  author = {Li, Ding},
  title = {Research on Rubber Concrete Using MATLAB Machine Learning},
  year = {2026},
  url = {https://github.com/Li-Ding-PhDL/Research-on-Rubber-Concrete-Using-MATLAB-Machine-Learning},
  organization = {GitHub}
}
```

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE)

## 👨‍💻 作者信息

- **Li Ding** ()
- GitHub: [@Li-Ding-PhDL](https://github.com/Li-Ding-PhDL)

## 🤝 贡献指南

欢迎通过以下方式参与项目：
1. 提交 Issue 报告问题或提出建议
2. Fork 仓库并提交 Pull Request
3. 分享您的实验结果和改进方案

## 📧 联系方式

如有学术合作或技术咨询需求，请通过 GitHub Issues 联系。

---

**最后更新**: 2026 年 8 月 29 日
