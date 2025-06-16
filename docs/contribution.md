# DbAI-py 项目贡献指南

## 一、贡献流程
1. **Fork & 克隆**  
   Fork项目并克隆到本地：  
   `git clone https://github.com/your-username/DbAI-py.git`

2. **创建虚拟环境**  
   `python -m venv env && source env/bin/activate`

3. **安装依赖**  
   `pip install -r requirements.txt`

4. **创建分支**  
   从`main`分支创建功能分支：  
   `git checkout -b feature/功能名`

5. **开发与测试**  
   遵循代码规范，确保测试通过
   例: 
   `pytest tests/`

6. **提交PR**  
   推送分支并创建Pull Request到`main`分支  


## 二、核心代码规范
### 1. 格式规范
- 4空格缩进
- 运算符两侧留空格（如`a + b`）
- 函数/类定义间空2行

### 2. 命名规范
| 类型       | 规则            | 示例            |
|------------|-----------------|-----------------|
| 模块       | 小写+下划线    | `data_utils.py` |
| 类         | 驼峰命名        | `DialogModel`   |
| 函数       | 小写+下划线    | `preprocess_data` |
| 常量       | 全大写+下划线  | `MAX_LENGTH`    |

### 3. 注释规范
- 行内注释以`# `开头，与代码间隔2空格
- 注释解释"为什么"而非"是什么"


## 三、提交规范
### 1. PR要求
- 标题格式：`类型: 描述`（类型如feat/fix/docs）
- 描述需包含：解决的问题、测试方式、相关Issue链接
- 必须通过代码检查：`black . && isort . && mypy src/`

### 2. Issue规范
- Bug报告需包含：环境信息、复现步骤、错误日志
- 功能建议需说明：应用场景、预期效果
- 标题简明，使用`[Bug]`/`[Feature]`前缀


## 四、其他注意事项
- 贡献代码遵循项目开源许可证, 本项目使用Apache 2.0开源协议
[LICENSE](../LICENSE)
- 优先使用英文交流（Issue/PR标题描述）
- 重大功能开发前请先创建Issue讨论方案

遵循以上规范可高效推进贡献流程，如有疑问请通过Issue与维护者沟通！