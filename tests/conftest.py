"""
1. Pytest 的插件机制 conftest.py 是 Pytest 的一种特殊配置文件，它会在测试运行时被自动加载。
    通过在该文件中编写钩子函数或配置逻辑，可以影响整个测试会话的行为。
2. 动态修改模块搜索路径 在 conftest.py 中，可以通过操作 sys.path 来动态添加项目源代码目录（如 src）到 Python 的模块搜索路径中。
    这样，在测试文件中就可以直接使用绝对导入方式访问项目模块，而无需手动处理路径问题。
3. 全局生效 conftest.py 的配置对同级目录及其子目录下的所有测试文件均有效，避免了在每个测试文件中重复设置路径的麻烦。
"""

import sys
from pathlib import Path

# 添加 src 目录到 PYTHONPATH
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
