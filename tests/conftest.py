"""
pytest 会自动找到并加载这个文件。它唯一的作用是:
把项目根目录(agent.py、tools.py 所在的位置)加进 Python 的搜索路径里,
这样 tests/ 子文件夹里的测试文件才能 `import agent`、`import tools`。

这跟 eval/run_eval.py 里那行 sys.path.insert 是同一个道理——
子文件夹里的代码要用到上一层的模块,就得先告诉 Python 去哪里找。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
