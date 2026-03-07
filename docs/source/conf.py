# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

project = 'nd2py'
copyright = '2026, YuMeow'
author = 'YuMeow'
release = '2.4.0'

autodoc_mock_imports = [
    'torch', 
    'torch_geometric', 
    'networkx',
    'numpy', 
    'pandas', 
    'scipy', 
    'sklearn',
    'matplotlib', 
    'seaborn', 
    'tqdm', 
    'pyyaml',
    'rich',
    'dotenv',
    'requests',
    'pyperclip'
]

extensions = [
    'sphinx.ext.autodoc',      # 自动从 docstring 生成文档
    'sphinx.ext.napoleon',     # 支持 Google/NumPy 风格的 docstring
    'sphinx.ext.viewcode',     # 在文档中添加“查看源代码”链接
    'myst_parser',             # 支持 Markdown (.md) 文件
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = []
language = 'en'
suppress_warnings = ['ref.python', 'ref.term', 'ref.ref']
html_theme = 'sphinx_book_theme'
html_title = "nd2py (Neural Discovery of Network Dynamics)"
html_theme_options = {
    "repository_url": "https://github.com/yuzhTHU/nd2py",
    "use_repository_button": True,
    "show_toc_level": 2,
}
autodoc_default_options = {
    'members': True,                # 抓取类里所有的方法
    'member-order': 'bysource',     # 按照代码里的顺序排列，而不是字母表
    'special-members': '__init__',  # 把 __init__ 的注释也抓取出来
    'undoc-members': True,          # 即使没写 docstring 的函数/变量也列出来
    'exclude-members': '__weakref__'
}