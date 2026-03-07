import os
import sys

# 1. 将项目根目录加入系统路径，这样 Sphinx 才能 import 你的 nd2py 模块并读取 docstring
sys.path.insert(0, os.path.abspath('../../'))


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'nd2py'
copyright = '2024, YuMeow'
author = 'YuMeow'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',       # 核心：从 docstring 自动生成文档
    'sphinx.ext.napoleon',      # 支持解析 Google 和 NumPy 风格的 docstring (强烈推荐)
    'sphinx.ext.viewcode',      # 在文档中添加指向源代码的链接
    'sphinx.ext.autosummary',   # 自动生成摘要表格
    'myst_parser',              # 解析 Markdown
    'sphinx_autodoc_typehints', # 更好地处理类型提示 (可选)
]

# 3. 配置 MyST Parser 以支持 Markdown
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# 4. 设置主题为 sphinx-book-theme
html_theme = 'sphinx_book_theme'

# 主题的一些定制化配置 (可选)
html_theme_options = {
    "repository_url": "https://github.com/your-username/nd2py",
    "use_repository_button": True,
    "show_toc_level": 2,
}

# 5. Autodoc 配置：按源代码顺序显示，而不是字母顺序
autodoc_member_order = 'bysource'

html_static_path = ['_static']
