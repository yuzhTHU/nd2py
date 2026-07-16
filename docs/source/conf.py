# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import os
import sys
from importlib.metadata import version as package_version
sys.path.insert(0, os.path.abspath('../../'))

project = 'nd2py'
copyright = '2026, YuMeow'
author = 'YuMeow'
release = package_version('nd2py')
version = '.'.join(release.split('.')[:2])

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
    'sphinx.ext.autosummary',  # 从人工选择的公开 API 生成详细页
    'sphinx.ext.napoleon',     # 支持 Google/NumPy 风格的 docstring
    'sphinx.ext.viewcode',     # 在文档中添加“查看源代码”链接
    'sphinx.ext.doctest',
    'myst_parser',             # 支持 Markdown (.md) 文件
]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = [
    '_build', 'build', 'Thumbs.db', '.DS_Store',
    # Legacy sphinx-apidoc wrapper pages. The curated API landing page links
    # directly to the public sections instead.
    'api/*.rst',
    # These files are included as maintainable sections of the long-form
    # Guide and Examples pages, rather than built as separate navigation pages.
    'getting_started/*.md', 'user_guide/*.md',
    'examples/index.md', 'examples/grouped_parameter.md', 'examples/custom_operator.md',
]
language = 'en'
suppress_warnings = ['ref.python', 'ref.term', 'ref.ref']
html_theme = 'pydata_sphinx_theme'
html_title = "nd2py (Neural Discovery of Network Dynamics)"
html_sidebars = {
    # The documentation deliberately uses a few long top-level pages. The
    # theme's default section sidebar only shows children of the active page,
    # which leaves this column empty. Reuse the global navigation here so the
    # three primary destinations remain visible on every page.
    '**': ['navbar-nav.html'],
}
html_theme_options = {
    "navbar_align": "left",
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "secondary_sidebar_items": ["page-toc"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/yuzhTHU/nd2py",
            "icon": "fa-brands fa-github",
        },
    ],
}
autodoc_default_options = {
    'members': True,                # 抓取类里所有的方法
    'member-order': 'bysource',     # 按照代码里的顺序排列，而不是字母表
    'special-members': '__init__',  # 把 __init__ 的注释也抓取出来
    'exclude-members': '__weakref__'
}

html_static_path = ['_static']
html_css_files = ['custom.css']
