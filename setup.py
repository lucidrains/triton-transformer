from setuptools import setup, find_packages

setup(
  name = 'triton-transformer',
  packages = find_packages(),
  version = '0.0.7',
  license='MIT',
  description = 'Transformer in Triton',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/triton-transformer',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'transformers'
  ],
  install_requires=[
    'einops',
    'torch>=1.6',
    'triton>=1.0.1dev'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
