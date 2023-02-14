from setuptools import setup, find_packages

setup(
  name = 'musiclm-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.24',
  license='MIT',
  description = 'MusicLM - AudioLM + Audio CLIP to text to music synthesis',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/musiclm-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'text to music',
    'contrastive learning'
  ],
  install_requires=[
    'accelerate',
    'audiolm-pytorch>=0.10.4',
    'beartype',
    'einops>=0.6',
    'vector-quantize-pytorch>=1.0.0',
    'x-clip',
    'torch>=1.12',
    'torchaudio'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
