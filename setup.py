#!/usr/bin/env python
from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
from unillm.version import __version__

setup(
    name='unillm',
    version=__version__,
    description='Unified Large Language Model Interface for ChatGPT, LLaMA, Mistral, Claude, and RAG',
    url='https://github.com/fuzihaofzh/unillm',
    author='Your Name',
    author_email='your.email@example.com',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3.8',
    ],
    keywords='language models AI NLP ChatGPT Llama Mistral Claude MistralAI RAG',
    packages=find_packages(),
    install_requires=[
        "openai",
        "torch",
        "transformers",
        "yaml",
        "peft",  
        "anthropic",
        "mistralai",  
        "llama_index",
        "fire",
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'unillm=unillm.unillm:run_cmd',  
        ],
    },
)
