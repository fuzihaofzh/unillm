# UniLLM: Unified Large Language Model Interface

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/unillm.svg)](https://pypi.org/project/unillm/)
[![GitHub stars](https://img.shields.io/github/stars/fuzihaofzh/unillm?style=social)](https://github.com/fuzihaofzh/unillm)
[![Documentation Status](https://readthedocs.org/projects/unillm/badge/?version=latest)](https://unillm.readthedocs.io/en/latest/?badge=latest)

UniLLM is a versatile Python library and command-line tool designed to provide unified access to various large language models such as [ChatGPT](https://openai.com/chatgpt), [Llama2](https://llama.meta.com/), [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), [Claude](https://www.anthropic.com/), [MistralAI](https://mistral.ai/), [RAG](https://www.llamaindex.ai/), [Llama3](https://llama.meta.com/), and [CommandRPlus](https://cohere.ai/). This library simplifies the integration of these models into your projects or allows for direct interaction via the command line.

## Features

- Unified API for interacting with multiple language models.
- Support for both API and local models.
- Extensible framework allowing the addition of more models in the future.
- Command-line tool for easy interaction with models.
- Configuration via YAML file for API keys.

## Installation

Install UniLLM using pip:

```bash
pip install unillm
```

## Configuration

Configure your API keys for the models by creating a `.unillm.yaml` file in your home directory:

```yaml
chatgpt: YOUR_CHATGPT_API_KEY
claude: YOUR_CLAUDE_API_KEY
mistralai: YOUR_MISTRALAI_API_KEY
# Add other model API keys as needed
```

## Supported Models

| Model         | Support API | Support Local |
|---------------|:-----------:|:-------------:|
| ChatGPT       | ✅          |               |
| Llama2        |             | ✅            |
| Mistral       | ✅          | ✅            |
| Claude        | ✅          |               |
| MistralAI     | ✅          |               |
| RAG           | ✅          | ✅            |
| Llama3        |             | ✅            |
| CommandRPlus  |             | ✅            |

## Usage

### As a Python Library

Interact with language models seamlessly in your Python projects:

```python
from unillm import UniLLM

# Initialize Llama with specific settings
model = UniLLM('Llama2', peft_path="path_to_peft_model", max_new_tokens=1024)

# Generate a response
response = model.generate_response("How can AI help humans?")
print(response)
```

### As a Command-Line Tool

Start the CLI by running:

```bash
unillm
```

Follow the prompts to select a model and enter your queries. For example:

```bash
Please choose a model by number (default is 1):
1: ChatGPT
2: Llama2
...

👨Please Ask a Question: What are the latest AI trends?
🤖 (ChatGPT): AI trends include...
```

To exit, type `exit`.

## Contributing

We welcome contributions! If you have suggestions or enhancements, fork the repository, create a feature branch, and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
