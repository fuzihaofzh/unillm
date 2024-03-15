# UniLLM: Unified Large Language Model Interface

<p>
<a href="https://opensource.org/licenses/MIT">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
</a>
<a href="https://pypi.org/project/unillm/">
  <img src="https://img.shields.io/pypi/v/unillm.svg" alt="PyPI">
</a>
<a href="https://github.com/fuzihaofzh/unillm">
  <img src="https://img.shields.io/github/stars/fuzihaofzh/unillm.svg?style=social&label=Star&maxAge=2592000" alt="GitHub stars">
</a>
</p>

UniLLM is a versatile Python library and command-line tool designed to provide unified access to various large language models, including [ChatGPT](https://openai.com/chatgpt), [Llama](https://llama.meta.com/), [Mistral (local)](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), [Claude](https://www.anthropic.com/), [MistralAI (API)](https://mistral.ai/), and [RAG (llamaindex)](https://www.llamaindex.ai/). It simplifies the process of interacting with these models, whether you're integrating them into your Python projects or using them directly via the command line.

## Features

- Unified API for interacting with multiple language models.
- Supports models like ChatGPT, LLaMA, Mistral, Claude, MistralAI, and RAG.
- Easy-to-use command-line interface for quick interactions.
- Extensible framework allowing the addition of more models in the future.

## Installation

To install UniLLM, you can use pip directly:

```bash
pip install unillm
```

## Configuration

Before using UniLLM, you need to configure your API keys for the models you intend to use. Create a `.unillm.yaml` file in your home directory with the following structure:

```yaml
chatgpt: 'YOUR_CHATGPT_API_KEY'
llama: 'YOUR_LLAMA_API_KEY'
claude: 'YOUR_CLAUDE_API_KEY'
mistralai: 'YOUR_MISTRALAI_API_KEY'
```

Replace `YOUR_MODEL_API_KEY` with the actual API keys for the models you plan to use.

## Supported Models

| Model       | Support API| Support Local |
|-------------|:----------:|:-------------:|
| ChatGPT     | ✅         |               |
| Llama       |            | ✅            |
| Mistral     | ✅         | ✅            |
| Claude      | ✅         |               | 
| RAG         | ✅         | ✅            |
| Gemini       | Soon       |               |

## Usage

### As a Python Library

You can use UniLLM in your Python projects to interact with various language models seamlessly.

Example:

```python
from unillm.unillm import UniLLM

# Initialize the model (e.g., Llama with PEFT)
model = UniLLM('llama', peft_path="output/my_lora", max_new_tokens=1024)

# Generate a response
response = model.generate_response("Hello!")
print(response)
```

### As a Command-Line Tool

UniLLM also provides a command-line interface to interact with the supported language models.

To start the CLI, simply run:

```bash
unillm
```


Follow the prompts to choose a model and enter your queries. For example:

```bash
Please choose a model by number (default is 1):
1: chatgpt
2: llama
...

👨Please Ask a Question: How are you?
🤖 (chatgpt): I'm just a virtual assistant, but I'm here to help you!
```

For using Llama with a PEFT model:

```bash
unillm --model_type llama --peft_path "output/my_lora" --max_new_tokens 1024
```

To exit, type `exit`.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please feel free to fork the repository, create a feature branch, and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
