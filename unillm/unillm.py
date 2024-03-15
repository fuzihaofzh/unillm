import os
import yaml
from pathlib import Path
from threading import Thread

# Define a function to retrieve API keys
def get_api_key(model_type):
    config_path = Path.home() / '.unillm.yaml'
    api_key_template = {
        'chatgpt': 'YOUR_API_KEY_HERE',
        'llama': 'YOUR_API_KEY_HERE',
        'mistral': 'YOUR_API_KEY_HERE',
        'claude': 'YOUR_API_KEY_HERE',
        'mistralai': 'YOUR_API_KEY_HERE',
        'rag': 'YOUR_API_KEY_HERE'
    }

    if not config_path.exists():
        with config_path.open('w') as file:
            yaml.dump(api_key_template, file, default_flow_style=False)
            print(f"Please save your API key into {config_path}")

    with config_path.open('r') as file:
        config = yaml.safe_load(file)

    return config.get(model_type, os.getenv(f"{model_type.upper()}_API_KEY", 'YOUR_API_KEY_HERE'))


class UniLLMBase:
    def generate_response(self, message):
        raise NotImplementedError("This method should be implemented by subclasses.")

class ChatGPT(UniLLMBase):
    def __init__(self, api_key = None, model_name="gpt-3.5-turbo"):
        from openai import OpenAI
        api_key = get_api_key("chatgpt") if not api_key else api_key
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
    
    def generate_response(self, message):
        response = self.client.chat.completions.create(
            model=self.model_name, 
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message}
            ]
        )
        return response.choices[0].message.content

class Llama(UniLLMBase):
    def __init__(self, model_id='meta-llama/Llama-2-7b-chat-hf', peft_path=None, max_new_tokens=50, top_p=0.95, top_k=50, temperature=1.0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
        from peft import PeftModel, PeftConfig
        self.TextIteratorStreamer = TextIteratorStreamer
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda" if torch.cuda.is_available() else None)
        self.system_prompt = "You are a helpful assistant."
        self.chat_history = []

        if peft_path is not None:
            try:
                self.model = PeftModel.from_pretrained(self.model, peft_path)
                print(f"Loaded PEFT model from {peft_path}")
            except Exception as e:
                print(f"Failed to load PEFT model from {peft_path}: {e}")

    def generate_response(self, message):
        prompt = self.get_prompt(message)
        inputs = self.tokenizer([prompt], return_tensors='pt', add_special_tokens=False).to('cuda')

        streamer = self.TextIteratorStreamer(self.tokenizer,
                                        timeout=10.,
                                        skip_prompt=True,
                                        skip_special_tokens=True)

        generate_kwargs = {
            'input_ids': inputs['input_ids'],
            'streamer': streamer,
            'max_new_tokens': self.max_new_tokens,
            'do_sample': True,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'temperature': self.temperature,
            'num_beams': 1,
        }

        thread = Thread(target=self.model.generate, kwargs=generate_kwargs)
        thread.start()

        outputs = []
        for text in streamer:
            outputs.append(text)
        thread.join()  # Ensure the generation thread has completed
        return ''.join(outputs)

    def get_prompt(self, message, chat_history = [], system_prompt = ""):
        texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
        # The first user input is _not_ stripped
        do_strip = False
        for user_input, response in chat_history:
            user_input = user_input.strip() if do_strip else user_input
            do_strip = True
            texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
        message = message.strip() if do_strip else message
        texts.append(f'{message} [/INST]')
        return ''.join(texts)


class Mistral(UniLLMBase):
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.2"):
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
        self.TextIteratorStreamer = TextIteratorStreamer
        import torch
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        self.chat_history = []

    def generate_response(self, message, max_new_tokens=1024, temperature=0.6, top_p=0.9, top_k=50, repetition_penalty=1.2):
        print("in generate_response")
        MAX_INPUT_TOKEN_LENGTH = 1024  # Define a maximum token length
        # Prepare conversation history for input
        conversation = []
        for user, assistant in self.chat_history:
            conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
        conversation.append({"role": "user", "content": message})

        # Process input for the model
        input_ids = self.tokenizer.apply_chat_template(conversation, return_tensors="pt")
        if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
            input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
            print(f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens.")
        input_ids = input_ids.to(self.model.device)

        # Set up text streaming for output generation
        streamer = self.TextIteratorStreamer(self.tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_beams=1,
            repetition_penalty=repetition_penalty,
        )

        # Start a separate thread for text generation
        thread = Thread(target=self.model.generate, kwargs=generate_kwargs)
        thread.start()

        # Collect and yield generated text fragments
        outputs = []
        for text in streamer:
            outputs.append(text)
        thread.join()
        "".join(outputs)

        # Update chat history
        output = "".join(outputs)
        self.chat_history.append((message, output))
        return output


class Claude(UniLLMBase):
    def __init__(self, api_key = None, model_id="claude-3-opus-20240229"):
        import anthropic
        api_key = get_api_key("claude") if not api_key else api_key
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_id = model_id
    
    def generate_response(self, message, max_tokens=1000, temperature=0):
        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": message
                        }
                    ]
                }
            ]
        )
        return response.content[0].text


class MistralAI(UniLLMBase):
    def __init__(self, api_key=None, model="mistral-large-latest"):
        from mistralai.client import MistralClient
        from mistralai.models.chat_completion import ChatMessage
        self.ChatMessage = ChatMessage
        api_key = get_api_key("mistralai") if not api_key else api_key
        self.client = MistralClient(api_key=api_key)
        self.model = model
    
    def generate_response(self, message):
        messages = [self.ChatMessage(role="user", content=message)]
        chat_response = self.client.chat(model=self.model, messages=messages)
        return chat_response.choices[0].message.content

class RAG(UniLLMBase):
    def __init__(self, data_path, api_key = None, ragmodel=None):
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
        from llama_index.legacy.llms.ollama import Ollama
        api_key = get_api_key("chatgpt") if not api_key else api_key
        os.environ["OPENAI_API_KEY"] = api_key
        documents = SimpleDirectoryReader(data_path).load_data()
        self.rag_index = VectorStoreIndex.from_documents(documents)
        
        if ragmodel:
            self.rag_query_engine = self.rag_index.as_query_engine(llm=Ollama(model=ragmodel, request_timeout=60.0))
        else:
            self.rag_query_engine = self.rag_index.as_query_engine()
    
    def generate_response(self, message):
        return self.rag_query_engine.query(message)

class UniLLM:
    def __init__(self, model_type, **kwargs):
        self.model = self.initialize_model(model_type, **kwargs)

    def initialize_model(self, model_type, **kwargs):
        model_mapping = {
            'chatgpt': ChatGPT,
            'llama': Llama,
            'mistral': Mistral,
            'claude': Claude,
            'mistralai': MistralAI,
            'rag': RAG
        }
        if model_type in model_mapping:
            return model_mapping[model_type](**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def generate_response(self, message):
        return self.model.generate_response(message)

def cmd(model_type=None, **kwargs):
    model_options = {
        '1': 'chatgpt',
        '2': 'llama',
        '3': 'mistral',
        '4': 'claude',
        '5': 'mistralai',
        '6': 'rag'
    }

    if not model_type or model_type not in model_options.values():
        print("Available models:")
        for num, name in model_options.items():
            print(f"{num}: {name}")
        model_choice = input("Please choose a model by number (default is 1): ")
        model_type = model_options.get(model_choice, 'chatgpt')


    bot = UniLLM(model_type=model_type, **kwargs)
    chat_history = []

    while True:
        query = input("👨Please Ask a Question: ")
        if query.lower() == 'exit':
            break
        response = bot.generate_response(query)
        print(f"🤖 ({model_type}): {response}")
        chat_history.append((query, response))

def run_cmd():
    import fire
    fire.Fire(cmd)

if __name__ == "__main__":
    run_cmd()