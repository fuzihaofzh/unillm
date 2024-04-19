import os
import re
import yaml
from pathlib import Path
from threading import Thread

# Define a function to retrieve API keys
def get_api_key(model_type):
    """
    Retrieve the API key for a given model type from a YAML configuration file.

    Args:
        model_type (str): The type of the model for which the API key is requested.

    Returns:
        str: The API key for the requested model type.
    """
    config_path = Path.home() / '.unillm.yaml'
    api_key_template = {
        'chatgpt': 'YOUR_API_KEY_HERE',
        'claude': 'YOUR_API_KEY_HERE',
        'mistralai': 'YOUR_API_KEY_HERE',
    }

    if not config_path.exists():
        with config_path.open('w') as file:
            yaml.dump(api_key_template, file, default_flow_style=False)
            print(f"Please save your API key into {config_path}")

    with config_path.open('r') as file:
        config = yaml.safe_load(file)

    return config.get(model_type, os.getenv(f"{model_type.upper()}_API_KEY", 'YOUR_API_KEY_HERE'))


class UniLLMBase:
    """
    Base class for all UniLLM models providing a template for generating responses.
    """
    def generate_response(self, message):
        """
        Abstract method to generate a response to a given message.

        Args:
            message (str): The message to respond to.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

class ChatGPT(UniLLMBase):
    """
    Class representing a ChatGPT model for generating text responses.
    """
    def __init__(self, api_key = None, model_id="gpt-3.5-turbo"):
        """
        Initializes the ChatGPT model with an API key and model name.

        Args:
            api_key (str, optional): The API key for the OpenAI service. If not provided, it's fetched using get_api_key.

            model_name (str, optional): The name of the GPT model to use. Default is `gpt-3.5-turbo`.
        """
        from openai import OpenAI
        api_key = get_api_key("chatgpt") if not api_key else api_key
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_id
    
    def generate_response(self, message):
        """
        Generate a text response using the ChatGPT model.

        Args:
            message (str): The input message to respond to.

        Returns:
            str: The generated response.
        """
        response = self.client.chat.completions.create(
            model=self.model_name, 
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message}
            ]
        )
        return response.choices[0].message.content

class Llama2(UniLLMBase):
    """
    Class representing a Llama model for generating text responses.
    """
    def __init__(self, model_id='meta-llama/Llama-2-7b-chat-hf', peft_path=None, max_new_tokens=50, top_p=0.95, top_k=50, temperature=1.0):
        """
        Initializes the Llama model with specified parameters.

        Args:
            model_id (str): Identifier for the Llama model. Default is `meta-llama/Llama-2-7b-chat-hf`.

            peft_path (str, optional): Path to the PEFT model.

            max_new_tokens (int): Maximum new tokens to generate.

            top_p (float): Nucleus sampling probability.

            top_k (int): Top k filtering.

            temperature (float): Temperature for generation.
        """
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
        """
        Generate a text response using the Llama model.

        Args:
            message (str): The input message to respond to.

        Returns:
            str: The generated response.
        """
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
        """
        Constructs the prompt for the Llama model from the given message and chat history.

        Args:
            message (str): The input message.

            chat_history (list): A list of tuples containing the chat history.

            system_prompt (str): A prompt describing the system's role.

        Returns:
            str: The constructed prompt.
        """
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
    """
    Class representing a Mistral model for generating text responses.
    """
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initializes the Mistral model with a specified model ID.

        Args:
            model_id (str): Identifier for the Mistral model. Default is `mistralai/Mistral-7B-Instruct-v0.2`.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
        self.TextIteratorStreamer = TextIteratorStreamer
        import torch
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        self.chat_history = []

    def generate_response(self, message, max_new_tokens=1024, temperature=0.6, top_p=0.9, top_k=50, repetition_penalty=1.2):
        """
        Generate a text response using the Mistral model.

        Args:
            message (str): The input message to respond to.

            max_new_tokens (int): Maximum new tokens to generate.

            temperature (float): Temperature for generation.

            top_p (float): Nucleus sampling probability.

            top_k (int): Top k filtering.

            repetition_penalty (float): Repetition penalty.

        Returns:
            str: The generated response.
        """
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
    """
    Class representing a Claude model for generating text responses.
    """
    def __init__(self, api_key = None, model_id="claude-3-opus-20240229"):
        """
        Initializes the Claude model with an API key and model ID.

        Args:
            api_key (str, optional): The API key for Claude's services. If not provided, it's fetched using get_api_key.

            model_id (str): Identifier for the Claude model. Default is `claude-3-opus-20240229`
        """
        import anthropic
        api_key = get_api_key("claude") if not api_key else api_key
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_id = model_id
    
    def generate_response(self, message, max_tokens=1000, temperature=0):
        """
        Generate a text response using the Claude model.

        Args:
            message (str): The input message to respond to.

            max_tokens (int): Maximum number of tokens to generate.

            temperature (float): Temperature for generation.

        Returns:
            str: The generated response.
        """
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
    """
    Class representing a MistralAI model for generating text responses via API.
    """
    def __init__(self, api_key=None, model_id="mistral-large-latest"):
        """
        Initializes the MistralAI model with an API key and model identifier.

        Args:
            api_key (str, optional): The API key for MistralAI's services. If not provided, it's fetched using get_api_key.

            model_id (str): The model identifier for MistralAI. Default is `mistral-large-latest`.
        """
        from mistralai.client import MistralClient
        from mistralai.models.chat_completion import ChatMessage
        self.ChatMessage = ChatMessage
        api_key = get_api_key("mistralai") if not api_key else api_key
        self.client = MistralClient(api_key=api_key)
        self.model = model_id
    
    def generate_response(self, message):
        """
        Generate a text response using the MistralAI model.

        Args:
            message (str): The input message to respond to.

        Returns:
            str: The generated response.
        """
        messages = [self.ChatMessage(role="user", content=message)]
        chat_response = self.client.chat(model=self.model, messages=messages)
        return chat_response.choices[0].message.content

class RAG(UniLLMBase):
    """
    Class representing a Retrieval-Augmented Generation (RAG) model for generating text responses.
    """
    def __init__(self, data_path, api_key = None, ragmodel=None):
        """
        Initializes the RAG model with the data path, API key, and model identifier.

        Args:
            data_path (str): The path to the data for the RAG model.

            api_key (str, optional): The API key for the service. If not provided, it's fetched using get_api_key.

            ragmodel (str, optional): Identifier for a specific RAG model. e.g. `meta-llama/Llama-2-7b-chat-hf`. If ragmodel is not set, it will use chatgpt by default.
        """
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
        from llama_index.llms.huggingface import HuggingFaceLLM
        from llama_index.core import Settings
        api_key = get_api_key("chatgpt") if not api_key else api_key
        os.environ["OPENAI_API_KEY"] = api_key
        documents = SimpleDirectoryReader(data_path).load_data()
        self.rag_index = VectorStoreIndex.from_documents(documents)
        
        if ragmodel:
            # ragmodel should be sting like "meta-llama/Llama-2-7b-chat-hf"
            # https://docs.llamaindex.ai/en/stable/examples/customization/llms/SimpleIndexDemo-Huggingface_stablelm.html
            # pip install llama-index-llms-huggingface
            
            from llama_index.core import PromptTemplate

            system_prompt = """<|SYSTEM|># Llama Index Tuned 
            - You are helpful assistant
            """

            # This will wrap the default prompts that are internal to llama-index
            query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

            llm = HuggingFaceLLM(
                context_window=4096,
                max_new_tokens=256,
                generate_kwargs={"temperature": 0.7, "do_sample": False},
                system_prompt=system_prompt,
                query_wrapper_prompt=query_wrapper_prompt,
                tokenizer_name=ragmodel,
                model_name=ragmodel,
                device_map="auto",
                stopping_ids=[50278, 50279, 50277, 1, 0],
                tokenizer_kwargs={"max_length": 4096},
                # uncomment this if using CUDA to reduce memory usage
                # model_kwargs={"torch_dtype": torch.float16}
            )

            Settings.llm = llm
            Settings.chunk_size = 1024

            index = VectorStoreIndex.from_documents(documents)

            self.rag_query_engine = index.as_query_engine()
        else:
            self.rag_query_engine = self.rag_index.as_query_engine()
    
    def generate_response(self, message):
        """
        Generate a text response using the RAG model.

        Args:
            message (str): The input message to respond to.

        Returns:
            str: The generated response.
        """
        return self.rag_query_engine.query(message)
    
class Llama3(UniLLMBase):
    """
    Class representing the Llama3 model for generating text responses.
    """
    def __init__(self, model_id="meta-llama/Meta-Llama-3-8B-Instruct", device="cuda", max_new_tokens=256, temperature=0.6, top_p=0.9):
        """
        Initializes the Llama3 model with specified parameters.

        Args:
            model_id (str): Identifier for the Llama model. Default is `meta-llama/Meta-Llama-3-8B-Instruct`.
            device (str): Device to run the model on, default is 'cuda'.
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Temperature parameter for sampling.
            top_p (float): Top p parameter for nucleus sampling.
            torch_dtype (torch.dtype): Torch data type for the model, default is bfloat16.
        """
        import transformers
        import torch
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device
        )
        self.tokenizer = self.pipeline.tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def generate_response(self, user_input):
        """
        Generate a text response using the Llama3 model.

        Args:
            user_input (str): User's input text to be incorporated into the message template.

        Returns:
            str: The generated text output.
        """
        messages = [
            {"role": "system", "content": "You are a helpful AI. Try to answer all questions as much as you know."},
            {"role": "user", "content": user_input},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return outputs[0]["generated_text"][len(prompt):]

class CommandRPlus(UniLLMBase):
    """
    Class representing the CommandRPlus model for generating text responses.
    """
    def __init__(self, model_id="CohereForAI/c4ai-command-r-plus-4bit"):
        """
        Initializes the CommandRPlus model with specified parameters.

        Args:
            model_id (str): Identifier for the model. Default is "CohereForAI/c4ai-command-r-plus-4bit".
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)

    def generate_response(self, user_input):
        """
        Generate a text response using the CommandRPlus model.

        Args:
            user_input (str): User's input text to be incorporated into the message template.

        Returns:
            str: The generated text output, cleaned of any system tokens or command prompts.
        """
        messages = [{"role": "user", "content": user_input}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )

        gen_tokens = self.model.generate(
            input_ids, 
            max_new_tokens=100, 
            do_sample=True, 
            temperature=0.3,
        )

        gen_text = self.tokenizer.decode(gen_tokens[0])
        # Remove everything before the last <BOS_TOKEN> to clean up the response
        clean_text = re.split(r"<[^>]*>", gen_text)[-2]
        return clean_text

class UniLLM:
    """
    A factory class to initialize and interact with various UniLLM models.
    """
    def __init__(self, model_type, **kwargs):
        """
        Initializes the specified UniLLM model type with the given arguments.

        Args:
            model_type (str): The type of model to initialize.

            **kwargs: Additional keyword arguments to pass to the model's constructor.
        """
        self.model = self.initialize_model(model_type, **kwargs)

    def initialize_model(self, model_type, **kwargs):
        """
        Initializes a specific model based on the model type.

        Args:
            model_type (str): The type of model to initialize.
            **kwargs: Additional keyword arguments for model initialization.

        Returns:
            UniLLMBase: An instance of the specified model type.

        Raises:
            ValueError: If the specified model type is unsupported.
        """
        clss = get_all_subclasses(UniLLMBase)
        model_mapping = {c.__name__.split(".")[-1] : c for c in clss}
        if model_type in model_mapping:
            return model_mapping[model_type](**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def generate_response(self, message):
        """
        Generates a response using the initialized model.

        Args:
            message (str): The message to generate a response for.

        Returns:
            str: The generated response.
        """
        return self.model.generate_response(message)
    
def get_all_subclasses(cls):
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))
    return all_subclasses

def cmd(model_type=None, **kwargs):
    """
    Command line interface to interact with various UniLLM models.

    Args:
        model_type (str, optional): The type of model to interact with. If not specified, the user will be prompted to choose one.
        
        **kwargs: Additional keyword arguments to pass to the model's constructor.
    """

    clss = get_all_subclasses(UniLLMBase)
    model_options = {str(i + 1) : c.__name__.split(".")[-1] for i, c in enumerate(clss)}

    if not model_type or model_type not in model_options.values():
        print("Available models:")
        for num, name in model_options.items():
            print(f"{num}: {name}")
        model_choice = input("Please choose a model by number (default is 1): ")
        model_type = model_options.get(model_choice, 'ChatGPT')


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
    """
    Initializes and runs the command line interface using the Fire library.
    """
    import fire
    fire.Fire(cmd)

if __name__ == "__main__":
    run_cmd()