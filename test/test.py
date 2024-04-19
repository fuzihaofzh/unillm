import unittest
import sys
sys.path.append("unillm/")
from unillm import UniLLM

class TestUniLLM(unittest.TestCase):

    def test_chatgpt(self):
        question = "Who are you?"
        print(f"👨 Testing ChatGPT with the question: '{question}'")
        chatgpt_bot = UniLLM(model_type='ChatGPT', model_name="gpt-3.5-turbo")
        response = chatgpt_bot.generate_response(question)
        print(f"🤖Response: {response}")
        self.assertIsNotNone(response)
        self.assertNotEqual(response, "")

    def test_llama(self):
        question = "Who are you?"
        print(f"👨 Testing Llama2 with the question: '{question}'")
        llama_bot = UniLLM(model_type='Llama2', model_id='meta-llama/Llama-2-7b-chat-hf')
        response = llama_bot.generate_response(question)
        print(f"🤖Response: {response}")
        self.assertIsNotNone(response)
        self.assertNotEqual(response, "")

    def test_mistral(self):
        question = "What's your name?"
        print(f"👨 Testing Mistral with the question: '{question}'")
        mistral_bot = UniLLM(model_type='Mistral')
        response = mistral_bot.generate_response(question)
        print(f"🤖Response: {response}")
        self.assertIsNotNone(response)
        self.assertNotEqual(response, "")

    def test_claude(self):
        question = "What's your name?"
        print(f"👨 Testing Claude with the question: '{question}'")
        claude_bot = UniLLM(model_type='Claude')
        response = claude_bot.generate_response(question)
        print(f"🤖Response: {response}")
        self.assertIsNotNone(response)
        self.assertNotEqual(response, "")

    def test_mistralai(self):
        question = "What's your name?"
        print(f"👨 Testing MistralAI with the question: '{question}'")
        mistralai_bot = UniLLM(model_type='MistralAI')
        response = mistralai_bot.generate_response(question)
        print(f"🤖Response: {response}")
        self.assertIsNotNone(response)
        self.assertNotEqual(response, "")

    def test_rag(self):
        question = "What is the reason of repetition problem?"
        print(f"👨 Testing RAG with the question: '{question}'")
        rag_bot = UniLLM(model_type='RAG', data_path="test/data")
        response = rag_bot.generate_response(question)
        print(f"🤖Response: {response}")
        self.assertIsNotNone(response)
        self.assertNotEqual(response, "")

    def test_llama3(self):
        question = "How can AI help humans?"
        print(f"👨 Testing Llama3 with the question: '{question}'")
        llama3_bot = UniLLM(model_type='Llama3')
        response = llama3_bot.generate_response(question)
        print(f"🤖Response: {response}")
        self.assertIsNotNone(response)
        self.assertNotEqual(response, "")

    def test_commandrplus(self):
        question = "What are the latest AI developments?"
        print(f"👨 Testing CommandRPlus with the question: '{question}'")
        commandrplus_bot = UniLLM(model_type='CommandRPlus')
        response = commandrplus_bot.generate_response(question)
        print(f"🤖Response: {response}")
        self.assertIsNotNone(response)
        self.assertNotEqual(response, "")

if __name__ == '__main__':
    unittest.main()
