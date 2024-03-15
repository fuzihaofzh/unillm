import unittest
import sys
sys.path.append("unillm/")
from unillm import UniLLM

class TestUniLLM(unittest.TestCase):

    def test_chatgpt(self):
        question = "Who are you?"
        print(f"👨 Testing ChatGPT with the question: '{question}'")
        chatgpt_bot = UniLLM(model_type='chatgpt', model_name="gpt-3.5-turbo")
        response = chatgpt_bot.generate_response(question)
        print(f"🤖Response: {response}")
        self.assertIsNotNone(response)
        self.assertNotEqual(response, "")

    def test_llama(self):
        question = "Who are you?"
        print(f"👨 Testing LLAMA with the question: '{question}'")
        llama_bot = UniLLM(model_type='llama', model_id='meta-llama/Llama-2-7b-chat-hf')
        response = llama_bot.generate_response(question)
        print(f"🤖Response: {response}")
        self.assertIsNotNone(response)
        self.assertNotEqual(response, "")

    def test_mistral(self):
        question = "What's your name?"
        print(f"👨 Testing Mistral with the question: '{question}'")
        mistral_bot = UniLLM(model_type='mistral')
        response = mistral_bot.generate_response(question)
        print(f"🤖Response: {response}")
        self.assertIsNotNone(response)
        self.assertNotEqual(response, "")

    def test_claude(self):
        question = "What's your name?"
        print(f"👨 Testing Claude with the question: '{question}'")
        claude_bot = UniLLM(model_type='claude')
        response = claude_bot.generate_response(question)
        print(f"🤖Response: {response}")
        self.assertIsNotNone(response)
        self.assertNotEqual(response, "")

    def test_mistralai(self):
        question = "What's your name?"
        print(f"👨 Testing MistralAI with the question: '{question}'")
        mistralai_bot = UniLLM(model_type='mistralai')
        response = mistralai_bot.generate_response(question)
        print(f"🤖Response: {response}")
        self.assertIsNotNone(response)
        self.assertNotEqual(response, "")

    def test_rag(self):
        question = "What is the reason of repetition problem?"
        print(f"👨 Testing RAG with the question: '{question}'")
        rag_bot = UniLLM(model_type='rag', data_path="test/data")
        response = rag_bot.generate_response(question)
        print(f"🤖Response: {response}")
        self.assertIsNotNone(response)
        self.assertNotEqual(response, "")

if __name__ == '__main__':
    unittest.main()
