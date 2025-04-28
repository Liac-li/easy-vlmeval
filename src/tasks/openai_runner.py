
from utils.config import load_config

from openai import OpenAI


class OpenAIRunner:
    def __init__(self, model_name: str):
        model_config = load_config()['models']['config'][model_name]
        api_key = model_config['api_key']
        api_base = model_config['api_base_url']

        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model_name = model_name

    def generate(self, prompt: str):

        response = self.client.chat.completions.create(model='gpt-4o-mini', messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content