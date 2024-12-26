import openai
import os
from dotenv import load_dotenv

def prompt_openai(prompt):
    load_dotenv()
    openai_key = os.getenv('API_KEY')
    openai.api_key = openai_key

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt}
        ]
    )
    ans = response.choices[0].message['content']
    return ans

if __name__ == '__main__':
    print(prompt_openai('hi'))
