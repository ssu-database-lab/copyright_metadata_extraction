import os
from openai import OpenAI

# The API keys for the Singapore and China (Beijing) regions are different.
# To obtain an API key, see:
#   https://modelstudio.console.alibabacloud.com/?tab=model#/api-key
#
# Never hard-code a key here. Set it in the environment (or the project .env):
#   export DASHSCOPE_API_KEY="sk-..."
#
# Base URL by region:
#   Singapore (international): https://dashscope-intl.aliyuncs.com/compatible-mode/v1
#   China (Beijing):           https://dashscope.aliyuncs.com/compatible-mode/v1

try:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY is not set. Export it or add it to the project .env "
            "before running this test."
        )

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        ),
    )

    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'What is the biggest city in the world?'}
            ]
    )
    print(completion)
    print("-"*100)
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"Error message: {e}")
    print("For more information, see https://www.alibabacloud.com/help/en/model-studio/developer-reference/error-code")
