# import os
# from openai import OpenAI

# client = OpenAI(
#     # The API keys for the Singapore and Beijing regions are different. To obtain an API key: https://www.alibabacloud.com/help/en/model-studio/get-api-key
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     # api_key="sk-95ba1e2608c642439b05b7b572fee422", 
#     # The following is the base_url for the Singapore region. If you use a model in the Beijing region, replace the base_url with https://dashscope.aliyuncs.com/compatible-mode/v1
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
# )
# completion = client.chat.completions.create(
#     model="qwen-plus", # Model list: https://www.alibabacloud.com/help/en/model-studio/getting-started/models
#     messages=[{"role": "user", "content": "Who are you?"}]
# )
# print(completion.choices[0].message.content)


import os
from openai import OpenAI

try:
    client = OpenAI(
        # The API keys for the Singapore and China (Beijing) regions are different. To obtain an API key, see https://modelstudio.console.alibabacloud.com/?tab=model#/api-key
        # If you have not configured an environment variable, replace the following line with your Model Studio API key: api_key="sk-xxx",
        #api_key=os.getenv("DASHSCOPE_API_KEY"),
        api_key="sk-95ba1e2608c642439b05b7b572fee422",
        # api_key="sk-ad15d36aeaf8492cbd6d7264acff7ca7",
        # The following URL is for the Singapore region. If you use a model in the China (Beijing) region, replace the URL with: https://dashscope.aliyuncs.com/compatible-mode/v1
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
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