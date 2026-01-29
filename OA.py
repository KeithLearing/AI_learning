from openai import OpenAI

# client = OpenAI(api_key="sk-khswmrfaxsnrejplxbvzgwhsnbseomgglydoumrrolxhzeiu",
#                 base_url="https://api.siliconflow.cn/v1")
# response = client.chat.completions.create(
#     # model='Pro/deepseek-ai/DeepSeek-R1',
#     model="Qwen/Qwen2.5-72B-Instruct",
#     messages=[
#         {'role': 'user',
#          'content': "金融大模型怎么开发"}
#     ],
#     stream=True
# )
#
# for chunk in response:
#     if not chunk.choices:
#         continue
#     if chunk.choices[0].delta.content:
#         print(chunk.choices[0].delta.content, end="", flush=True)
#     if chunk.choices[0].delta.reasoning_content:
#         print(chunk.choices[0].delta.reasoning_content, end="", flush=True)

from openai import OpenAI
client = OpenAI(api_key="sk-khswmrfaxsnrejplxbvzgwhsnbseomgglydoumrrolxhzeiu", base_url="https://api.siliconflow.cn/v1")
response = client.chat.completions.create(
    model="Qwen/QVQ-72B-Preview",
    messages=[
        {"role": "system", "content": "你是数据分析专家，用Markdown输出结果"},
        {"role": "user", "content": "写一篇800字的有关金融数据文章"}
    ],
    temperature=0.7,
    max_tokens=1024,
    stream = True
)
for chunk in response:
    if not chunk.choices:
        continue
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
    if chunk.choices[0].delta.reasoning_content:
        print(chunk.choices[0].delta.reasoning_content, end="", flush=True)