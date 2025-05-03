from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:3465/v1",
    #api_key="",
    api_key="3857skg389dh",
)

completion = client.chat.completions.create(
  model="./QwQ-32B",
  messages=[
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)
