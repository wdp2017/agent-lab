import json
from pathlib import Path

cards_path = Path(__file__).resolve().parent / "board_cards.json"
cards = json.loads(cards_path.read_text(encoding="utf-8"))
print("卡片数量：", len(cards))

question = "Rag 的工作流程是什么？"
keywords = ["流程", "工作"]
hits = []
for card in cards:
    text = card["title"] + card["content"]
    if any(word in text for word in keywords):
        hits.append(card)

print("检索到：", len(hits), "张")
for card in hits:
    print("-", card["title"])


from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

context = "\n\n".join(f"{c['title']}\n{c['content']}" for c in hits)
messages = [
    {
        "role": "system",
        "content": "只根据下面提供的资料回答。如果资料里没有，就说不知道，不要编造。\n\n" + context,
    },
    {"role": "user", "content": question},
]
response = client.chat.completions.create(
    model="deepseek-flash",
    messages=messages,
    extra_body={"thinking": {"type": "disabled"}},
)
print("模型回答：", response.choices[0].message.content)