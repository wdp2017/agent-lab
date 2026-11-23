import os
from jsonschema import validate, ValidationError
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError("没有读取到 DEEPSEEK_API_KEY，请检查根目录 .env")

print("依赖导入成功，密钥已读取")

import json
from jsonschema import validate, ValidationError

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)
MAX_STEPS = 3
schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "minLength": 1},
        "content": {"type": "string", "minLength": 1},
        "type": {"const": "text"},
        "continue": {"type": "boolean"},
    },
    "required": ["title", "content", "type", "continue"],
    "additionalProperties": False,
}

messages = [
    {
        "role": "system",
        "content": (
            "你是板书内容生成器。只输出 JSON 对象，"
            "包含 title、content、type、continue 四个字段。"
            "title 是简短标题，content 用中文简洁解释知识点，"
            "type 固定为字符串 text。"
            "如果还需要下一张卡片补充，continue 为 true，否则为 false。"
            "不要给出通用的字数与 token 换算比例，说明需要使用对应模型的分词器测量。"
        ),
    },
    {"role": "user", "content": "解释 Function Calling"},
]

cards = []
for step in range(MAX_STEPS):
    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=messages,
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}},
    )
    raw = response.choices[0].message.content
    print("模型原始输出：", raw)
    try:
        data = json.loads(raw)
        validate(instance=data, schema=schema)
    except json.JSONDecodeError as exc:
        print("JSON 解析失败：", exc)
        break
    except ValidationError as exc:
        print("卡片字段校验失败：", exc.message)
        break

    cards.append(data)
    print(f"第 {step + 1} 张：", data["title"], "continue=", data["continue"])
    if not data["continue"]:
        break
    messages.append({"role": "assistant", "content": raw})
    messages.append({
        "role": "user",
        "content": "如果还需要补充，再生成下一张卡片；否则把 continue 设为 false。不要重复已经讲过的内容。",
    })

print(f"共生成 {len(cards)} 张卡片")

output_path = Path(__file__).resolve().parent / "board_cards.json"
output_path.write_text(
    json.dumps(cards, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("卡片已保存：", output_path)

