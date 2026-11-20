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

schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "minLength": 1},
        "content": {"type": "string", "minLength": 1},
        "type": {"const": "text"},
    },
    "required": ["title", "content", "type"],
    "additionalProperties": False,
}

response = client.chat.completions.create(
    model="deepseek-flash",
    messages=[
        {
            "role": "system",
            "content": (
                "你是板书内容生成器。只输出 JSON 对象，"
                "包含 title、content、type 三个字段。"
                "title 是简短标题，content 用中文简洁解释知识点，"
                "type 固定为字符串 text。"
                "不要给出通用的字数与 token 换算比例，说明需要使用对应模型的分词器测量。"
            ),
        },
        {"role": "user", "content": "解释 Function Calling"},
    ],
    response_format={"type": "json_object"},
    extra_body={"thinking": {"type": "disabled"}},
)

raw = response.choices[0].message.content
print("模型原始输出：", raw)

try:
    card = json.loads(raw)
    validate(instance=card, schema=schema)
except json.JSONDecodeError as exc:
    print("JSON 解析失败：", exc)
except ValidationError as exc:
    print("卡片字段校验失败：", exc.message)
else:
    output_path = Path(__file__).resolve().parent / "board_card.json"
    output_path.write_text(
        json.dumps(card, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("卡片已保存：", output_path)