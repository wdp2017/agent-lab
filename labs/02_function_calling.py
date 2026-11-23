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


def get_weather(city: str) -> dict:
    return {"city": city, "weather": "晴", "temperature": 5}


client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气，当前返回模拟数据",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"}
            },
            "required": ["city"],
        },
    },
}]

schema = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "weather": {"type": "string"},
        "temperature": {"type": "number"},
        "is_mock": {"const": True},
    },
    "required": ["city", "weather", "temperature", "is_mock"],
    "additionalProperties": False,
}

messages = [
    {"role": "system", "content": "天气工具返回的是模拟数据。回答时必须明确说明是模拟数据，不得称为当前或实时天气。"},
    {"role": "user", "content": "上海天气怎么样？请使用天气工具。"},
]

response = client.chat.completions.create(
    model="deepseek-flash",
    messages=messages,
    tools=tools,
    extra_body={"thinking": {"type": "disabled"}},
)

message = response.choices[0].message
print("模型的调用请求：", message.tool_calls)


if not message.tool_calls:
    raise RuntimeError("模型没有提出工具调用")

messages.append(message)

for tool_call in message.tool_calls:
    if tool_call.function.name != "get_weather":
        raise ValueError("未知工具：" + tool_call.function.name)

    args = json.loads(tool_call.function.arguments)
    result = None
    for attempt in range(2):
        try:
            
            city = json.loads(tool_call.function.arguments)["city"]
            result = get_weather(city)
            print("本地函数结果：", result)
            break
        except Exception as exc:
            result = {"error": str(exc)}
            print(f"第 {attempt + 1} 次失败：", exc)

    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result, ensure_ascii=False),
    })

messages.append({
    "role": "user",
    "content": (
        "请根据工具结果输出 JSON，包含 city、weather、"
        "temperature、is_mock 四个字段。"
        "temperature 是数字，is_mock 是布尔值 true，表示模拟数据。"
    ),
})

print("正在执行 JSON Mode 请求")

final_response = client.chat.completions.create(
    model="deepseek-flash",
    messages=messages,
    response_format={"type": "json_object"},
    extra_body={"thinking": {"type": "disabled"}},
)

content = final_response.choices[0].message.content
print("原始 JSON：", content)
try:
    data = json.loads(content)
    validate(instance=data, schema=schema)
except json.JSONDecodeError as exc:
    print("JSON 解析失败：", exc)
except ValidationError as exc:
    print("字段校验失败：", exc.message)
else:
    if "error" in result:
        print("工具已失败，跳过一致性检查：", result["error"])
    elif data["temperature"] != result["temperature"]:
        print("一致性检查失败：温度与工具结果不一致")
    else:
        print("校验通过：", data)
        
