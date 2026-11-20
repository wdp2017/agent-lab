import json
from jsonschema import validate, ValidationError

schema = {
    "type": "object",
    "properties": {
        "temperature": {"type": "number"}
    },
    "required": ["temperature"],
    "additionalProperties": False,
}

content = '{"temperature": "很冷"}'

try:
    data = json.loads(content)
    validate(instance=data, schema=schema)
except json.JSONDecodeError:
    print("解析失败：不是合法 JSON")
except ValidationError as exc:
    print("字段校验失败：", exc.message)
else:
    print("校验通过：", data)