from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class BoardState(TypedDict):
    topic: str
    title: str


def make_title(state: BoardState) -> dict:
    return {"title": f"板书：{state['topic']}"}

def route_topic(state: BoardState) -> str:
    if state["topic"].strip():
        return "generate"
    return "missing"


def ask_topic(state: BoardState) -> dict:
    return {"title": "请先提供板书主题"}


graph = StateGraph(BoardState)
graph.add_node("make_title", make_title)
graph.add_node("ask_topic", ask_topic)

graph.add_conditional_edges(
    START,
    route_topic,
    {"generate": "make_title", "missing": "ask_topic"},
)
graph.add_edge("make_title", END)
graph.add_edge("ask_topic", END)

app = graph.compile()
print("有主题：", app.invoke({"topic": "RAG", "title": ""}))
print("无主题：", app.invoke({"topic": "   ", "title": ""}))