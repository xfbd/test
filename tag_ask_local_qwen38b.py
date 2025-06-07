
import json
import datetime
import requests
from file_db import model, collection

def generate(prompt):
    messages = [{"role": "user", "content": prompt}]
    headers = {
        "Content-Type": "application/json"
     }
    payload = {
        "model": "qwen3:8b",
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.7}
    }
    response = requests.post(
        "http://localhost:11434/api/generate",
        headers=headers,
        json=payload
    )
    response_data = response.json()
    print("Response Data:", response_data)  # 打印响应数据以便调试
    content = ""  # 初始化content变量
    try:
        if "choices" in response_data:
            content = response_data["choices"][0]["message"]["content"]
    except Exception as e:
        print("发生错误:", e)
    return content

def retrieval(query):
    context = " "
    query_embedding = model.encode(query)
    results = collection.query(query_embedding.tolist(),n_results=1)
    text = results["documents"][0]
    for T in text:
        context += T
        context += "\n----------------\n"
       
    return context

def argument(query,context=" "):
    if not context:
       return f"请回答下面的问题：{query}"
    else:
        prompt = f"""请更加上下文信息回答问题，如果上下文信息不明确，请直接回答："给的信息不充分，无法回答！"
        上下文信息：
        {context}
        问题：
        {query}
        """
        return prompt

if __name__ == "__main__":
    qury = "中国的首都是哪里？"
    textes = retrieval(qury)   
    prompt = argument(qury)
    generate_text = generate(qury)
    print(generate_text)
