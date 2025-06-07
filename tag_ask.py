import json
import datetime
import requests
from file_db import model, collection

def generate(prompt):
    messages = [{"role": "user", "content": prompt}]
    headers = {
        "Content-Type": "application/json",
        # 确保这里使用的是有效的API密钥
        "Authorization": "Bearer sk-630d6039ef374689a6df062f47520f18"
    }
    payload = {
        "model": "qwen-plus",
        "messages": messages
    }
    response = requests.post(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        headers=headers,
        json=payload
    )
    response_data = response.json()
    try:
    # 由于之前已经检查了'choices'键，这里可以安全访问
        if "choices" in response_data :
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
    qury = "有红灯闪烁，什么情况？"
    textes = retrieval(qury)   
    prompt = argument(qury,textes)
    generate_text = generate(prompt)
    print(generate_text)
    
