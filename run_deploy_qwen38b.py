import requests
import json
from datetime import datetime

class OllamaClient:
    def __init__(self):
        self.base_url = "http://localhost:11434/api/generate"
        self.model = "qwen3:8b"
        self.headers = {"Content-Type": "application/json"}
        self.current_date = datetime.now().strftime('%Y年%m月%d日')
    
    def _requires_code_wrap(self, question: str) -> bool:
        """严格判断是否需要代码包裹"""
        # 条件a：完整项目请求检测
        project_keywords = ["实现", "项目", "程序", "构建", "创建", "开发", "编写"]
        has_project = any(kw in question for kw in project_keywords)
        
        # 条件b：明确HTML/CSS请求
        has_web = any(kw in question.lower() for kw in ["html", "css"])
        
        # 必须同时满足项目级请求特征
        return (has_project and "功能" in question) or has_web
    
    def _format_response(self, question: str, raw_response: str) -> str:
        """严格按规则格式化响应"""
        if not raw_response.strip():
            return "未获取到有效回答"
            
        # 非代码类回答直接返回
        if not self._requires_code_wrap(question):
            return raw_response
            
        # 代码类回答处理
        code_blocks = []
        current_block = []
        in_code = False
        
        for line in raw_response.split('\n'):
            if "" in line and in_code:
                in_code = False
                current_block.append(line)
                code_blocks.append("\n".join(current_block))
                current_block = []
            elif in_code:
                current_block.append(line)
        
        return "\n\n".join(code_blocks) if code_blocks else raw_response
    
    def generate(self, question: str) -> str:
        """生成符合规范的回答"""
        prompt = f"{question}\n当前日期：{self.current_date}"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7}
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                data=json.dumps(payload),
                timeout=50
            )
            
            # 严格处理JSON响应
            try:
                result = response.json()
                if response.status_code == 200:
                    raw_answer = result.get("response", "").strip()
                    return self._format_response(question, raw_answer)
                return f"请求失败: {response.status_code} {response.text}"
            except json.JSONDecodeError:
                return "响应解析错误：返回的不是有效JSON格式"
                
        except requests.exceptions.RequestException as e:
            return f"请求失败: {str(e)}"

def main():
    client = OllamaClient()
    print(f"Ollama客户端已启动（模型: {client.model}）")
    print("输入'退出'或'exit'结束对话\n" + "="*50)
    
    while True:
        try:
            user_input = input("\n用户提问: ").strip()
            if user_input.lower() in ["退出", "exit"]:
                break
                
            response = client.generate(user_input)
            print("\nAI回答:")
            print(response)
            
        except KeyboardInterrupt:
            print("\n对话已终止")
            break

if __name__ == "__main__":
    main()
