from locust import HttpUser, task, between, events
import json
import uuid

class ChatUser(HttpUser):
    wait_time = between(1, 2)
    
    def on_start(self):
        self.session_id = str(uuid.uuid4())

    @task(3)
    def test_health(self):
        self.client.get("/health")

    @task(1)
    def test_ask(self):
        """测试 /ask 端点。"""
        payload = {
            "query": "What is the architecture of this project?",
            "session_id": self.session_id
        }
        with self.client.post("/ask", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                if "answer" not in response.json():
                    response.failure("Missing answer in response")
            elif response.status_code == 429:
                response.success() # 触发限流，在负载下是预期的行为
            else:
                response.failure(f"Failed with status code: {response.status_code}")
