import json
from JSONHandler import JSONHandler
from SafeGuard import SafeGuard

import os
from dotenv import load_dotenv
from openai import OpenAI

class Predictor:
    def __init__(self, 
                 fragments_jh: JSONHandler, 
                 safeguard: SafeGuard, 
                 dataset_path: str,
                 provider: str = "openai",
                 model_name: str = "gpt-3.5-turbo"):
        
        self.fragments_jh = fragments_jh
        self.safeguard = safeguard
        self.dataset_path = dataset_path
        self.dataset_sample = self._load_dataset_sample(limit=5)
        self.provider = provider.lower()

        self.predictions_jh = JSONHandler()
        self.predictions_data = []
        
        if self.provider == "openai":
            #load_dotenv()
            self.client = OpenAI()
        elif self.provider == "litellm":
            load_dotenv()

        self.model_name = model_name

    def _load_dataset_sample(self, limit: int = 5) -> list:
        tasks = []
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                for _ in range(limit):
                    line = f.readline()
                    if not line:
                        break
                    tasks.append(json.loads(line))
        except Exception as e:
            print(f"Error loading dataset: {e}")
        return tasks

    def build_prompt(self, fragment_text: str, task_prompt: str) -> str:
        result: str = f"""
You are an expert Java developer. 
Use the following programming advice to help you solve the task.

Programming Advice:
{fragment_text}

Task:
Complete the following Java code:
{task_prompt}"""
        return result

    def clean_llm_output(self, raw_text: str) -> str:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = '\n'.join(lines)
        return cleaned.strip()

    def get_llm_prediction(self, prompt_text: str) -> str:
        if self.provider == "openai":
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a strict Java code generator. You must output ONLY pure, raw Java code. NEVER use markdown formatting like ```java. NEVER write explanations or comments before or after the code. Just the raw code."},
                        {"role": "user", "content": prompt_text}
                    ],
                    max_tokens=300,
                    temperature=0.2
                )
                return response.choices[0].message.content
            except Exception as e:
                return None
        elif self.provider == "litellm":
            BASE_URL = os.getenv("LITELLM_BASE_URL")
            API_KEY = os.getenv("LITELLM_API_KEY", "")

            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt_text}],
                "temperature": 0.7,
            }

            r = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            result = data["choices"][0]["message"]["content"]
            return result
        else:
            print(f"Unknown provider: {self.provider}")
            return None

    def run_predictions(self, output_filepath: str, fragments_limit: int = 3):
        ready_fragments = self.fragments_jh.get_by_status('tokenized')
        fragments_to_process = ready_fragments[:fragments_limit]

        for fragment in fragments_to_process:
            for task in self.dataset_sample:
                final_prompt = self.build_prompt(fragment['text'], task['prompt'])
                check = self.safeguard.is_fragment_safe(final_prompt, model_base="cl100k_base")
                
                if check["is_safe"]:
                    predicted_code = self.get_llm_prediction(final_prompt)
                    if predicted_code:
                        clean_code = self.clean_llm_output(predicted_code)
                        prediction_item = {
                            "fragment_id": fragment["id"],
                            "task_id": task["task_id"],
                            "prompt_tokens": check["tokens"],
                            "predicted_code": clean_code,
                            "status": "pending_evaluation" 
                        }
                        self.predictions_data.append(prediction_item)
                else:
                    print(f"Tokens limit exceeded.")

        self.predictions_jh.save(output_filepath, self.predictions_data)