import json
from JSONHandler import JSONHandler
from SafeGuard import SafeGuard

import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import time

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
        self.dataset_sample = self._load_dataset_sample(limit=40)
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
        
        # Find the first instance of ```java and ending ```
        start_marker = "```java"
        end_marker = "```"
        
        start_idx = cleaned.find(start_marker)
        if start_idx != -1:
            # Move past the ```java marker
            content_start = start_idx + len(start_marker)
            # Find the closing ``` after the opening ```java
            end_idx = cleaned.find(end_marker, content_start)
            if end_idx != -1:
                # Extract only the content between the markers
                cleaned = cleaned[content_start:end_idx].strip()
                return cleaned
        
        # Fallback to original logic if ```java not found
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

            # Retry logic: 5 attempts, then sleep 5 minutes and retry up to 6 times
            max_5min_retries = 6
            retry_cycle = 0
            
            while retry_cycle < max_5min_retries:
                for attempt in range(1, 6):  # 5 attempts
                    try:
                        r = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload, timeout=60)
                        r.raise_for_status()
                        data = r.json()
                        result = data["choices"][0]["message"]["content"]
                        return result
                    except Exception as e:
                        print(f"{datetime.now()} LiteLLM request failed (attempt {attempt}/5, cycle {retry_cycle + 1}/{max_5min_retries}): {e}")
                        if attempt < 5:
                            time.sleep(3)  # Brief pause between retries
                        else:
                            # All 5 attempts failed
                            retry_cycle += 1
                            if retry_cycle < max_5min_retries:
                                print(f"{datetime.now()} All 5 retry attempts failed. Sleeping for 5 minutes before trying again (cycle {retry_cycle}/{max_5min_retries})...")
                                time.sleep(300)  # Sleep for 5 minutes (300 seconds)
                                break  # Break inner loop to restart the retry cycle
                            else:
                                print(f"{datetime.now()} Maximum retry cycles ({max_5min_retries}) reached. Giving up.")
                                return None
            
            return None
        else:
            print(f"Unknown provider: {self.provider}")
            return None

    def run_predictions(self, output_filepath: str, fragments_limit: int = 3):
        ready_fragments = self.fragments_jh.get_by_status('tokenized')
        fragments_to_process = ready_fragments[:fragments_limit]

        existing_predictions = self.predictions_jh.load_raw_pure(output_filepath)
        for item in existing_predictions:
            self.predictions_data.append(item)
        existing_predictions = {fragment_id:set(map(lambda x: x['task_id'], filter(lambda x: x["fragment_id"]==fragment_id, existing_predictions))) for fragment_id in set(map(lambda x: x["fragment_id"],existing_predictions))}

        i = 1
        i_max = len(fragments_to_process)
        for fragment in fragments_to_process:
            j = 1
            j_max = len(self.dataset_sample)
            for task in self.dataset_sample:
                skip = fragment["id"] in existing_predictions and task["task_id"] in existing_predictions[fragment["id"]]
                print(f"{datetime.now()} Fragment {i}/{i_max} Task {j}/{j_max} {"Skipping" if skip else "Predicting"}")
                
                j = j + 1

                if skip:
                    continue

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

                        self.predictions_jh.save(output_filepath, self.predictions_data)
                else:
                    print(f"Tokens limit exceeded.")
                
            i = i + 1

        self.predictions_jh.save(output_filepath, self.predictions_data)