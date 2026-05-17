import json
import os

class JSONHandler:
    def __init__(self):
        self.data = {}
        self.status_index = {}

    @staticmethod
    def get_template(item_id, text, source, category="sentence"):
        return {
            "id": item_id,
            "text": text,
            "source": source,
            "type": category,
            "status": "parsed",
            "metadata": {
                "length": len(text),
                "tokens": None,
                "metrics": {
                    "compilation": 0,
                    "test_pass_rate": None,
                    "code_bleu": None
                }
            }
        }

    @staticmethod
    def is_valid(filepath: str) -> bool:
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                json.load(f)
            return True
        except (json.JSONDecodeError, ValueError):
            return False

    def load(self, filepath: str) -> dict:
        if not os.path.exists(filepath):
            self.data = {}
            self.status_index = {}
            return self.data
            
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_list = json.load(f)
            self.data = {item['id']: item for item in raw_list}
            self._rebuild_index()
        return self.data

    def _rebuild_index(self):
        self.status_index = {}
        for item_id, item in self.data.items():
            status = item.get("status")
            if status not in self.status_index:
                self.status_index[status] = list()
            self.status_index[status].append(item_id)

    def get_by_id(self, item_id: str):
        return self.data.get(item_id)

    def get_by_status(self, status: str) -> list:
        ids = self.status_index.get(status, list())
        return [self.data[i] for i in ids]

    def update_item(self, item_id: str, new_fields: dict) -> bool:
        if item_id not in self.data:
            return False
        
        item = self.data[item_id]
        old_status = item.get("status")

        if "metadata" in new_fields and "metadata" in item:
            item["metadata"].update(new_fields.pop("metadata"))
        
        item.update(new_fields)
        new_status = item.get("status")

        if old_status != new_status:
            if old_status in self.status_index:
                self.status_index[old_status].discard(item_id)
            if new_status not in self.status_index:
                self.status_index[new_status] = list()
            self.status_index[new_status].append(item_id)
            
        return True

    def save(self, filepath: str, data: list = None):
        save_data = data if data is not None else list(self.data.values())
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

    def load_raw_pure(self, filepath: str) -> dict:
        raw_data = {}

        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
            except (FileNotFoundError, json.decoder.JSONDecodeError):
                pass
        return raw_data