import tiktoken

class Tokenizer:
    def __init__(self, json_handler):
        self.jh = json_handler

        self.encoder_3_5 = tiktoken.get_encoding("cl100k_base") 

        try:
            self.encoder_4o = tiktoken.get_encoding("o200k_base")
        except ValueError:
            self.encoder_4o = None

    def run(self) -> int:
        to_process = self.jh.get_by_status("parsed")
        updated_count = 0

        for item in to_process:
            text = item["text"]

            tokens_dict = {
                "cl100k_base": len(self.encoder_3_5.encode(text))
            }
            if self.encoder_4o:
                tokens_dict["o200k_base"] = len(self.encoder_4o.encode(text))
            
            self.jh.update_item(item["id"], {
                "status": "tokenized",
                "metadata": {
                    "tokens": tokens_dict
                }
            })
            updated_count += 1
            
        return updated_count