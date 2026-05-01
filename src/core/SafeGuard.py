import tiktoken

class SafeGuard:
    def __init__(self, json_handler, token_limit: int = 1_000_000):
        self.json_handler = json_handler
        self.token_limit = token_limit


        self.encoder_3_5 = tiktoken.get_encoding("cl100k_base")
        try:
            self.encoder_4o = tiktoken.get_encoding("o200k_base")
        except ValueError:
            self.encoder_4o = None

    def check(self) -> dict:
        ready_items = self.json_handler.get_by_status('tokenized')

        total_tokens = {
            "cl100k_base": 0,
            "o200k_base": 0
        }

        for item in ready_items:
            tokens_dict = item.get("metadata", {}).get("tokens")
            
            if isinstance(tokens_dict, dict):
                total_tokens["cl100k_base"] += tokens_dict.get("cl100k_base", 0)
                total_tokens["o200k_base"] += tokens_dict.get("o200k_base", 0)


        max_tokens = max(total_tokens.values()) if total_tokens else 0
        is_safe = max_tokens <= self.token_limit

        return {
            "is_safe": is_safe,
            "total_tokens": total_tokens, 
            "items_count": len(ready_items),
            "limit": self.token_limit
        }

    def is_fragment_safe(self, 
                       prompt_text: str, 
                       model_base: str = "cl100k_base", 
                       max_context: int = 16_385) -> dict:
        
        # if model_base == "o200k_base" and self.encoder_4o:
        #     tokens = len(self.encoder_4o.encode(prompt_text))
        # else:
        #     tokens = len(self.encoder_3_5.encode(prompt_text))
            
        # is_safe = tokens <= max_context
        
        # return {
        #     "is_safe": is_safe,
        #     "tokens": tokens,
        #     "limit": max_context
        # }

        tokens_dict = {
            "cl100k_base": len(self.encoder_3_5.encode(prompt_text))
        }
        if self.encoder_4o:
            tokens_dict["o200k_base"] = len(self.encoder_4o.encode(prompt_text))
            
        is_safe = max(tokens_dict.values()) <= max_context
        
        return {
            "is_safe": is_safe,
            "tokens": tokens_dict,
            "limit": max_context
        }