import fitz
import re
import hashlib
from JSONHandler import JSONHandler

class PDFParser:
    def __init__(self, path: str):
        self.path = path
        self.doc = fitz.open(path)
        self.source_name = path.split('/')[-1].replace('.pdf', '')

    def extract_raw_text(self, start: int, end: int) -> str:
        full_text = ""
        for i in range(max(0, start), min(end, len(self.doc))):
            full_text += self.doc[i].get_text() + "\n\n"
        return full_text

    def clean_text(self, text: str) -> str:
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text) 
        text = text.replace('- ', '')   
        return text.strip()

    def generate_id(self, content: str) -> str:
        hash_part = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"node_{hash_part}"

    def split_content(self, text: str, strategy="sentence"):
        if strategy == "sentence":
            cleaned = self.clean_text(text)
            pattern = r'(?<=[.!?])\s+(?=[A-ZА-Я])'
            return [s.strip() for s in re.split(pattern, cleaned) if len(s) > 10]
        
        elif strategy == "paragraph":
            paragraphs = re.split(r'\n\s*\n', text)
            return [self.clean_text(p) for p in paragraphs if len(p.strip()) > 40]
        else:
            raise ValueError("Unknown splitting strategy")

        

        #else error
        return [self.clean_text(text)]

    def parse(self, start_page, end_page, strategy="sentence"):
        raw = self.extract_raw_text(start_page, end_page)
        
        fragments = self.split_content(raw, strategy)
        
        results = []
        for frag in fragments:
            node = JSONHandler.get_template(
                item_id=self.generate_id(frag),
                text=frag,
                source=self.source_name,
                category=strategy
            )
            results.append(node)
        return results