import sys
from pathlib import Path

core_path = str(Path(__file__).parent.parent / "core")
if core_path not in sys.path:
    sys.path.append(core_path)

from DirectoryManager import DirectoryManager
from JSONHandler import JSONHandler
from PDFParser import PDFParser

def main():
    dm = DirectoryManager()
    book_name = "Effective Java (2017, Addison-Wesley).pdf" 
    book_path = dm.get_book_path(book_name)
    output_path = dm.get_output_path("effective_java_data.json")

    parser = PDFParser(book_path)
    
    
    data = parser.parse(start_page=296, end_page=300, strategy="paragraph")

    jh = JSONHandler()
    jh.save(output_path, data)
    print("Parsing finished. Data saved to JSON.")

if __name__ == "__main__":
    main()