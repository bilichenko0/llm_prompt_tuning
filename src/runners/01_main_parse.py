# import sys
# from pathlib import Path

# core_path = str(Path(__file__).parent.parent / "core")
# if core_path not in sys.path:
#     sys.path.append(core_path)

# from DirectoryManager import DirectoryManager
# from JSONHandler import JSONHandler
# from PDFParser import PDFParser

# def main():
#     dm = DirectoryManager()
#     book_name = "Effective Java (2017, Addison-Wesley).pdf" 
#     book_path = dm.get_book_path(book_name)
#     output_path = dm.get_output_path("effective_java_data.json")

#     parser = PDFParser(book_path)
    
    
#     data = parser.parse(start_page=0, end_page=1000, strategy="paragraph")

#     jh = JSONHandler()
#     jh.save(output_path, data)
#     print("Parsing finished. Data saved to JSON.")

# if __name__ == "__main__":
#     main()



import sys
from pathlib import Path

core_path = str(Path(__file__).parent.parent / "core")
if core_path not in sys.path:
    sys.path.append(core_path)

from ConfigManager import ConfigManager
from JSONHandler import JSONHandler
from PDFParser import PDFParser

def main():
    cfg = ConfigManager()
    jh = JSONHandler()
    
    books = cfg.get_books(only_enabled=True)
    strategies = cfg.get_fragmentation_strategies()
    start_page, end_page = cfg.get_parsing_page_range()

    print(f"[{'DEBUG' if cfg.is_debug else 'PROD'}] Parsing. Pages from {start_page} to {end_page}")

    for book in books:
        book_path = cfg.dm.get_book_path(book["filename"])
        
        if not Path(book_path).exists():
            print(f"[!] cannot find book file: {book_path}.")
            continue

        parser = PDFParser(book_path)
        
        for strategy in strategies:
            out_filename = f"{book['id']}_{strategy}.json"
            output_path = cfg.dm.get_output_path(out_filename)
            
            data = parser.parse(start_page=start_page, end_page=end_page, strategy=strategy)
            
            jh.save(output_path, data)
            print(f"Saved: {len(data) if isinstance(data, list) else 'OK'} fragments.")

if __name__ == "__main__":
    main()