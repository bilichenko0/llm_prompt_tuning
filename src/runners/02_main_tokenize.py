# import sys
# from pathlib import Path

# core_path = str(Path(__file__).parent.parent / "core")
# if core_path not in sys.path:
#     sys.path.append(core_path)

# from DirectoryManager import DirectoryManager
# from JSONHandler import JSONHandler
# from Tokenizer import Tokenizer

# def main():
#     dm = DirectoryManager()
#     output_path = dm.get_output_path("effective_java_data.json")

#     jh = JSONHandler()
#     jh.load(output_path)
    
#     tz = Tokenizer(jh)
#     updated_count = tz.run()
    
#     jh.save(output_path)
#     print(f"Tokenization finished. Updated {updated_count} fragments.")

# if __name__ == "__main__":
#     main()


import sys
from pathlib import Path

core_path = str(Path(__file__).parent.parent / "core")
if core_path not in sys.path:
    sys.path.append(core_path)

from ConfigManager import ConfigManager
from JSONHandler import JSONHandler
from Tokenizer import Tokenizer

def main():
    cfg = ConfigManager()
    jh = JSONHandler()
    
    books = cfg.get_books(only_enabled=True)
    strategies = cfg.get_fragmentation_strategies()

    for book in books:
        for strategy in strategies:
            filename = f"{book['id']}_{strategy}.json"
            file_path = cfg.dm.get_output_path(filename)
            
            if not Path(file_path).exists():
                print(f"[!] File {filename} does not exist.")
                continue

            jh.load(file_path)
            
            tz = Tokenizer(jh)
            updated_count = tz.run()
            
            jh.save(file_path)
            print(f"{filename} is tokenized. Number of updated: {updated_count}")

if __name__ == "__main__":
    main()