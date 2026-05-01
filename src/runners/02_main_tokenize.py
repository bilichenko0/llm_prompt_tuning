import sys
from pathlib import Path

core_path = str(Path(__file__).parent.parent / "core")
if core_path not in sys.path:
    sys.path.append(core_path)

from DirectoryManager import DirectoryManager
from JSONHandler import JSONHandler
from Tokenizer import Tokenizer

def main():
    dm = DirectoryManager()
    output_path = dm.get_output_path("effective_java_data.json")

    jh = JSONHandler()
    jh.load(output_path)
    
    tz = Tokenizer(jh)
    updated_count = tz.run()
    
    jh.save(output_path)
    print(f"Tokenization finished. Updated {updated_count} fragments.")

if __name__ == "__main__":
    main()