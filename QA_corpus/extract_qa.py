import json
import os

INPUT_FILE = "/data/h50056789/Rag_Chunking/Corpus/LongBench/narrativeqa.jsonl"
OUTPUT_DIR = "/data/h50056789/Rag_Chunking/QA_corpus"


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def main():
    create_directory(OUTPUT_DIR)
    
    print(f"Reading input file: {INPUT_FILE}")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Total lines: {len(lines)}")
    
    qa_pairs = []
    dataset_name = None
    
    for line in lines:
        data = json.loads(line)
        
        if dataset_name is None:
            dataset_name = data.get('dataset', 'unknown')
        
        qa_pair = {
            "input": data.get("input", ""),
            "_id": data.get("_id", ""),
            "answers": data.get("answers", [])
        }
        
        qa_pairs.append(qa_pair)
    
    output_file = os.path.join(OUTPUT_DIR, f"{dataset_name}.jsonl")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + '\n')
    
    print(f"\nExtracted {len(qa_pairs)} QA pairs")
    print(f"Dataset: {dataset_name}")
    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
