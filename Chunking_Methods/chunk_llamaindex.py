import json
import time
import os
import multiprocessing
from tqdm import tqdm
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import Document

INPUT_FILE = "/data/h50056789/Rag_Chunking/Corpus/LongBench/2wikimqa.jsonl" 
OUTPUT_DIR = "/data/h50056789/Rag_Chunking/Chunk_Result/Llamaindex_Chunk"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
NUM_WORKERS = 4


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def init_parser():
    cache_dir = "/data/h50056789/Rag_Chunking/tiktoken_cache" 
    os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir
    return SimpleNodeParser.from_defaults(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )


def process_context(context_text, doc_id, parser):
    start_time = time.time()
    
    doc = Document(text=context_text)
    nodes = parser.get_nodes_from_documents([doc], show_progress=False)
    
    splits = []
    for node in nodes:
        node_text = node.text if hasattr(node, 'text') else node.get_content()
        splits.append([node_text, doc_id])
    
    end_time = time.time()
    
    return {
        'splits': splits,
        'time_cost': end_time - start_time
    }


def process_line(line_data):
    try:
        data = json.loads(line_data)
        doc_id = data.get('_id', '')
        context = data.get('context', '')
        
        if not context:
            print(f"Skip empty context for ID: {doc_id}")
            return None
        
        print(f"Processing ID: {doc_id}")
        parser = init_parser()
        result = process_context(context, doc_id, parser)
        
        return result
    except Exception as e:
        print(f"Error processing line: {e}")
        return None


def main():
    create_directory(OUTPUT_DIR)
    
    print(f"Reading input file: {INPUT_FILE}")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Total lines: {len(lines)}")
    
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(process_line, lines), total=len(lines)):
            if result:
                results.append(result)
    
    all_splits = []
    total_time = 0
    for result in results:
        all_splits.extend(result['splits'])
        total_time += result['time_cost']
    
    output_data = {
        "filepath": INPUT_FILE,
        "splits": all_splits,
        "time_cost": total_time
    }
    
    input_basename = os.path.basename(INPUT_FILE).replace('.jsonl', '')
    output_file = os.path.join(OUTPUT_DIR, f"{input_basename}_llamaindex_chunk.json")
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessed {len(results)} documents")
    print(f"Total splits: {len(all_splits)}")
    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
