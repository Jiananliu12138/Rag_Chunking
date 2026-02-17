import json
import time
import os
import multiprocessing
from tqdm import tqdm
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document

# 配置
INPUT_FILE = "f:/thesis/Meta-Chunking/meta-chunking-dataset/meta-chunking/Original_Dataset/LongBench-main/data/2wikimqa.jsonl"
OUTPUT_DIR = "f:/thesis/Meta-Chunking/Chunk_Result"
EMBED_MODEL_PATH = "/data/h50056789/Rag_chunk_bench/model/bge-large-en-v1.5"
BUFFER_SIZE = 1
BREAKPOINT_THRESHOLD = 74
NUM_WORKERS = 4

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def init_splitter():
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_PATH)
    splitter = SemanticSplitterNodeParser(
        buffer_size=BUFFER_SIZE,
        breakpoint_percentile_threshold=BREAKPOINT_THRESHOLD,
        embed_model=embed_model
    )
    return splitter


def process_context(context_text, doc_id, splitter):
    start_time = time.time()
    
    doc = Document(text=context_text)
    nodes = splitter.get_nodes_from_documents([doc], show_progress=False)
    
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
        splitter = init_splitter()
        result = process_context(context, doc_id, splitter)
        
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
    output_file = os.path.join(OUTPUT_DIR, f"{input_basename}_semantic_chunk.json")
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessed {len(results)} documents")
    print(f"Total splits: {len(all_splits)}")
    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
