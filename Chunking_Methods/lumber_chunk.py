import json
import time
import re
import requests
import os
import multiprocessing
from tqdm import tqdm

# 配置
INPUT_FILE = "/data/h50056789/Rag_Chunking/Corpus/LongBench/2wikimqa.jsonl"
OUTPUT_DIR = "/data/h50056789/Rag_Chunking/Chunk_Result/Lumber_Chunk"
MODEL_TYPE = "Qwen2.5-7B-Instruct"
DS_BASE_URL = os.environ.get('DS_BASE_URL', 'http://localhost:8005')
NUM_WORKERS = 4
TEMPERATURE = 0.2
MAX_TOKENS = 3072


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def count_words(input_string):
    words = input_string.split()
    return round(1.2 * len(words))


def add_ids_(p, current_id):
    p = f'ID {current_id}: {p}'
    return p


system_prompt = """You will receive as input an english document with paragraphs identified by 'ID XXXX: <text>'.

Task: Find the first paragraph(not the first one) where the content clearly changes compared to the previous paragraphs.

Output: Return the ID of the paragraph with the content shift as in the exemplified format: 'Answer: ID XXXX'.

Additional Considerations: Avoid very long groups of paragraphs. 

Aim for a good balance between identifying content shifts and keeping groups manageable."""


def qw_prompt(user_prompt):
    while True:
        try:
            _prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            _post_data = {
                "model": MODEL_TYPE,
                "temperature": TEMPERATURE,
                "prompt": _prompt,
                'max_tokens': MAX_TOKENS,
            }
            response = requests.post(f"{DS_BASE_URL}/v1/completions", json=_post_data, timeout=600)
            res = response.json()
            return res['choices'][0]['text']
        except KeyError as e:
            print(f"KeyError: {e}. Full response: {res}")
            time.sleep(60)
        except Exception as e:
            if str(e) == "list index out of range":
                print("Model thinks prompt is unsafe")
                return "content_flag_increment"
            else:
                print(f"An error occurred: {e}. Retrying in 1 minute...")
                time.sleep(60)


def build_prompt(chunk_number, id_chunks_without_title):
    word_count = 0
    i = 0
    while word_count < 550 and i + chunk_number < len(id_chunks_without_title) - 1:
        i += 1
        final_document = "\n".join(map(lambda k: id_chunks_without_title[k], range(chunk_number, i + chunk_number)))
        word_count = count_words(final_document)

    if i == 1:
        final_document = "\n".join(map(lambda k: id_chunks_without_title[k], range(chunk_number, i + chunk_number)))
    else:
        final_document = "\n".join(map(lambda k: id_chunks_without_title[k], range(chunk_number, i - 1 + chunk_number)))

    question = f"\nDocument:\n{final_document}"

    word_count = count_words(final_document)
    chunk_number = chunk_number + i - 1

    return question, word_count, chunk_number


def get_final_chunks(new_id_list, id_chunks):
    new_final_chunks = []
    for i in range(len(new_id_list)):
        start_idx = new_id_list[i - 1] if i > 0 else 0
        end_idx = new_id_list[i]
        new_final_chunks.append('\n'.join(id_chunks[start_idx: end_idx]))

    return new_final_chunks


def process_context(context_text, doc_id):
    """对单个 context 进行分块"""
    start_time = time.time()
    
    # 按行分割 context (每个 Passage 作为一个段落)
    paragraph_chunks = context_text.strip().splitlines()
    paragraph_chunks = [line.strip() for line in paragraph_chunks if line.strip()]
    
    current_id = 0
    id_chunks = []
    id_chunks_without_title = []
    for c in paragraph_chunks:
        id_chunks.append(add_ids_(c, current_id))
        id_chunks_without_title.append(add_ids_(c.lstrip('# '), current_id))
        current_id += 1

    chunk_number = 0
    new_id_list = []
    word_count_aux = []
    
    while chunk_number < len(id_chunks_without_title) - 5:
        question, word_count, chunk_number = build_prompt(chunk_number, id_chunks_without_title)
        word_count_aux.append(word_count)

        gpt_output = qw_prompt(user_prompt=question)

        if gpt_output == "content_flag_increment":
            chunk_number = chunk_number + 1
            continue

        pattern = r"Answer: ID \d+"
        match = re.search(pattern, gpt_output)

        if match is None:
            print(gpt_output)
            print("repeat this one")
            continue

        gpt_output1 = match.group(0)
        print(gpt_output1)
        pattern = r'\d+'
        match = re.search(pattern, gpt_output1)
        chunk_number = int(match.group())
        new_id_list.append(chunk_number)
        if new_id_list[-1] == chunk_number:
            chunk_number = chunk_number + 1

    new_id_list.append(len(id_chunks))

    id_chunks = list(map(lambda c: re.sub(r'^ID \d+:\s*', '', c), id_chunks))

    new_final_chunks = get_final_chunks(new_id_list, id_chunks)

    end_time = time.time()
    
    splits = [[chunk, doc_id] for chunk in new_final_chunks]
    
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
        result = process_context(context, doc_id)
        
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
    output_file = os.path.join(OUTPUT_DIR, f"{input_basename}_lumber_chunk_{MODEL_TYPE}.json")
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessed {len(results)} documents")
    print(f"Total splits: {len(all_splits)}")
    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
