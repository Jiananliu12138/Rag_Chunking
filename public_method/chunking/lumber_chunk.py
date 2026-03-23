import json
import time
import re
import requests
import os
import multiprocessing
from functools import partial
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


def sanitize_filename_component(value):
    """Convert arbitrary model identifiers into a filesystem-safe filename part."""
    text = str(value or "").strip()
    if not text:
        return "model"
    text = text.replace("\\", "_").replace("/", "_").replace(":", "_")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("._") or "model"


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


def qw_prompt_with_params(user_prompt, model_type=MODEL_TYPE, ds_base_url=DS_BASE_URL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS):
    while True:
        try:
            _prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            _post_data = {
                "model": model_type,
                "temperature": temperature,
                "prompt": _prompt,
                'max_tokens': max_tokens,
            }
            response = requests.post(f"{ds_base_url}/v1/completions", json=_post_data, timeout=600)
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
    
    if doc_id:
        splits = [
            [chunk, doc_id, chunk_id]
            for chunk_id, chunk in enumerate(new_final_chunks, start=1)
        ]
    else:
        splits = [[chunk] for chunk in new_final_chunks]
    
    return {
        'splits': splits,
        'time_cost': end_time - start_time
    }


def process_context_with_params(context_text, doc_id, model_type=MODEL_TYPE, ds_base_url=DS_BASE_URL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS):
    start_time = time.time()
    
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

        gpt_output = qw_prompt_with_params(user_prompt=question, model_type=model_type, ds_base_url=ds_base_url, temperature=temperature, max_tokens=max_tokens)

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
    
    if doc_id:
        splits = [
            [chunk, doc_id, chunk_id]
            for chunk_id, chunk in enumerate(new_final_chunks, start=1)
        ]
    else:
        splits = [[chunk] for chunk in new_final_chunks]
    
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


def _process_line_with_params(line_data, model_type, ds_base_url, temperature, max_tokens):
    """
    模块级函数，用于 multiprocessing，避免 pickle 局部函数的问题。
    """
    try:
        data = json.loads(line_data)
        doc_id = data.get('_id', '')
        context = data.get('context', '')
        
        if not context:
            return None
        
        return process_context_with_params(context, doc_id, model_type, ds_base_url, temperature, max_tokens)
    except Exception as e:
        print(f"Error processing line: {e}")
        return None


def chunk_file(input_file: str, output_dir: str, model_type: str = MODEL_TYPE, ds_base_url: str = DS_BASE_URL, temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS, num_workers: int = NUM_WORKERS):
    """
    对文件进行基于LLM的语义分块处理
    
    Args:
        input_file: 输入文件路径（必填）
        output_dir: 输出目录路径（必填）
        model_type: 模型类型（用户必填，默认使用全局MODEL_TYPE）
        ds_base_url: 模型服务URL（用户必填，默认使用全局DS_BASE_URL）
        temperature: 温度参数（可选，默认使用全局TEMPERATURE）
        max_tokens: 最大Token数（可选，默认使用全局MAX_TOKENS）
        num_workers: 工作进程数（可选，默认使用全局NUM_WORKERS）
    
    Returns:
        dict: {"success": bool, "output_file": str, "message": str}
    """
    try:
        create_directory(output_dir)
        
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 使用 functools.partial 绑定参数，避免 pickle 局部函数的问题
        process_func = partial(
            _process_line_with_params,
            model_type=model_type,
            ds_base_url=ds_base_url,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = []
            for result in tqdm(pool.imap_unordered(process_func, lines), total=len(lines)):
                if result:
                    results.append(result)
        
        all_splits = []
        total_time = 0
        for result in results:
            all_splits.extend(result['splits'])
            total_time += result['time_cost']
        
        output_data = {
            "filepath": input_file,
            "splits": all_splits,
            "time_cost": total_time
        }
        
        input_basename = os.path.basename(input_file).replace('.jsonl', '')
        safe_model_type = sanitize_filename_component(model_type)
        output_file = os.path.join(output_dir, f"{input_basename}_lumber_chunk_{safe_model_type}.json")
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nProcessed {len(results)} documents")
        print(f"Total splits: {len(all_splits)}")
        print(f"Results saved to: {output_file}")
        
        return {
            "success": True,
            "output_file": output_file,
            "message": f"Successfully processed {len(results)} documents with {len(all_splits)} splits"
        }
    
    except Exception as e:
        return {
            "success": False,
            "output_file": "",
            "message": f"Error: {str(e)}"
        }


def chunk_text(text_input: str, model_type: str = MODEL_TYPE, ds_base_url: str = DS_BASE_URL, temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS, num_workers: int = NUM_WORKERS):
    """
    对文本进行基于LLM的语义分块处理
    
    Args:
        text_input: 输入文本内容（必填）
        model_type: 模型类型（用户必填，默认使用全局MODEL_TYPE）
        ds_base_url: 模型服务URL（用户必填，默认使用全局DS_BASE_URL）
        temperature: 温度参数（可选，默认使用全局TEMPERATURE）
        max_tokens: 最大Token数（可选，默认使用全局MAX_TOKENS）
        num_workers: 工作进程数（对单个文本处理不使用，保留参数以保持接口一致性）
    
    Returns:
        dict: {"success": bool, "splits": [[text], ...], "time_cost": float, "message": str}
    """
    try:
        result = process_context_with_params(text_input, None, model_type, ds_base_url, temperature, max_tokens)
        
        return {
            "success": True,
            "splits": result['splits'],
            "time_cost": result['time_cost'],
            "message": f"Successfully chunked text into {len(result['splits'])} splits"
        }
    
    except Exception as e:
        print(f"Error processing text: {e}")
        return {
            "success": False,
            "splits": [],
            "time_cost": 0,
            "message": f"Error: {str(e)}"
        }


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
    safe_model_type = sanitize_filename_component(MODEL_TYPE)
    output_file = os.path.join(OUTPUT_DIR, f"{input_basename}_lumber_chunk_{safe_model_type}.json")
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessed {len(results)} documents")
    print(f"Total splits: {len(all_splits)}")
    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
