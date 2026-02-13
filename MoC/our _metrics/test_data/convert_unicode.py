"""
Unicode 转义序列转换工具
将 JSON 文件中的 Unicode 编码转换为可读的中文
"""

import json
import sys

def convert_file(input_file, output_file=None):
    """
    转换 JSON 文件中的 Unicode 编码
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径（可选，默认为 input_file_readable.json）
    """
    # 读取文件
    print(f"读取文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 确定输出文件名
    if output_file is None:
        if input_file.endswith('.json'):
            output_file = input_file.replace('.json', '_readable.json')
        else:
            output_file = input_file + '_readable.json'
    
    # 保存为可读格式
    print(f"转换并保存到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    # 显示前几个元素预览
    print("\n" + "="*60)
    print("预览（前3个元素）:")
    print("="*60)
    for i, item in enumerate(data[:3]):
        print(f"\n[{i}] {item[:100]}..." if len(item) > 100 else f"\n[{i}] {item}")
    
    if len(data) > 3:
        print(f"\n... 还有 {len(data)-3} 个元素")
    
    print("\n" + "="*60)
    print(f"✓ 转换完成！共 {len(data)} 个文本块")
    print(f"✓ 可读文件: {output_file}")
    print("="*60)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        # 默认转换当前目录的文件
        input_file = 'db_qa_semantic_68.json'
        print(f"使用默认文件: {input_file}")
    else:
        input_file = sys.argv[1]
    
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None
    
    try:
        convert_file(input_file, output_file)
    except FileNotFoundError:
        print(f"错误: 文件不存在 - {input_file}")
    except json.JSONDecodeError:
        print(f"错误: 文件不是有效的 JSON 格式 - {input_file}")
    except Exception as e:
        print(f"错误: {e}")
