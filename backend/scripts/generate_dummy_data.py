"""
生成测试用 Markdown 数据的脚本。
主要用于生成包含随机文本和模拟敏感信息的虚拟文档，供后续测试（如文本分块、敏感信息清理等）使用。
"""
import os
import random
import string

def generate_random_word():
    """生成一个长度在 3 到 10 之间的随机小写字母单词。"""
    return ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))

def generate_sentence():
    """生成一个包含 5 到 15 个随机单词的句子，首字母大写并以句号结尾。"""
    words = [generate_random_word() for _ in range(random.randint(5, 15))]
    return " ".join(words).capitalize() + "."

def generate_paragraph():
    """生成一个包含 3 到 8 个句子的段落。"""
    sentences = [generate_sentence() for _ in range(random.randint(3, 8))]
    return " ".join(sentences)

def generate_dummy_markdown(data_dir: str, num_files: int = 100, words_per_file: int = 5000):
    """
    生成指定数量的虚拟 Markdown 文档。
    
    Args:
        data_dir (str): 生成文档的保存目录。
        num_files (int): 需要生成的文件总数，默认为 100。
        words_per_file (int): 每个文件期望生成的单词数，默认为 5000。
    """
    # 如果目标目录不存在，则创建该目录
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    total_words = 0
    for i in range(num_files):
        # 格式化文件名为 doc_001.md, doc_002.md 等格式
        filepath = os.path.join(data_dir, f"doc_{i+1:03d}.md")
        with open(filepath, "w", encoding="utf-8") as f:
            # 写入基础的 Markdown 结构
            f.write(f"# 技术文档 {i+1}\n\n")
            f.write("## 1. 简介\n\n")
            
            words_written = 0
            # 循环生成段落，直到满足目标单词数
            while words_written < words_per_file:
                paragraph = generate_paragraph()
                f.write(paragraph + "\n\n")
                words_written += len(paragraph.split())
            
            # Add some sensitive info to test cleaning
            # 添加一些敏感信息，用于测试后续的数据脱敏/清理功能
            f.write("联系邮箱：test_user@example.com\n")
            f.write("联系电话：123-456-7890\n")
            
            total_words += words_written
            
    print(f"Generated {num_files} files with total words: {total_words}")

if __name__ == "__main__":
    # 将生成的测试数据保存在相对于当前脚本的两层父目录的 data 文件夹下
    generate_dummy_markdown(os.path.join(os.path.dirname(__file__), "../../data"))
