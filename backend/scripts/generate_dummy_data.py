import os
import random
import string

def generate_random_word():
    return ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))

def generate_sentence():
    words = [generate_random_word() for _ in range(random.randint(5, 15))]
    return " ".join(words).capitalize() + "."

def generate_paragraph():
    sentences = [generate_sentence() for _ in range(random.randint(3, 8))]
    return " ".join(sentences)

def generate_dummy_markdown(data_dir: str, num_files: int = 100, words_per_file: int = 5000):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    total_words = 0
    for i in range(num_files):
        filepath = os.path.join(data_dir, f"doc_{i+1:03d}.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# 技术文档 {i+1}\n\n")
            f.write("## 1. 简介\n\n")
            words_written = 0
            while words_written < words_per_file:
                paragraph = generate_paragraph()
                f.write(paragraph + "\n\n")
                words_written += len(paragraph.split())
            
            # Add some sensitive info to test cleaning
            f.write("联系邮箱：test_user@example.com\n")
            f.write("联系电话：123-456-7890\n")
            
            total_words += words_written
            
    print(f"Generated {num_files} files with total words: {total_words}")

if __name__ == "__main__":
    generate_dummy_markdown(os.path.join(os.path.dirname(__file__), "../../data"))
