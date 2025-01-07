

# save_results.py
def save_to_file(results, filename="results.txt"):
    with open(filename, "w") as file:  # 使用 'a' 模式，避免每次覆盖内容
        file.write(results + "\n")

# 导入各个文件的结果

print("Results saved to file.")
