import re

# 输入文件路径
input_file_path = '/gb/HZY/EMMT/bpe/test.zh'
# 输出文件路径
output_file_path = '/gb/HZY/EMMT/bpe/test_pre.zh'

# 检查字符是否为中文的函数
def is_chinese(char):
    return '\u4e00' <= char <= '\u9fff'

# 打开输入文件和输出文件
with open(input_file_path, 'r', encoding='utf-8') as input_file, \
     open(output_file_path, 'w', encoding='utf-8') as output_file:
    # 逐行读取输入文件
    for line in input_file:
        # 移除行首尾的空格
        line = line.strip()
        # 处理每个字符
        processed_line = ''
        for char in line:
            if is_chinese(char):
                processed_line += ' ' + char
            else:
                processed_line += char
        # 移除多余的空格
        processed_line = re.sub(r'\s+', ' ', processed_line).strip()
        # 写入处理后的行到输出文件
        output_file.write(processed_line + '\n')
