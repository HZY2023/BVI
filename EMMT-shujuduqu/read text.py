import pandas as pd

# 文件路径
# parquet_file = ['/gb/HZY/EMMT/EMMT-train/0000.parquet', '/gb/HZY/EMMT/EMMT-train/0001.parquet', '/gb/HZY/EMMT/EMMT-train/0002.parquet', '/gb/HZY/EMMT/EMMT-train/0003.parquet',]
parquet_file = ['/gb/HZY/EMMT/EMMT-test/0000.parquet']
for i in parquet_file:
    # 使用pandas读取parquet文件
    df = pd.read_parquet(parquet_file, engine='pyarrow')  # 可以替换为 engine='fastparquet' 如果你使用fastparquet

    # 在循环外部打开文件
    with open('/gb/HZY/EMMT/EMMT-test/src_text_test.txt', 'a+', encoding='utf-8') as source_txt, \
         open('/gb/HZY/EMMT/EMMT-test/tgt_text_test.txt', 'a+', encoding='utf-8') as tgt_txt:
        for source, target in zip(df['src_text'], df['trg_text']):
            source_txt.write(source + '\n')  # 确保使用 '\n' 作为换行符
            tgt_txt.write(target + '\n')