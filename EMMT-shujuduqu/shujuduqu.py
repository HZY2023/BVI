# import pandas as pd
#
# from PIL import Image
# import io
# import os
# # 文件路径
# parquet_file =r'/gb/HZY/EMMT/EMMT-test/0000.parquet'
#
# # 使用pandas读取parquet文件
# df = pd.read_parquet(parquet_file, engine='pyarrow')  # 可以替换为 engine='fastparquet' 如果你使用fastparquet
#
# # 假设df是您的DataFrame，并且df['image']列包含图像的字节数据
# for index, row in df.iterrows():
#     image_bytes = row['image']['bytes']
#     image_stream = io.BytesIO(image_bytes)
#     image = Image.open(image_stream)
#     # 现在可以显示、保存或处理图像
#     # 例如，显示图像:
#     # image.show()
#     # 或者保存图像:
#     image.save(os.path.join('/gb/HZY/EMMT/EMMT-test/images', f"{row['image']['path']}"))


# 如果需要，可以将图像保存到文件
# image.save('output.jpg')

# 或者转换为其他格式，例如PNG
# image.save('output.png')


import pandas as pd
from PIL import Image
import io
import os

# 文件路径
parquet_file = '/gb/HZY/EMMT/EMMT-train/0000.parquet'

# 使用pandas读取parquet文件
df = pd.read_parquet(parquet_file, engine='pyarrow')  # 可以替换为 engine='fastparquet' 如果你使用fastparquet

# 创建保存图像的目录
images_dir = '/gb/HZY/EMMT/EMMT-train/images'
os.makedirs(images_dir, exist_ok=True)

# 假设df是您的DataFrame，并且df['image']列包含图像的字节数据
for index, row in df.iterrows():
    # 读取图像字节数据
    image_bytes = row['image']['bytes']

    # 创建一个BytesIO流来读取字节数据
    image_stream = io.BytesIO(image_bytes)

    # 使用PIL打开图像流
    image = Image.open(image_stream)

    # 检查图像是否为RGBA模式，并在必要时转换为RGB
    if image.mode == 'RGBA':
        # 转换为RGB模式
        image = image.convert('RGB')

    # 保存图像。确保'path'键对应于您的DataFrame中的正确列
    image_save_path = os.path.join('/gb/HZY/EMMT/EMMT-train/images', f"{row['image']['path']}")
