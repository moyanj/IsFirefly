from PIL import Image
import os
import shutil
import torchvision.transforms as transforms
from tqdm import tqdm

# 定义原始目录和输出目录
raw_dirs = ["dataset/firefly", "dataset/no"]
output_dir = "preprocessed"

# 删除已存在的输出目录（如果存在）
shutil.rmtree(output_dir, ignore_errors=True)

# 定义数据增强操作
# 这里定义了随机旋转、随机水平翻转、随机调整亮度和对比度
transform = transforms.Compose(
    [
        transforms.RandomRotation(degrees=25),  # 随机旋转角度范围为 [-15, 15]
        transforms.RandomHorizontalFlip(p=0.2),  # 随机水平翻转，概率为 0.5
        transforms.RandAugment(4, 5),
    ]
)

# 遍历每个原始目录
for raw_dir in raw_dirs:
    # 获取当前目录的类别名称
    class_name = os.path.basename(raw_dir)
    output_class_dir = os.path.join(output_dir, class_name)

    # 遍历当前目录中的所有文件
    for file in tqdm(os.listdir(raw_dir)):
        # 检查文件是否是图片（可以根据需要扩展支持更多格式）
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            # 打开图片并进行预处理
            img_path = os.path.join(raw_dir, file)
            img = Image.open(img_path)
            img = img.convert("RGB")  # 确保图片是 RGB 格式
            img = img.resize((640, 640))  # 调整图片大小

            # 创建输出目录（如果不存在）
            os.makedirs(output_class_dir, exist_ok=True)

            # 保存原始处理后的图片
            output_file = os.path.splitext(file)[0] + ".jpg"  # 确保输出为 JPG 格式
            output_path = os.path.join(output_class_dir, output_file)
            img.save(output_path)

            # 对每张图片生成多张增强后的图片
            for i in range(10):  # 假设每张图片生成 5 张增强后的图片
                # 应用数据增强
                augmented_img = transform(img)

                # 保存增强后的图片
                augmented_output_file = os.path.splitext(file)[0] + f"_{i}.jpg"
                augmented_output_path = os.path.join(
                    output_class_dir, augmented_output_file
                )
                augmented_img.save(augmented_output_path)

print(f"Preprocessing completed. Processed images saved to '{output_dir}'.")

total = 0
for root, dirs, files in os.walk(output_dir):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            total += 1

print(f"Total number of processed images: {total}")

shutil.make_archive("preprocessed", "zip", "preprocessed")
print(f"Zip archive created: 'preprocessed.zip'")
