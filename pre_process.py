from PIL import Image
import os
import shutil
import torchvision.transforms as transforms
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# 定义原始目录和输出目录
raw_dirs = ["dataset/yes", "dataset/no"]
output_dir = "preprocessed"

# 定义数据增强操作（使用更高效的操作）
transform = transforms.Compose(
    [
        transforms.RandomRotation(degrees=25),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomSolarize(p=0.25, threshold=128),
    ]
)


# 预计算所有文件路径
def get_image_paths(raw_dirs):
    image_paths = []
    for raw_dir in raw_dirs:
        class_name = os.path.basename(raw_dir)
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        for file in os.listdir(raw_dir):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(raw_dir, file)
                output_base = os.path.splitext(file)[0]
                base_output_path = os.path.join(output_class_dir, f"{output_base}.jpg")

                image_paths.append((img_path, output_class_dir, output_base))
    return image_paths


# 处理单张图片的函数
def process_image(args):
    img_path, output_class_dir, output_base = args
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = img.resize((480, 480))

            # 保存原始处理后的图片
            base_output_path = os.path.join(output_class_dir, f"{output_base}.jpg")
            img.save(base_output_path, quality=95, optimize=True)

            # 生成增强图片
            for i in range(1):  # 生成1张增强图
                augmented_img = transform(img)
                augmented_path = os.path.join(
                    output_class_dir, f"{output_base}_{i}.jpg"
                )
                augmented_img.save(augmented_path, quality=95, optimize=True)

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return 0

    return 1  # 返回处理成功的图片数


def main():
    # 获取所有需要处理的图片路径
    image_paths = get_image_paths(raw_dirs)

    # 使用多进程处理
    print(f"Processing {len(image_paths)} images with {cpu_count()} workers...")
    with Pool(processes=cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap(process_image, image_paths),
                total=len(image_paths),
                desc="Processing images",
            )
        )

    # 统计总数
    total = sum(results) * 2  # 原始图 + 增强图
    print(f"Preprocessing completed. Total processed images: {total}")

    # 创建压缩包
    shutil.make_archive("preprocessed", "zip", "preprocessed")
    print("Zip archive created: 'preprocessed.zip'")


if __name__ == "__main__":
    main()
