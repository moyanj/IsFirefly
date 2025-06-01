import csv
import time
import httpx
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# 确保下载文件夹存在
os.makedirs("dataset/downloads", exist_ok=True)


# 读取所有图片信息
def load_image_data(file_path):
    with open(file_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # 跳过第一行（表头）
        return list(reader)


# 下载单个图片
def download_image(row):
    image_id = row[0]
    image_url = row[2]
    try_count = 0
    if os.path.exists(f"dataset/downloads/{image_id}.jpg"):
        return image_id, True
    try:
        with httpx.Client() as client:
            response = client.get(image_url)
            response.raise_for_status()

            with open(f"dataset/downloads/{image_id}.jpg", "wb") as img_file:
                img_file.write(response.content)
        return image_id, True
    except Exception as e:
        #  print(f"Failed to download {image_url}: {e}")
        try_count += 1
        time.sleep(1)
    return image_id, False


# 多线程下载图片
def download_images_concurrently(file_path, max_workers=20):
    image_data = load_image_data(file_path)
    total = len(image_data)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 使用list comprehension确保所有future在一个列表中
        futures = [executor.submit(download_image, row) for row in image_data]

        # 使用tqdm的position参数防止多线程输出混乱
        with tqdm(total=total, desc="Downloading", unit="img") as pbar:
            for future in as_completed(futures):
                try:
                    image_id, success = future.result()

                    # 可选：根据success状态更新描述
                    if success:
                        pbar.update(1)
                        pbar.set_postfix_str(f"Last: {image_id}")
                    else:
                        pbar.set_postfix_str(f"Failed: {image_id}")
                except Exception as e:
                    pbar.set_postfix_str(f"Error: {str(e)}")


# 主程序
if __name__ == "__main__":
    file_path = "photos.csv000"
    download_images_concurrently(file_path, max_workers=6)
