import csv
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
    try:
        with httpx.Client() as client:
            with client.stream("GET", image_url) as response:
                response.raise_for_status()
                if os.path.exists(f"dataset/downloads/{image_id}.jpg"):
                    return image_id, True
                with open(f"dataset/downloads/{image_id}.jpg", "wb") as img_file:
                    for chunk in response.iter_bytes():
                        img_file.write(chunk)
        return image_id, True
    except Exception as e:
        print(f"Failed to download {image_url}: {e}")
        return image_id, False


# 多线程下载图片
def download_images_concurrently(file_path, max_workers=10):
    image_data = load_image_data(file_path)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_image = {
            executor.submit(download_image, row): row for row in image_data
        }
        for future in tqdm(as_completed(future_to_image), total=len(image_data)):
            image_id, success = future.result()
            if success:
                tqdm.write(f"Downloaded {image_id}.jpg")
            else:
                tqdm.write(f"Failed to download {image_id}.jpg")


# 主程序
if __name__ == "__main__":
    file_path = "photos.tsv000"
    download_images_concurrently(file_path, max_workers=4)
