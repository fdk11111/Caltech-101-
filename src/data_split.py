import os
import shutil
from sklearn.model_selection import train_test_split

# 配置路径（根据实际路径修改）
ORIGINAL_DIR = r"D:\Projects\Caltech101-Finetuning\data\caltech-101\original\101_ObjectCategories"  # 原始数据路径
TARGET_DIR = r"D:\Projects\Caltech101-Finetuning\data\caltech-101"  # 目标路径
EXCLUDED_CLASS = "BACKGROUND_Google"  # 需要排除的背景类文件夹名


def main():
    # 检查原始路径是否存在
    assert os.path.exists(ORIGINAL_DIR), f"原始数据路径不存在: {ORIGINAL_DIR}"
    print(f"原始数据路径验证通过: {ORIGINAL_DIR}")

    # 创建目标目录（train/val/test）
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(TARGET_DIR, split)
        os.makedirs(split_dir, exist_ok=True)
        print(f"已创建目录: {split_dir}")

    # 获取有效类别列表（排除背景类）
    class_names = [
        cls for cls in os.listdir(ORIGINAL_DIR)
        if os.path.isdir(os.path.join(ORIGINAL_DIR, cls)) and cls != EXCLUDED_CLASS
    ]
    print(f"有效类别数: {len(class_names)} (已排除 {EXCLUDED_CLASS})")

    # 遍历每个有效类别
    for class_name in class_names:
        class_path = os.path.join(ORIGINAL_DIR, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"处理类别 [{class_name}] | 原始图片数: {len(images)}")

        # 按比例划分：train(70%) → val(15%) + test(15%)
        train, temp = train_test_split(images, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        print(f"划分结果: train={len(train)}, val={len(val)}, test={len(test)}")

        # 复制文件到目标目录
        for split, files in zip(["train", "val", "test"], [train, val, test]):
            dest_dir = os.path.join(TARGET_DIR, split, class_name)
            os.makedirs(dest_dir, exist_ok=True)

            for img in files:
                src = os.path.join(class_path, img)
                dst = os.path.join(dest_dir, img)
                shutil.copy2(src, dst)  # 保留文件元数据

            print(f"复制完成: {split}/{class_name} ({len(files)}张)")

    print("\n数据集划分完成！最终结构验证：")
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(TARGET_DIR, split)
        num_classes = len(os.listdir(split_dir))
        print(f"- {split}: {num_classes}个类别")


if __name__ == "__main__":
    main()