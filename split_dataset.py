import os
import shutil
import random
import csv
from glob import glob
from pathlib import Path

# ===================== 【核心自定义参数】 =====================
SOURCE_DIR = "/root/autodl-tmp/datasets/MiniImageNet"
OUTPUT_DIR = "/root/autodl-tmp/xiaorong-train/5way1shot-MiniImageNet"
WAY = 5
SHOT = 1
QUERY_PER_CLASS = 15
NUM_TASKS = 1000
GROUND_TRUTH_FILENAME = "ground_truth.csv"

# ===================== 【新增：随机种子】 =====================
RANDOM_SEED = 42   # 固定随机种子；改成别的整数，就会生成另一套可复现任务

# ===================== 【新增：episode列表文件名】 =====================
EPISODE_FILENAME = "episode_list.csv"
# -------------------------------------------------------------


def get_image_paths(category_dir):
    """
    获取指定类别文件夹下的所有有效图片路径
    :param category_dir: 类别文件夹路径
    :return: 有效图片路径列表（仅包含PNG/JPG/JPEG格式）
    """
    img_extensions = ['.png', '.jpg', '.jpeg']
    img_paths = []
    for ext in img_extensions:
        img_paths.extend(glob(os.path.join(category_dir, f'*{ext}'), recursive=False))
        img_paths.extend(glob(os.path.join(category_dir, f'*{ext.upper()}'), recursive=False))
    return img_paths


def print_parameters():
    """仅打印参数信息，去掉用户输入确认步骤"""
    print("="*80)
    print("📋 本次数据集构建参数信息：")
    print(f"  1. 源图片根目录        : {SOURCE_DIR}")
    print(f"  2. 输出数据集目录      : {OUTPUT_DIR}")
    print(f"  3. 少样本way值         : {WAY} (每个task包含{WAY}个类别)")
    print(f"  4. 少样本shot值        : {SHOT} (每个类别在support集有{SHOT}张样本)")
    print(f"  5. 每个类别query样本数 : {QUERY_PER_CLASS} (每个类别在query集有{QUERY_PER_CLASS}张样本)")
    print(f"  6. 要构建的task数量    : {NUM_TASKS}")
    print(f"  7. 随机模式            : 固定随机种子 = {RANDOM_SEED}")
    print(f"  8. 真实标签CSV文件名   : {GROUND_TRUTH_FILENAME}")
    print(f"  9. Episode列表文件名   : {EPISODE_FILENAME}")
    print("="*80)
    print("\n🚀 开始构建小样本学习数据集...")


def build_few_shot_dataset():
    """构建小样本学习数据集核心函数（标准小样本划分范式+生成真实标签CSV+保存episode列表）"""

    # ===================== 【新增：创建固定随机数生成器】 =====================
    rng = random.Random(RANDOM_SEED)

    # ===================== 1. 初始化真实标签存储列表 =====================
    ground_truth = [["img_name", "label"]]

    # ===================== 【新增：初始化episode列表】 =====================
    episode_records = [[
        "task_id",
        "split",
        "label",
        "original_category",
        "original_image_path",
        "saved_relative_path",
        "saved_image_name"
    ]]

    # ===================== 2. 参数校验 =====================
    if not os.path.exists(SOURCE_DIR):
        raise ValueError(f"❌ 源数据目录不存在：{SOURCE_DIR}")
    if WAY <= 0:
        raise ValueError(f"❌ way值必须大于0，当前值：{WAY}")
    if SHOT <= 0:
        raise ValueError(f"❌ shot值必须大于0，当前值：{SHOT}")
    if QUERY_PER_CLASS <= 0:
        raise ValueError(f"❌ 每个类别query样本数必须大于0，当前值：{QUERY_PER_CLASS}")
    if NUM_TASKS <= 0:
        raise ValueError(f"❌ 任务数量必须大于0，当前值：{NUM_TASKS}")

    category_folders = [f for f in glob(os.path.join(SOURCE_DIR, '*')) if os.path.isdir(f)]
    if len(category_folders) < WAY:
        raise ValueError(f"❌ 源目录下的类别数量({len(category_folders)})小于指定的way值({WAY})")

    # ===================== 3. 创建输出根目录 =====================
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ===================== 4. 遍历每个task构建数据集 =====================
    for task_id in range(1, NUM_TASKS + 1):
        task_dir = os.path.join(OUTPUT_DIR, f'task{task_id}')
        support_dir = os.path.join(task_dir, 'support')
        query_dir = os.path.join(task_dir, 'query')

        os.makedirs(support_dir, exist_ok=True)
        os.makedirs(query_dir, exist_ok=True)

        # ===================== 4.1 固定种子下随机选择way个类别 =====================
        selected_categories = rng.sample(category_folders, WAY)

        # ===================== 4.2 处理每个选中的类别 =====================
        query_img_counter = 1
        for label_id, category_dir in enumerate(selected_categories):
            all_imgs = get_image_paths(category_dir)
            required_imgs = SHOT + QUERY_PER_CLASS
            if len(all_imgs) < required_imgs:
                raise ValueError(
                    f"❌ 类别 {os.path.basename(category_dir)} 的有效图片数量({len(all_imgs)}) "
                    f"小于所需数量(shot={SHOT} + query_per_class={QUERY_PER_CLASS} = {required_imgs})"
                )

            # ===================== 4.3 固定种子下随机打乱图片顺序 =====================
            rng.shuffle(all_imgs)

            # ===================== 4.4 划分support集（前shot张） =====================
            category_support_dir = os.path.join(support_dir, str(label_id))
            os.makedirs(category_support_dir, exist_ok=True)

            support_imgs = all_imgs[:SHOT]
            original_category = os.path.basename(category_dir)

            for img_path in support_imgs:
                saved_img_name = os.path.basename(img_path)
                saved_path = os.path.join(category_support_dir, saved_img_name)
                shutil.copy(img_path, saved_path)

                # ===================== 【新增：记录support到episode列表】 =====================
                saved_relative_path = os.path.relpath(saved_path, OUTPUT_DIR)
                episode_records.append([
                    task_id,
                    "support",
                    label_id,
                    original_category,
                    img_path,
                    saved_relative_path,
                    saved_img_name
                ])

            # ===================== 4.5 划分query集（从剩余中随机选QUERY_PER_CLASS张） =====================
            remaining_imgs = all_imgs[SHOT:]
            query_imgs = rng.sample(remaining_imgs, QUERY_PER_CLASS)

            for img_path in query_imgs:
                img_ext = Path(img_path).suffix
                new_img_name = f'task{task_id}_{query_img_counter}{img_ext}'
                saved_path = os.path.join(query_dir, new_img_name)

                shutil.copy(img_path, saved_path)

                # ground truth
                ground_truth.append([new_img_name, label_id])

                # ===================== 【新增：记录query到episode列表】 =====================
                saved_relative_path = os.path.relpath(saved_path, OUTPUT_DIR)
                episode_records.append([
                    task_id,
                    "query",
                    label_id,
                    original_category,
                    img_path,
                    saved_relative_path,
                    new_img_name
                ])

                query_img_counter += 1

        print(f"✅ 完成Task {task_id} 的构建，路径：{task_dir}")

    # ===================== 5. 生成真实标签CSV文件 =====================
    csv_file_path = os.path.join(OUTPUT_DIR, GROUND_TRUTH_FILENAME)
    try:
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(ground_truth)
        print(f"✅ 真实标签CSV文件已生成：{csv_file_path}")
    except Exception as e:
        raise ValueError(f"❌ 生成真实标签CSV失败：{str(e)}")

    # ===================== 【新增：生成episode列表CSV文件】 =====================
    episode_file_path = os.path.join(OUTPUT_DIR, EPISODE_FILENAME)
    try:
        with open(episode_file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(episode_records)
        print(f"✅ Episode列表CSV文件已生成：{episode_file_path}")
    except Exception as e:
        raise ValueError(f"❌ 生成Episode列表CSV失败：{str(e)}")


if __name__ == '__main__':
    print_parameters()

    try:
        build_few_shot_dataset()
        print(f"\n🎉 所有任务构建完成！最终数据集路径：{OUTPUT_DIR}")
    except Exception as e:
        print(f"\n❌ 构建失败：{str(e)}")
