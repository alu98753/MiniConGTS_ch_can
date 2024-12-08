import json
import matplotlib.pyplot as plt
from collections import Counter
import re
from transformers import AutoTokenizer, AutoModel

# 定義過濾函式
def contains_digit(text):
    """檢查文字中是否包含數字"""
    return bool(re.search(r'\d', text))

def is_single_character(text):
    """檢查文字是否為單一字符"""
    return len(text) == 1

def is_duplicate_triplet(existing_triplets, new_triplet):
    """
    檢查新 triplet 是否已存在於 existing_triplets 中
    假設 triplet 的唯一性基於 Aspect 和 Opinion
    """
    aspect, opinion, _, _ = new_triplet
    for trip in existing_triplets:
        if trip[0] == aspect and trip[1] == opinion:
            return True
    return False

def contains_stopwords(text, stopwords):
    """檢查文字是否僅包含停用詞"""
    words = list(text)  # 假設是單字
    return all(word in stopwords for word in words)

def filter_triplet(aspect, opinion, existing_triplets, stop_words=[], min_length=2, max_length=10):
    """
    應用過濾條件，返回 (True, None) 表示該 triplet 合法，
    否則返回 (False, "原因")
    """
    # 條件1: 排除包含數字的 triplet
    if contains_digit(aspect) or contains_digit(opinion):
        return False, "包含數字"

    # 條件2: 排除單一字符的 aspect 或 opinion
    if is_single_character(aspect) or is_single_character(opinion):
        return False, "單一字符"

    # 條件3: 排除重複的 triplet
    if is_duplicate_triplet(existing_triplets, (aspect, opinion, None, None)):
        return False, "重複 triplet"

    # 條件4: 排除僅包含停用詞的 triplet
    if stop_words:
        if contains_stopwords(aspect, stop_words) or contains_stopwords(opinion, stop_words):
            return False, "僅包含停用詞"

    # 條件5: 限制 Aspect 和 Opinion 的長度
    if not (min_length <= len(aspect) <= max_length):
        return False, "Aspect 長度不符"
    if not (min_length <= len(opinion) <= max_length):
        return False, "Opinion 長度不符"

    return True, None

# 讀取 JSON 文件
file_path = r"/mnt/md0/chen-wei/zi/MiniConGTS_chinese_can/data/D1/res14/NYCU_NLP_113A_TrainingSet.json"

# 加載 JSON 文件
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 初始化分類統計
intensity_v = []
intensity_a = []

# 初始化過濾後的 triplet 列表
filtered_data = []

# 初始化不符合條件的 triplet 列表
excluded_triplets = []

# 定義停用詞列表
stop_words = ["的", "是", "在", "和", "了", "有", "也"]  # 根據需要自定義

# 建立分詞器和模型（如果在此步驟中不需要使用分詞器和模型，可以省略）
# tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
# model = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext")

# 分類數據並過濾
for item in data:
    id_ = item.get("ID", "")
    sentence = item.get("Sentence", "")
    aspects = item.get("Aspect", [])
    aspect_fromtos = item.get("AspectFromTo", [])
    categories = item.get("Category", [])
    opinions = item.get("Opinion", [])
    opinion_fromtos = item.get("OpinionFromTo", [])
    intensities = item.get("Intensity", [])
    
    existing_triplets = []  # 用於檢查重複 triplet
    
    for aspect, aspect_ft, category, opinion, opinion_ft, intensity in zip(aspects, aspect_fromtos, categories, opinions, opinion_fromtos, intensities):
        # 提取 intensity 的 valence 和 arousal
        try:
            v, a = map(float, intensity.split('#'))
        except ValueError:
            # 如果 intensity 格式不正確，跳過該 triplet
            print(f"ID: {id_}, Aspect: {aspect}, Opinion: {opinion}, Intensity: {intensity} - 錯誤的 intensity 格式")
            excluded_triplets.append({
                "ID": id_,
                "Aspect": aspect,
                "Opinion": opinion,
                "Intensity": intensity,
                "Reason": "錯誤的 intensity 格式"
            })
            continue
        
        # 應用過濾條件
        valid, reason = filter_triplet(aspect, opinion, existing_triplets, stop_words=stop_words, min_length=2, max_length=10)
        if not valid:
            # 打印不符合條件的 triplet 及原因
            print(f"ID: {id_}, Aspect: {aspect}, Opinion: {opinion}, Intensity: {intensity} - 不符合條件: {reason}")
            excluded_triplets.append({
                "ID": id_,
                "Aspect": aspect,
                "Opinion": opinion,
                "Intensity": intensity,
                "Reason": reason
            })
            continue
        
        # 如果通過過濾，則追加至 existing_triplets
        existing_triplets.append((aspect, opinion, v, a))
        
        # 收集 intensity
        v_rounded = round(v)
        a_rounded = round(a)
        intensity_v.append(v_rounded)
        intensity_a.append(a_rounded)
        
        # 追加至過濾後的資料
        filtered_data.append({
            "ID": id_,
            "Sentence": sentence,
            "Aspect": aspect,
            "AspectFromTo": aspect_ft,
            "Category": category,
            "Opinion": opinion,
            "OpinionFromTo": opinion_ft,
            "Intensity": intensity
        })

# 計算 v 和 a 的比例
def calculate_ratios(intensity_list):
    total_count = len(intensity_list)
    counter = Counter(intensity_list)
    return {key: value / total_count for key, value in counter.items()}

ratios_v = calculate_ratios(intensity_v)
ratios_a = calculate_ratios(intensity_a)

# 打印結果
print("\nFiltered Data Count:", len(filtered_data))
print("\nIntensity 分布比例 (v):")
for intensity, ratio in sorted(ratios_v.items()):
    print(f"Intensity: {intensity}, 比例: {ratio:.2%}")

print("\nIntensity 分布比例 (a):")
for intensity, ratio in sorted(ratios_a.items()):
    print(f"Intensity: {intensity}, 比例: {ratio:.2%}")


print(f"excluded_triplets :")
for i in excluded_triplets:
    print(i)

# 如果需要將過濾後的資料保存至新 JSON 文件
# output_filtered_path = r"/mnt/md0/chen-wei/zi/MiniConGTS_chinese_can/data/D1/res14/NYCU_NLP_113A_TrainingSet_filtered.json"
# with open(output_filtered_path, 'w', encoding='utf-8') as f:
#     json.dump(filtered_data, f, ensure_ascii=False, indent=4)

# print(f"Filtered data saved to {output_filtered_path}")

# 將不符合條件的 triplets 保存至另一個 JSON 文件
output_excluded_path = r"/mnt/md0/chen-wei/zi/MiniConGTS_chinese_can/data/D1/res14/NYCU_NLP_113A_TrainingSet_excluded.json"
with open(output_excluded_path, 'w', encoding='utf-8') as f:
    json.dump(excluded_triplets, f, ensure_ascii=False, indent=4)

print(f"Excluded triplets saved to {output_excluded_path}")
