import os
import json


def append_data(info, json_path):
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}
    else:
        existing_data = []

    # 合并新旧数据
    existing_data.append(info)

    # 写回文件
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)


def save_data(info, json_path, cover=False):

    if not cover and os.path.exists(json_path):
        raise FileExistsError("File already exists")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)


def load_data(json_path):
    data = None
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    return data
