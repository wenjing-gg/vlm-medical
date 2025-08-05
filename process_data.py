#!/usr/bin/env python3
import os
import json
from datasets import load_dataset
from tqdm import tqdm
import base64
from PIL import Image
from io import BytesIO

def process_medical_dataset():
    """处理医学多模态数据集"""
    print("正在加载医学多模态数据集...")
    
    # 加载数据集
    dataset = load_dataset("FreedomIntelligence/Medical_Multimodal_Evaluation_Data")
    
    # 创建数据目录
    os.makedirs("data", exist_ok=True)
    
    processed_data = []
    
    # 处理所有分割
    for split_name, split_data in dataset.items():
        print(f"处理 {split_name} 分割...")
        
        for i, sample in enumerate(tqdm(split_data)):
            try:
                # 提取图像数据
                image = sample.get('image', None)
                
                # 处理图像格式
                if image is not None:
                    if isinstance(image, str):
                        # 如果是base64编码
                        if image.startswith('data:image'):
                            try:
                                image_data = image.split(',')[1]
                                image_bytes = base64.b64decode(image_data)
                                pil_image = Image.open(BytesIO(image_bytes))
                                # 转换为base64字符串存储
                                buffered = BytesIO()
                                pil_image.save(buffered, format="JPEG")
                                img_str = base64.b64encode(buffered.getvalue()).decode()
                                image = f"data:image/jpeg;base64,{img_str}"
                            except Exception as e:
                                print(f"处理图像时出错: {e}")
                                image = None
                        else:
                            # 如果是文件路径，尝试读取
                            try:
                                pil_image = Image.open(image)
                                buffered = BytesIO()
                                pil_image.save(buffered, format="JPEG")
                                img_str = base64.b64encode(buffered.getvalue()).decode()
                                image = f"data:image/jpeg;base64,{img_str}"
                            except Exception as e:
                                print(f"读取图像文件时出错: {e}")
                                image = None
                
                # 提取对话内容
                conversation = sample.get('conversation', [])
                
                # 如果没有对话，尝试其他字段
                if not conversation:
                    question = sample.get('question', '')
                    answer = sample.get('answer', '')
                    if question and answer:
                        conversation = [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer}
                        ]
                
                # 构建格式化的数据
                if conversation:
                    formatted_sample = {
                        "id": sample.get('id', f"{split_name}_{i}"),
                        "image": image,
                        "conversations": conversation
                    }
                    processed_data.append(formatted_sample)
                    
            except Exception as e:
                print(f"处理样本 {i} 时出错: {e}")
                continue
    
    # 分割训练和验证集
    import random
    random.shuffle(processed_data)
    
    val_size = int(len(processed_data) * 0.1)
    val_data = processed_data[:val_size]
    train_data = processed_data[val_size:]
    
    # 保存数据
    with open("data/train.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open("data/val.json", 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成！")
    print(f"训练集: {len(train_data)} 样本")
    print(f"验证集: {len(val_data)} 样本")
    print(f"数据已保存到 data/ 目录")

if __name__ == "__main__":
    process_medical_dataset() 