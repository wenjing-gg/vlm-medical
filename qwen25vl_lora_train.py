#!/usr/bin/env python3
import os
import torch
import json
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class SimpleDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=2048):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        conversations = sample.get('conversations', [])
        
        text = ""
        for conv in conversations:
            role = conv.get('role', '')
            content = conv.get('content', '')
            if role == 'user':
                text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == 'assistant':
                text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        text += "<|im_start|>assistant\n"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        vocab_size = self.tokenizer.vocab_size
        input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }
        
        return result

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    max_length = max(len(ids) for ids in input_ids)
    
    padded_input_ids = []
    padded_attention_masks = []
    padded_labels = []
    
    for ids, mask, label in zip(input_ids, attention_masks, labels):
        padding_length = max_length - len(ids)
        padded_input_ids.append(torch.cat([ids, torch.zeros(padding_length, dtype=ids.dtype)]))
        padded_attention_masks.append(torch.cat([mask, torch.zeros(padding_length, dtype=mask.dtype)]))
        padded_labels.append(torch.cat([label, torch.full((padding_length,), -100, dtype=label.dtype)]))
    
    return {
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(padded_attention_masks),
        'labels': torch.stack(padded_labels)
    }

def main():
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    train_data_path = "data/train.json"
    output_dir = "output"
    batch_size = 16
    num_epochs = 500
    learning_rate = 2e-4
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda:3")
    print(f"使用设备: {device}")
    
    print("加载模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map={"": 1},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    )
    model.config.use_cache = False  # 解决use_cache和gradient checkpointing冲突
    if not hasattr(model, 'lm_head'):
        vocab_size = tokenizer.vocab_size
        hidden_size = model.config.hidden_size
        model.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
    model.lm_head = model.lm_head.to(model.dtype).to(device)
    model = prepare_model_for_kbit_training(model)
    if not hasattr(model, 'prepare_inputs_for_generation'):
        def prepare_inputs_for_generation(self, *args, **kwargs):
            return self.model.prepare_inputs_for_generation(*args, **kwargs)
        model.prepare_inputs_for_generation = prepare_inputs_for_generation.__get__(model)
    print("设置LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model = model.to(device)
    model.print_trainable_parameters()
    
    print("加载数据集...")
    train_dataset = SimpleDataset(train_data_path, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print("开始训练...")
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                hidden_states = outputs.last_hidden_state
                if next(model.lm_head.parameters()).device != hidden_states.device:
                    model.lm_head = model.lm_head.to(hidden_states.device)
                logits = model.lm_head(hidden_states)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"显存不足，停止训练: {e}")
                    return
                else:
                    print(f"训练时出错: {e}")
                    continue
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} 完成, 平均损失: {avg_loss:.4f}")
        
        checkpoint_path = os.path.join(output_dir, f"epoch_{epoch+1}")
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
    
    print("训练完成！")

if __name__ == "__main__":
    main() 