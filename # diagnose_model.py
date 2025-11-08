# debug_model.py
import torch

def debug_model_weights():
    model_path = "Transformer/results/transformer_model_final.pth"
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"Checkpoint类型: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print("Checkpoint键:")
            for key in checkpoint.keys():
                print(f"  - {key}")
                if hasattr(checkpoint[key], 'shape'):
                    print(f"    形状: {checkpoint[key].shape}")
        else:
            print("Checkpoint是直接的状态字典")
            print(f"状态字典键: {list(checkpoint.keys())[:10]}")  # 只显示前10个
            
    except Exception as e:
        print(f"加载失败: {e}")

if __name__ == "__main__":
    debug_model_weights()