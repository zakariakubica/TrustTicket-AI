import torch 
 
print("="*50) 
print("GPU CONFIGURATION TEST") 
print("="*50) 
print(f"PyTorch version: {torch.__version__}") 
print(f"CUDA available: {torch.cuda.is_available()}") 
if torch.cuda.is_available(): 
    print(f"CUDA version: {torch.version.cuda}") 
    print(f"GPU Name: {torch.cuda.get_device_name(0)}") 
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB") 
    x = torch.rand(1000, 1000).cuda() 
    y = torch.rand(1000, 1000).cuda() 
    z = torch.matmul(x, y) 
    print(f"? GPU test successful! Tensor on: {z.device}") 
else: 
    print("? CUDA not available") 
print("="*50) 
