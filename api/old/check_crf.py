
import torch
try:
    from torchcrf import CRF
    print("Successfully imported CRF from torchcrf")
    
    num_tags = 5
    model = CRF(num_tags)
    print(f"CRF object attributes: {dir(model)}")
    
    if hasattr(model, 'decode'):
        print("Has decode method")
    else:
        print("No decode method")
        
except ImportError:
    print("Could not import torchcrf")
except Exception as e:
    print(f"An error occurred: {e}")
