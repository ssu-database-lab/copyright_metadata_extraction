import numpy
import torch
try:
    from torchcrf import CRF
    print("Imported CRF")
except ImportError:
    print("CRF not found")
    exit()

try:
    crf = CRF(5, batch_first=True)
    print("CRF supports batch_first=True")
except TypeError as e:
    print(f"CRF does NOT support batch_first=True: {e}")
    crf = CRF(5)
    print("Created CRF with default args")

# Test forward
emissions = torch.randn(3, 10, 5) # B, L, C
tags = torch.tensor([[0, 1, 2]*3 + [0], [0, 1, 2]*3 + [0], [0, 1, 2]*3 + [0]]) # B, L (10)
mask = torch.ones(3, 10).byte()

print(f"Emissions: {emissions.shape}")
print(f"Tags: {tags.shape}")

try:
    # Try batch_first=True usage if supported
    loss = crf(emissions, tags, mask=mask.bool(), reduction='mean')
    print("Forward pass successful (batch_first=True)")
except Exception as e:
    print(f"Forward pass failed (batch_first=True): {e}")
    
    # Try transposed
    print("Trying transposed inputs...")
    emissions_t = emissions.transpose(0, 1)
    tags_t = tags.transpose(0, 1)
    mask_t = mask.transpose(0, 1)
    try:
        loss = crf(emissions_t, tags_t, mask=mask_t.bool(), reduction='mean')
        print("Forward pass successful (transposed)")
    except Exception as e:
        print(f"Forward pass failed (transposed): {e}")
