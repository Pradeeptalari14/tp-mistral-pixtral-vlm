"""Dynamic Patch Tokenizer & Aspect Ratio Resizer for Pixtral 12B.

Pixtral 12B does not force images into a fixed square resolution (e.g., 224x224 or 336x336).
Instead, it dynamically allocates tokens based on native image aspect ratio and pixel dimensions,
ensuring extreme fidelity for technical blueprints, wide documents, and fine text OCR.
"""

import math
from typing import Dict, Tuple


class DynamicPatchTokenizer:
    """Calculates patch tokens dynamically while preserving native aspect ratios."""

    def __init__(self, patch_size: int = 16, max_image_tokens: int = 4096, min_image_tokens: int = 128):
        self.patch_size = patch_size
        self.max_image_tokens = max_image_tokens
        self.min_image_tokens = min_image_tokens

    def calculate_tokens(self, width: int, height: int) -> Dict[str, any]:
        """Computes grid dimensions, token count, and aspect ratio metadata."""
        aspect_ratio = width / height
        raw_patches_w = math.ceil(width / self.patch_size)
        raw_patches_h = math.ceil(height / self.patch_size)
        raw_tokens = raw_patches_w * raw_patches_h

        # Downscale grid if tokens exceed max budget
        if raw_tokens > self.max_image_tokens:
            scale = math.sqrt(self.max_image_tokens / raw_tokens)
            effective_w = int(raw_patches_w * scale)
            effective_h = int(raw_patches_h * scale)
            effective_tokens = effective_w * effective_h
        else:
            effective_w = max(1, raw_patches_w)
            effective_h = max(1, raw_patches_h)
            effective_tokens = max(self.min_image_tokens, raw_tokens)

        return {
            "original_dimensions": (width, height),
            "aspect_ratio": round(aspect_ratio, 3),
            "patch_grid": (effective_w, effective_h),
            "allocated_tokens": effective_tokens,
            "compression_ratio": round((width * height) / (effective_tokens * (self.patch_size ** 2)), 2),
        }


def main():
    tokenizer = DynamicPatchTokenizer(patch_size=16, max_image_tokens=2048)
    
    # 4K Architectural Blueprint (3840 x 2160)
    spec_4k = tokenizer.calculate_tokens(3840, 2160)
    print("4K Blueprint Patch Allocation:", spec_4k)
    assert spec_4k["allocated_tokens"] <= 2048

    # Tall Mobile Screenshot (1080 x 2400)
    spec_tall = tokenizer.calculate_tokens(1080, 2400)
    print("Tall Mobile Screenshot Patch Allocation:", spec_tall)
    assert spec_tall["aspect_ratio"] < 1.0


if __name__ == "__main__":
    main()
