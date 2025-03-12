import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np
import os

class ObjectRemover:
    def __init__(self, model_id="stabilityai/stable-diffusion-2-inpainting"):
        """
        Initialize object removal pipeline.
        
        Args:
            model_id: Hugging Face model ID for inpainting
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the inpainting model with high-quality settings
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

    def prepare_image(self, image_path, preserve_original=True):
        """
        Prepare image while preserving original dimensions.
        
        Args:
            image_path: Path to input image or PIL Image
            preserve_original: Whether to preserve original dimensions
        
        Returns:
            Prepared high-resolution image and original dimensions
        """
        # Handle both file paths and PIL image objects
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path.convert("RGB")
        
        # Store original dimensions for later restoration
        original_width, original_height = image.size
        
        # Determine optimal processing size (SD works best with multiples of 8)
        # But we'll preserve aspect ratio
        max_dim = max(original_width, original_height)
        if max_dim > 1024:
            # For very large images, restrict size for processing
            scaling_factor = 1024 / max_dim
            process_width = int(original_width * scaling_factor)
            process_height = int(original_height * scaling_factor)
            # Ensure dimensions are multiples of 8
            process_width = (process_width // 8) * 8
            process_height = (process_height // 8) * 8
            process_image = image.resize((process_width, process_height), Image.LANCZOS)
        else:
            # For smaller images, adjust to nearest multiple of 8
            process_width = ((original_width + 7) // 8) * 8
            process_height = ((original_height + 7) // 8) * 8
            if process_width != original_width or process_height != original_height:
                process_image = image.resize((process_width, process_height), Image.LANCZOS)
            else:
                process_image = image
        
        return process_image, (original_width, original_height)

    def prepare_mask(self, mask_path, image_size):
        """
        Prepare mask to match image dimensions.
        
        Args:
            mask_path: Path to mask image or PIL Image
            image_size: Target size (width, height) to resize mask
        
        Returns:
            Processed mask
        """
        # Handle both file paths and PIL image objects
        if isinstance(mask_path, str):
            mask = Image.open(mask_path).convert("L")  # Convert to grayscale
        else:
            mask = mask_path.convert("L")
        
        # Resize mask to match image dimensions
        mask = mask.resize(image_size, Image.NEAREST)
        
        # Ensure mask has proper values (0 for background, 255 for masked area)
        mask_array = np.array(mask)
        if mask_array.max() > 0:  # Normalize if not binary
            mask_array = (mask_array > 127).astype(np.uint8) * 255
        
        # Optional: Refine mask edges (uncomment if needed)
        # kernel = np.ones((5,5), np.uint8)
        # mask_array = cv2.dilate(mask_array, kernel, iterations=1)
        # mask_array = cv2.GaussianBlur(mask_array, (11, 11), 0)
        
        return Image.fromarray(mask_array)

    def remove_object(self, 
                    image_path, 
                    mask_path,
                    output_path=None, 
                    prompt="completely empty space, absolutely nothing, clean blank background, perfectly clear area, transparent, no objects whatsoever, pristine empty surface, void space, complete nothingness",
                    negative_prompt="any object, thing, item, subject, content, elements, artifacts, shapes, structures, patterns, textures, features, details, distortion, noise, anything at all",
                    num_inference_steps=50,
                    guidance_scale=9.0,
                    preserve_original_quality=True):
        """
        Remove object using a provided mask.
        
        Args:
            image_path: Input image path or PIL Image
            mask_path: Mask image path or PIL Image
            output_path: Output image path (optional)
            prompt: Positive guidance prompt
            negative_prompt: Negative guidance prompt
            num_inference_steps: Number of diffusion steps for better quality
            guidance_scale: How strongly to adhere to the prompt (higher = stronger)
            preserve_original_quality: Whether to maintain original image dimensions and quality
        
        Returns:
            Inpainted image with object removed
        """
        # Prepare optimized processing image and store original dimensions
        image, original_dimensions = self.prepare_image(image_path, preserve_original_quality)
        
        # Prepare mask to match image dimensions
        mask = self.prepare_mask(mask_path, image.size)
        
        # Perform inpainting with enhanced settings
        inpainted_image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=1  # Control how much the image can change
        ).images[0]
        
        # Restore original dimensions if needed
        if preserve_original_quality and inpainted_image.size != original_dimensions:
            inpainted_image = inpainted_image.resize(original_dimensions, Image.LANCZOS)
        
        # Save result if output path is provided
        if output_path:
            # Detect original format if image_path is a string
            if isinstance(image_path, str):
                _, file_extension = os.path.splitext(image_path)
            else:
                file_extension = ".png"  # Default to PNG if image is passed as PIL Image
            
            # Save with appropriate quality based on format
            if file_extension.lower() in ['.jpg', '.jpeg']:
                inpainted_image.save(output_path, quality=95)  # High JPEG quality
            elif file_extension.lower() in ['.png']:
                inpainted_image.save(output_path, compress_level=1)  # Low compression for PNG
            else:
                # Default high quality save
                inpainted_image.save(output_path, quality=95)
                
            print(f"Image with object removed saved to {output_path}")
        
        return inpainted_image

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Object Removal with Pre-existing Mask")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--mask", required=True, help="Path to mask image")
    parser.add_argument("--output", help="Path to save result image")
    parser.add_argument("--steps", type=int, default=50, 
                        help="Number of inference steps (higher = better quality)")
    parser.add_argument("--guidance", type=float, default=9.0,
                        help="Guidance scale (higher = stronger prompt adherence)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt for inpainting guidance")
    parser.add_argument("--negative_prompt", type=str, default=None,
                        help="Custom negative prompt for inpainting guidance")
    parser.add_argument("--no-preserve-quality", action="store_false", dest="preserve_quality",
                        help="Disable preservation of original image quality")
    
    parser.set_defaults(preserve_quality=True)
    
    args = parser.parse_args()
    
    # Set default output path
    if not args.output:
        import os
        base, ext = os.path.splitext(args.image)
        args.output = f"{base}_removed{ext}"
    
    # Initialize object remover
    remover = ObjectRemover()
    
    # Use default or custom prompts
    prompt = args.prompt if args.prompt else "completely empty space, absolutely nothing, clean blank background, perfectly clear area, transparent, no objects whatsoever, pristine empty surface, void space, complete nothingness"
    negative_prompt = args.negative_prompt if args.negative_prompt else "any object, thing, item, subject, content, elements, artifacts, shapes, structures, patterns, textures, features, details, distortion, noise, anything at all"
    
    remover.remove_object(
        image_path=args.image, 
        mask_path=args.mask,
        output_path=args.output, 
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        preserve_original_quality=args.preserve_quality
    )

# Example usage:
# python object_remover.py --image path/to/image.jpg --mask path/to/mask.png --output result.png

if __name__ == "__main__":
    main()