import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np
import cv2
import os

class HighQualityInpainter:
    def __init__(self, model_id="stabilityai/stable-diffusion-2-inpainting"):
        """
        Initialize high-quality inpainting pipeline.
        
        Args:
            model_id: Hugging Face model ID for inpainting
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the inpainting model with high-quality settings
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

        # Optimize pipeline
        # if self.device == "cuda":
            # self.pipe.enable_xformers_memory_efficient_attention()

    def prepare_high_quality_image(self, image_path, preserve_original=True):
        """
        Prepare high-quality image while preserving original dimensions.
        
        Args:
            image_path: Path to input image
            preserve_original: Whether to preserve original dimensions
        
        Returns:
            Prepared high-resolution image and original dimensions
        """
        # Open image
        image = Image.open(image_path).convert("RGB")
        
        # Store original dimensions for later restoration
        original_width, original_height = image.size
        
        # Determine optimal processing size (SD works best with multiples of 8 or 64)
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

    def create_mask_from_points(self, image, points):
        """
        Create a mask from a list of points [start_x, start_y, end_x, end_y, width, height].
        
        Args:
            image: Input image
            points: List containing [start_x, start_y, end_x, end_y, width, height]
        
        Returns:
            Mask for inpainting
        """
        # Convert image to numpy array to get dimensions
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Create empty mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Parse points
        if len(points) >= 6:
            start_x, start_y, end_x, end_y, width, height = points[:6]
            
            # Option 1: Use start and end coordinates to define a rectangle
            x1, y1 = start_x, start_y
            x2, y2 = end_x, end_y
            
            # Ensure coordinates are within image boundaries
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))
            
            # Draw rectangle on mask
            mask[y1:y2, x1:x2] = 255
            
            # Option 2: If option 1 doesn't create a reasonable mask, use width/height
            if np.sum(mask) == 0 or abs(x2-x1) < 5 or abs(y2-y1) < 5:
                # Use start_x, start_y as the top-left corner and width, height for dimensions
                x = max(0, min(start_x, w-1))
                y = max(0, min(start_y, h-1))
                width = min(width, w-x)
                height = min(height, h-y)
                
                mask[y:y+height, x:x+width] = 255
        else:
            print("Invalid points format. Using default rectangle in center.")
            # Default rectangle in the center of the image
            x = w // 4
            y = h // 4
            width = w // 2
            height = h // 2
            mask[y:y+height, x:x+width] = 255
        
        # Mask refinement
        # Expand mask slightly
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Smooth mask edges
        mask = cv2.GaussianBlur(mask, (11, 11), 0)
        
        return Image.fromarray(mask)

    def inpaint(self, 
                image_path, 
                output_path, 
                points,
                prompt="completely empty space, absolutely nothing, clean blank background, perfectly clear area, transparent, no objects whatsoever, pristine empty surface, void space, complete nothingness",
                negative_prompt="any object, thing, item, subject, content, elements, artifacts, shapes, structures, patterns, textures, features, details, distortion, noise, anything at all",
                num_inference_steps=50,
                guidance_scale=9.0,
                preserve_original_quality=True):
        """
        Advanced inpainting with points-based mask creation.
        
        Args:
            image_path: Input image path
            output_path: Output image path
            points: List containing [start_x, start_y, end_x, end_y, width, height]
            prompt: Positive guidance prompt
            negative_prompt: Negative guidance prompt
            num_inference_steps: Number of diffusion steps for better quality
            guidance_scale: How strongly to adhere to the prompt (higher = stronger)
            preserve_original_quality: Whether to maintain original image dimensions and quality
        
        Returns:
            Inpainted image
        """
        # Prepare optimized processing image and store original dimensions
        image, original_dimensions = self.prepare_high_quality_image(image_path, preserve_original_quality)
        
        # Create mask from points
        mask = self.create_mask_from_points(image, points)
        
        # Make sure mask has proper dimensions
        mask = mask.resize(image.size, Image.NEAREST)
        
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
        
        # Save high-quality result with original image's quality preserved
        # Detect original format
        _, file_extension = os.path.splitext(image_path)
        
        # Save with appropriate quality based on format
        if file_extension.lower() in ['.jpg', '.jpeg']:
            inpainted_image.save(output_path, quality=95)  # High JPEG quality
        elif file_extension.lower() in ['.png']:
            inpainted_image.save(output_path, compress_level=1)  # Low compression for PNG
        else:
            # Default high quality save
            inpainted_image.save(output_path, quality=95)
            
        print(f"High-quality inpainted image saved to {output_path}")
        
        return inpainted_image

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="High-Quality Image Inpainting with Points")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", help="Path to save inpainted image")
    parser.add_argument("--points", type=int, nargs='+', required=True,
                        help="Points as a list [start_x, start_y, end_x, end_y, width, height]")
    parser.add_argument("--steps", type=int, default=100, 
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
        args.output = f"{base}_inpainted{ext}"
    
    # Initialize and run high-quality inpainting
    inpainter = HighQualityInpainter()
    
    # Use default or custom prompts
    prompt = args.prompt if args.prompt else "completely empty space, absolutely nothing, clean blank background, perfectly clear area, transparent, no objects whatsoever, pristine empty surface, void space, complete nothingness"
    negative_prompt = args.negative_prompt if args.negative_prompt else "any object, thing, item, subject, content, elements, artifacts, shapes, structures, patterns, textures, features, details, distortion, noise, anything at all"
    
    inpainter.inpaint(
        image_path=args.image, 
        output_path=args.output, 
        points=args.points,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        preserve_original_quality=args.preserve_quality
    )

# Example usage:
# python script.py --image "path/to/image.jpg" --points 100 150 400 350 300 200

if __name__ == "__main__":
    main()