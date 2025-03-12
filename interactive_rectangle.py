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
       #  if self.device == "cuda":
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

    def create_advanced_mask(self, image, mask_method='interactive', mask_params=None):
        """
        Create an advanced mask with refinement options.
        
        Args:
            image: Input image
            mask_method: Masking technique
            mask_params: Additional mask generation parameters
        
        Returns:
            Refined mask
        """
        # Convert image to numpy array
        img_array = np.array(image)
        
        # Create initial mask
        if mask_method == 'rectangle':
            h, w = img_array.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Interactive rectangle selection
            mask_window = "Draw Rectangle (Click and drag, press 's' to save, 'ESC' to cancel)"
            cv2.namedWindow(mask_window)
            
            rect_start = None
            rect_end = None
            drawing = False
            temp_img = img_array.copy()
            
            def draw_rectangle(event, x, y, flags, param):
                nonlocal rect_start, rect_end, drawing, temp_img
                
                if event == cv2.EVENT_LBUTTONDOWN:
                    drawing = True
                    rect_start = (x, y)
                    # Reset temp_img to original when starting new rectangle
                    temp_img = img_array.copy()
                
                elif event == cv2.EVENT_MOUSEMOVE:
                    if drawing:
                        # Create a copy to draw preview rectangle
                        temp_img = img_array.copy()
                        rect_end = (x, y)
                        cv2.rectangle(temp_img, rect_start, rect_end, (0, 0, 255), 2)
                
                elif event == cv2.EVENT_LBUTTONUP:
                    drawing = False
                    rect_end = (x, y)
                    # Draw final rectangle
                    cv2.rectangle(temp_img, rect_start, rect_end, (0, 0, 255), 2)
            
            cv2.setMouseCallback(mask_window, draw_rectangle)
            
            while True:
                cv2.imshow(mask_window, temp_img)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s') and rect_start and rect_end:
                    # Create the mask from the selected rectangle
                    x1, y1 = min(rect_start[0], rect_end[0]), min(rect_start[1], rect_end[1])
                    x2, y2 = max(rect_start[0], rect_end[0]), max(rect_start[1], rect_end[1])
                    
                    # Fill the mask
                    mask[y1:y2, x1:x2] = 255
                    break
                elif key == 27:  # ESC key
                    # Reset everything
                    mask = np.zeros_like(mask)
                    break
            
            cv2.destroyAllWindows()
            
            # Use default rectangle if no selection was made
            if np.max(mask) == 0:
                print("No selection made. Using default rectangle.")
                x = mask_params.get('x', w//4) if mask_params else w//4
                y = mask_params.get('y', h//4) if mask_params else h//4
                width = mask_params.get('width', w//2) if mask_params else w//2
                height = mask_params.get('height', h//2) if mask_params else h//2
                
                mask[y:y+height, x:x+width] = 255
        
        elif mask_method == 'interactive':
            # OpenCV interactive mask drawing
            mask_window = "Draw Mask (Press 's' to save, 'ESC' to cancel)"
            cv2.namedWindow(mask_window)
            
            mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
            drawing = False
            
            def draw_mask(event, x, y, flags, param):
                nonlocal mask, drawing
                if event == cv2.EVENT_LBUTTONDOWN:
                    drawing = True
                    cv2.circle(mask, (x, y), 30, 255, -1)
                elif event == cv2.EVENT_MOUSEMOVE:
                    if drawing:
                        cv2.circle(mask, (x, y), 30, 255, -1)
                elif event == cv2.EVENT_LBUTTONUP:
                    drawing = False
            
            cv2.setMouseCallback(mask_window, draw_mask)
            
            while True:
                display = img_array.copy()
                display[mask > 0] = [0, 0, 255]
                cv2.imshow(mask_window, display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    break
                elif key == 27:
                    mask = np.zeros_like(mask)
                    break
            
            cv2.destroyAllWindows()
        
        # Mask refinement techniques
        # Expand mask slightly
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Smooth mask edges
        mask = cv2.GaussianBlur(mask, (11, 11), 0)
        
        return Image.fromarray(mask)

    def inpaint(self, 
                image_path, 
                output_path, 
                mask_method='interactive', 
                mask_params=None,
                prompt="completely empty space, absolutely nothing, clean blank background, perfectly clear area, transparent, no objects whatsoever, pristine empty surface, void space, complete nothingness",
                negative_prompt="any object, thing, item, subject, content, elements, artifacts, shapes, structures, patterns, textures, features, details, distortion, noise, anything at all",
                num_inference_steps=50,
                guidance_scale=9.0,
                preserve_original_quality=True):
        """
        Advanced inpainting with quality preservation.
        
        Args:
            image_path: Input image path
            output_path: Output image path
            mask_method: Mask generation technique
            mask_params: Mask generation parameters
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
        
        # Create advanced mask
        mask = self.create_advanced_mask(image, mask_method, mask_params)
        
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
    
    parser = argparse.ArgumentParser(description="High-Quality Image Inpainting")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", help="Path to save inpainted image")
    parser.add_argument("--method", default="interactive", 
                        choices=["rectangle", "interactive"],
                        help="Mask generation method")
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
        mask_method=args.method,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        preserve_original_quality=args.preserve_quality
    )

if __name__ == "__main__":
    main()