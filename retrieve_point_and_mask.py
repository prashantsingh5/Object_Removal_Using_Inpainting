import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np
import cv2
import os
import json

class HighQualityInpainter:
    def __init__(self, model_id="stabilityai/stable-diffusion-2-inpainting"):
        """
        Initialize high-quality inpainting pipeline.
        
        Args:
            model_id: Hugging Face model ID for inpainting
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Store selection points for reuse
        self.selection_points = {}
        
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

    def create_advanced_mask(self, image, image_path, mask_method='interactive', mask_params=None):
        """
        Create an advanced mask with refinement options.
        
        Args:
            image: Input image
            image_path: Path to input image (for naming saved mask files)
            mask_method: Masking technique
            mask_params: Additional mask generation parameters
        
        Returns:
            Refined mask
        """
        # Convert image to numpy array
        img_array = np.array(image)
        
        # Generate base filename for saving masks and points
        base_path, _ = os.path.splitext(image_path)
        mask_save_path = f"{base_path}_mask.png"
        points_save_path = f"{base_path}_points.json"
        
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
                    
                    # Save the points
                    selection_points = {
                        "method": "rectangle",
                        "points": {
                            "start_x": int(x1),
                            "start_y": int(y1),
                            "end_x": int(x2),
                            "end_y": int(y2),
                            "width": int(x2 - x1),
                            "height": int(y2 - y1)
                        }
                    }
                    
                    # Store in instance for reuse
                    self.selection_points[image_path] = selection_points
                    
                    # Save points to JSON file
                    with open(points_save_path, 'w') as f:
                        json.dump(selection_points, f, indent=4)
                    
                    print(f"Selection points saved to: {points_save_path}")
                    
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
                
                # Save default points
                selection_points = {
                    "method": "rectangle",
                    "points": {
                        "start_x": int(x),
                        "start_y": int(y),
                        "end_x": int(x + width),
                        "end_y": int(y + height),
                        "width": int(width),
                        "height": int(height)
                    }
                }
                self.selection_points[image_path] = selection_points
                
                # Save points to JSON file
                with open(points_save_path, 'w') as f:
                    json.dump(selection_points, f, indent=4)
                
                print(f"Default selection points saved to: {points_save_path}")
        
        elif mask_method == 'interactive':
            # OpenCV interactive mask drawing
            mask_window = "Draw Mask (Press 's' to save, 'ESC' to cancel)"
            cv2.namedWindow(mask_window)
            
            mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
            drawing = False
            
            # Store all brush strokes for saving
            brush_points = []
            
            def draw_mask(event, x, y, flags, param):
                nonlocal mask, drawing
                
                if event == cv2.EVENT_LBUTTONDOWN:
                    drawing = True
                    cv2.circle(mask, (x, y), 30, 255, -1)
                    # Add point to brush_points
                    brush_points.append({"x": int(x), "y": int(y), "event": "down"})
                
                elif event == cv2.EVENT_MOUSEMOVE:
                    if drawing:
                        cv2.circle(mask, (x, y), 30, 255, -1)
                        # Add point to brush_points
                        brush_points.append({"x": int(x), "y": int(y), "event": "move"})
                
                elif event == cv2.EVENT_LBUTTONUP:
                    drawing = False
                    # Add point to brush_points
                    brush_points.append({"x": int(x), "y": int(y), "event": "up"})
            
            cv2.setMouseCallback(mask_window, draw_mask)
            
            while True:
                display = img_array.copy()
                display[mask > 0] = [0, 0, 255]
                cv2.imshow(mask_window, display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    # Save brush points
                    selection_points = {
                        "method": "interactive",
                        "points": brush_points,
                        "brush_size": 30
                    }
                    self.selection_points[image_path] = selection_points
                    
                    # Save points to JSON file
                    with open(points_save_path, 'w') as f:
                        json.dump(selection_points, f, indent=4)
                    
                    print(f"Brush points saved to: {points_save_path}")
                    break
                elif key == 27:
                    mask = np.zeros_like(mask)
                    brush_points = []
                    break
            
            cv2.destroyAllWindows()
        
        # Mask refinement techniques
        # Expand mask slightly
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Smooth mask edges
        mask = cv2.GaussianBlur(mask, (11, 11), 0)
        
        # Save the mask image
        cv2.imwrite(mask_save_path, mask)
        print(f"Mask saved to: {mask_save_path}")
        
        return Image.fromarray(mask)
    
    def apply_saved_mask(self, image_path, points_path):
        """
        Apply a previously saved mask from points.
        
        Args:
            image_path: Path to input image
            points_path: Path to saved points JSON file
            
        Returns:
            Generated mask as PIL Image
        """
        # Load the image
        image, _ = self.prepare_high_quality_image(image_path)
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Create empty mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Load points from file
        with open(points_path, 'r') as f:
            selection_data = json.load(f)
        
        method = selection_data.get("method", "rectangle")
        points = selection_data.get("points", {})
        
        if method == "rectangle":
            # Extract rectangle coordinates
            x1 = points.get("start_x", 0)
            y1 = points.get("start_y", 0)
            x2 = points.get("end_x", w//2)
            y2 = points.get("end_y", h//2)
            
            # Fill the mask
            mask[y1:y2, x1:x2] = 255
            
        elif method == "interactive":
            # Draw brush strokes from saved points
            brush_size = selection_data.get("brush_size", 30)
            
            for point in points:
                x, y = point.get("x", 0), point.get("y", 0)
                cv2.circle(mask, (x, y), brush_size, 255, -1)
        
        # Apply refinements
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (11, 11), 0)
        
        return Image.fromarray(mask)

    def inpaint(self, 
                image_path, 
                output_path, 
                mask_method='interactive', 
                mask_params=None,
                points_path=None,
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
            points_path: Path to previously saved points file (optional)
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
        
        # Create or load mask
        if points_path and os.path.exists(points_path):
            print(f"Using saved points from: {points_path}")
            mask = self.apply_saved_mask(image_path, points_path)
        else:
            mask = self.create_advanced_mask(image, image_path, mask_method, mask_params)
        
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
    parser.add_argument("--points", type=str, default=None,
                        help="Path to previously saved points JSON file to reuse a mask")
    
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
        points_path=args.points,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        preserve_original_quality=args.preserve_quality
    )

if __name__ == "__main__":
    main()