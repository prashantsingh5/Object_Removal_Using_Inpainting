import cv2
import numpy as np
import os
import argparse
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline

class ObjectSelector:
    def __init__(self):
        """Initialize the object selector."""
        # Store selection points
        self.selection_points = {}
        
    def prepare_image(self, image_path):
        """
        Load and prepare image for selection while preserving quality.
        
        Args:
            image_path: Path to input image
        
        Returns:
            Loaded image at original quality
        """
        # Load image directly with OpenCV to preserve quality
        img_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        # If image couldn't be loaded, try with PIL as fallback
        if img_array is None:
            # Open image with PIL
            pil_image = Image.open(image_path).convert("RGB")
            # Convert to numpy/OpenCV format
            img_array = np.array(pil_image)
            # Convert RGB to BGR for OpenCV
            if img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
        return img_array
    
    def select_rectangle(self, image_path):
        """
        Interactive rectangle selection.
        
        Args:
            image_path: Path to input image
        
        Returns:
            Selection coordinates in format [x1 y1 x2 y2 width height]
        """
        # Load image
        img_array = self.prepare_image(image_path)
        h, w = img_array.shape[:2]
        
        # Setup for interactive rectangle selection
        mask_window = "Select Object (Click and drag, press 's' to save, 'ESC' to cancel)"
        cv2.namedWindow(mask_window, cv2.WINDOW_NORMAL)
        
        # Set window size appropriate for the image dimensions
        h, w = img_array.shape[:2]
        window_width = min(w, 1200)  # Limit maximum window width
        window_height = int(h * (window_width / w))  # Maintain aspect ratio
        cv2.resizeWindow(mask_window, window_width, window_height)
        
        rect_start = None
        rect_end = None
        drawing = False
        temp_img = img_array.copy()
        selection = None
        
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
                    cv2.rectangle(temp_img, rect_start, rect_end, (0, 255, 0), 2)
            
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                rect_end = (x, y)
                # Draw final rectangle
                cv2.rectangle(temp_img, rect_start, rect_end, (0, 255, 0), 2)
        
        cv2.setMouseCallback(mask_window, draw_rectangle)
        
        while True:
            cv2.imshow(mask_window, temp_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and rect_start and rect_end:
                # Create the selection coordinates
                x1, y1 = min(rect_start[0], rect_end[0]), min(rect_start[1], rect_end[1])
                x2, y2 = max(rect_start[0], rect_end[0]), max(rect_start[1], rect_end[1])
                width = x2 - x1
                height = y2 - y1
                
                # Format the selection as requested: [x1 y1 x2 y2 width height]
                selection = [x1, y1, x2, y2, width, height]
                break
            elif key == 27:  # ESC key
                # Cancel selection
                selection = None
                break
        
        cv2.destroyAllWindows()
        return selection

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
    parser = argparse.ArgumentParser(description="Object Removal Tool")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", help="Path to save inpainted image")
    parser.add_argument("--fullscreen", action="store_true", 
                        help="Display image in fullscreen mode for selection")
    parser.add_argument("--steps", type=int, default=50, 
                        help="Number of inference steps (higher = better quality)")
    parser.add_argument("--guidance", type=float, default=9.0,
                        help="Guidance scale (higher = stronger prompt adherence)")
    parser.add_argument("--no-preserve-quality", action="store_false", dest="preserve_quality",
                        help="Disable preservation of original image quality")
    parser.add_argument("--skip-selection", action="store_true",
                        help="Skip interactive selection and use provided points")
    parser.add_argument("--points", type=int, nargs='+',
                        help="Points as a list [start_x, start_y, end_x, end_y, width, height] (only used with --skip-selection)")
    
    parser.set_defaults(preserve_quality=True)
    
    args = parser.parse_args()
    
    # Check if the image file exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found.")
        return
    
    # Set default output path if not provided
    if not args.output:
        base, ext = os.path.splitext(args.image)
        args.output = f"{base}_removed{ext}"
    
    # Step 1: Get selection points (either interactively or from command line)
    points = None
    if args.skip_selection and args.points:
        # Use provided points
        points = args.points
        print(f"Using provided selection points: {points}")
    else:
        # Initialize and run object selector for interactive selection
        print("Please select the object to remove...")
        selector = ObjectSelector()
        
        # Show in fullscreen if requested
        if args.fullscreen:
            cv2.namedWindow("Select Object", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Select Object", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        points = selector.select_rectangle(args.image)
        
        if not points:
            print("Selection cancelled. Exiting.")
            return
        
        print(f"Selected region: {points}")
    
    # Step 2: Initialize and run inpainting with the selection points
    print("Starting object removal with inpainting...")
    inpainter = HighQualityInpainter()
    
    # Use default prompts for object removal
    prompt = "completely empty space, absolutely nothing, clean blank background, perfectly clear area, transparent, no objects whatsoever, pristine empty surface, void space, complete nothingness"
    negative_prompt = "any object, thing, item, subject, content, elements, artifacts, shapes, structures, patterns, textures, features, details, distortion, noise, anything at all"
    
    # Perform inpainting
    inpainter.inpaint(
        image_path=args.image, 
        output_path=args.output, 
        points=points,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        preserve_original_quality=args.preserve_quality
    )
    
    print(f"Object removal complete. Result saved to {args.output}")

if __name__ == "__main__":
    main()