# Object Removal Tool

## Overview
This project provides an advanced object removal tool that utilizes Stable Diffusion's inpainting capabilities. Users can interactively select objects from an image and seamlessly remove them while preserving the image quality.

## Features
- **Interactive Object Selection**: Click and drag to define the area to be removed.
- **High-Quality Inpainting**: Uses Stable Diffusion to reconstruct missing areas with realistic details.
- **Preserves Original Image Quality**: Ensures minimal loss in image fidelity.
- **Command-line Support**: Provides options to automate the process without manual selection.
- **CUDA Acceleration**: Uses GPU when available for faster processing.

## Installation
### Prerequisites
Ensure you have Python installed along with the required dependencies:
```bash
pip install torch torchvision torchaudio diffusers opencv-python numpy pillow argparse
```

### Clone the Repository
```bash
git clone https://github.com/your-username/object-removal-tool.git
cd object-removal-tool
```

## Usage
### Basic Usage
To run the tool with interactive object selection:
```bash
python object_removal.py --image path/to/image.jpg
```

### Save the Output to a Specific Path
```bash
python object_removal.py --image path/to/image.jpg --output path/to/output.jpg
```

### Skip Interactive Selection (Use Predefined Points)
```bash
python object_removal.py --image path/to/image.jpg --skip-selection --points x1 y1 x2 y2 width height
```

### Adjust Processing Quality
- Increase **inference steps** for better quality:
```bash
python object_removal.py --image path/to/image.jpg --steps 100
```
- Adjust **guidance scale** to control prompt adherence:
```bash
python object_removal.py --image path/to/image.jpg --guidance 12.0
```
- Disable quality preservation for faster processing:
```bash
python object_removal.py --image path/to/image.jpg --no-preserve-quality
```

## Example
**Input Image:**  
![1](https://github.com/user-attachments/assets/c58ec431-fc94-4c56-a8e0-f2b23ae783d0)


**Selection Process:**  
![2](https://github.com/user-attachments/assets/9704f6f8-436f-428f-bdb9-56e694df67f3)


**Output Image:**  
![3](https://github.com/user-attachments/assets/e8e57084-7925-40fd-936d-79422a72b981)


## Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests.

## License
This project is licensed under the MIT License.

## Acknowledgments
- Stable Diffusion by StabilityAI
- OpenCV for image processing

