
# Vanishing Point Detection with Python and OpenCV

This repository contains a Python script for detecting vanishing points in images using RANSAC and the Line Segment Detector (LSD) algorithm. It identifies line intersections, filters inliers, and visualizes the vanishing point along with associated line segments.

## Features

- **Line Segment Detection**:
  - Uses OpenCV's LSD (Line Segment Detector) to identify line segments in an image.

- **Vanishing Point Estimation**:
  - Employs RANSAC to determine the vanishing point by analyzing line intersections.
  - Filters inliers based on a customizable threshold (`sigma`).

- **Least-Squares Re-Estimation**:
  - Refines the vanishing point location using least-squares with inlier line equations.

- **Visualization**:
  - Draws detected line segments and overlays the vanishing point on the original image.
  - Supports visualization on a black canvas with optional extension of lines toward the vanishing point.

## Requirements

To run the script, ensure you have the following dependencies installed:

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib

Install the required packages using pip:

```bash
pip install opencv-python numpy matplotlib
```

## Usage

1. **Prepare Your Image**:
   - Place your input image in the `ELTECar_images` directory. Ensure the file path in the script matches the image name.

2. **Run the Script**:
   - Execute the script to process your image and visualize the vanishing point:
     ```bash
     python vanishing_point_detection.py
     ```

3. **View Results**:
   - The script displays the detected vanishing point and line segments.
   - Save the output manually if needed by modifying the script.

## Input Data

The script processes a single image, detecting line segments and identifying a vanishing point. The input image should be in a supported format (e.g., JPEG or PNG) and placed in the appropriate directory.

## Output

The script generates a visual representation of the vanishing point and the inlier lines:
- **Detected Line Segments**: Visualized on the original image.
- **Vanishing Point**: Highlighted with a green dot.
- **Inliers**: Displayed as lines extending toward the vanishing point.

## Example

Here’s an example of the detection process:

1. **Input Image**:
   ![Input Image](path/to/input_image.jpg)

2. **Detected Line Segments**:
   ![image](https://github.com/user-attachments/assets/82a442ae-4c16-404c-806f-22a5af0b04f9)

3. **Vanishing Point and Inliers**:
   ![image](https://github.com/user-attachments/assets/4cb84011-c2fc-4f17-9732-4d7c19503652)

---
## Author

**Your Name**  
[GitHub Profile](https://github.com/yourusername)
