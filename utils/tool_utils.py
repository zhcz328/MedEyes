import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


class MedicalImageTools:
    """
    Utility tools for medical image processing and visualization
    """

    def __init__(self):
        self.font_path = self._find_font()

    def _find_font(self) -> Optional[str]:
        """Find a suitable font for annotations"""
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:\\Windows\\Fonts\\Arial.ttf"
        ]
        for path in font_paths:
            if Path(path).exists():
                return path
        return None

    def extract_region(
            self,
            image: Union[torch.Tensor, np.ndarray, Image.Image],
            bbox: List[float],
            padding: float = 0.1
    ) -> Union[torch.Tensor, np.ndarray, Image.Image]:
        """
        Extract a region from image with optional padding

        Args:
            image: Input image
            bbox: Bounding box [x1, y1, x2, y2]
            padding: Padding ratio around the bbox

        Returns:
            Extracted region in the same format as input
        """
        # Convert to numpy for processing
        if isinstance(image, torch.Tensor):
            img_np = image.cpu().numpy()
            if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:
                img_np = np.transpose(img_np, (1, 2, 0))
            return_tensor = True
        elif isinstance(image, Image.Image):
            img_np = np.array(image)
            return_pil = True
            return_tensor = False
        else:
            img_np = image
            return_tensor = False
            return_pil = False

        h, w = img_np.shape[:2]

        # Add padding
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        x1 = max(0, int(x1 - width * padding))
        y1 = max(0, int(y1 - height * padding))
        x2 = min(w, int(x2 + width * padding))
        y2 = min(h, int(y2 + height * padding))

        # Extract region
        region = img_np[y1:y2, x1:x2]

        # Convert back to original format
        if return_tensor:
            if region.ndim == 3:
                region = np.transpose(region, (2, 0, 1))
            return torch.from_numpy(region)
        elif return_pil:
            return Image.fromarray(region)
        else:
            return region

    def zoom_region(
            self,
            image: Union[torch.Tensor, np.ndarray],
            center: Tuple[float, float],
            zoom_factor: float = 2.0,
            output_size: Optional[Tuple[int, int]] = None
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Zoom into a specific region

        Args:
            image: Input image
            center: Center point (x, y) for zooming
            zoom_factor: Zoom magnification
            output_size: Output size, defaults to input size

        Returns:
            Zoomed image
        """
        if isinstance(image, torch.Tensor):
            img_np = image.cpu().numpy()
            if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:
                img_np = np.transpose(img_np, (1, 2, 0))
            return_tensor = True
        else:
            img_np = image
            return_tensor = False

        h, w = img_np.shape[:2]
        output_size = output_size or (h, w)

        # Calculate crop region
        crop_h = h / zoom_factor
        crop_w = w / zoom_factor

        x1 = int(center[0] - crop_w / 2)
        y1 = int(center[1] - crop_h / 2)
        x2 = int(center[0] + crop_w / 2)
        y2 = int(center[1] + crop_h / 2)

        # Ensure within bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Extract and resize
        cropped = img_np[y1:y2, x1:x2]
        zoomed = cv2.resize(cropped, output_size, interpolation=cv2.INTER_LINEAR)

        if return_tensor:
            if zoomed.ndim == 3:
                zoomed = np.transpose(zoomed, (2, 0, 1))
            return torch.from_numpy(zoomed)
        else:
            return zoomed

    def apply_window_level(
            self,
            image: np.ndarray,
            window_center: float,
            window_width: float
    ) -> np.ndarray:
        """
        Apply window/level adjustment for medical images (especially CT)

        Args:
            image: Input image
            window_center: Window center value
            window_width: Window width value

        Returns:
            Windowed image
        """
        img_min = window_center - window_width / 2
        img_max = window_center + window_width / 2

        windowed = np.clip(image, img_min, img_max)
        windowed = (windowed - img_min) / (img_max - img_min)
        windowed = (windowed * 255).astype(np.uint8)

        return windowed

    def measure_distance(
            self,
            point1: Tuple[float, float],
            point2: Tuple[float, float],
            pixel_spacing: Optional[Tuple[float, float]] = None
    ) -> Dict[str, float]:
        """
        Measure distance between two points

        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
            pixel_spacing: Physical spacing per pixel (x_spacing, y_spacing)

        Returns:
            Dictionary with pixel and physical distances
        """
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]

        pixel_distance = np.sqrt(dx ** 2 + dy ** 2)

        result = {"pixel_distance": pixel_distance}

        if pixel_spacing:
            physical_dx = dx * pixel_spacing[0]
            physical_dy = dy * pixel_spacing[1]
            physical_distance = np.sqrt(physical_dx ** 2 + physical_dy ** 2)
            result["physical_distance"] = physical_distance
            result["unit"] = "mm"  # Assuming mm

        return result

    def measure_area(
            self,
            polygon: List[Tuple[float, float]],
            pixel_spacing: Optional[Tuple[float, float]] = None
    ) -> Dict[str, float]:
        """
        Measure area of a polygon

        Args:
            polygon: List of points defining the polygon
            pixel_spacing: Physical spacing per pixel

        Returns:
            Dictionary with pixel and physical areas
        """
        # Shoelace formula for polygon area
        n = len(polygon)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]
        pixel_area = abs(area) / 2.0

        result = {"pixel_area": pixel_area}

        if pixel_spacing:
            physical_area = pixel_area * pixel_spacing[0] * pixel_spacing[1]
            result["physical_area"] = physical_area
            result["unit"] = "mm²"

        return result

    def enhance_contrast(
            self,
            image: np.ndarray,
            method: str = "clahe"
    ) -> np.ndarray:
        """
        Enhance image contrast

        Args:
            image: Input image
            method: Enhancement method ('clahe', 'histogram', 'adaptive')

        Returns:
            Enhanced image
        """
        if image.dtype != np.uint8:
            # Normalize to 0-255
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

        if method == "clahe":
            # Contrast Limited Adaptive Histogram Equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            if len(image.shape) == 3:
                # Apply to each channel
                enhanced = np.stack([clahe.apply(image[:, :, i]) for i in range(image.shape[2])], axis=2)
            else:
                enhanced = clahe.apply(image)
        elif method == "histogram":
            # Simple histogram equalization
            if len(image.shape) == 3:
                enhanced = np.stack([cv2.equalizeHist(image[:, :, i]) for i in range(image.shape[2])], axis=2)
            else:
                enhanced = cv2.equalizeHist(image)
        else:
            # Adaptive contrast
            enhanced = cv2.convertScaleAbs(image, alpha=1.5, beta=0)

        return enhanced

    def detect_edges(
            self,
            image: np.ndarray,
            method: str = "canny",
            low_threshold: float = 50,
            high_threshold: float = 150
    ) -> np.ndarray:
        """
        Detect edges in medical image

        Args:
            image: Input image
            method: Edge detection method
            low_threshold: Low threshold for edge detection
            high_threshold: High threshold for edge detection

        Returns:
            Edge map
        """
        if image.dtype != np.uint8:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

        if len(image.shape) == 3:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        if method == "canny":
            edges = cv2.Canny(gray, low_threshold, high_threshold)
        elif method == "sobel":
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx ** 2 + sobely ** 2)
            edges = (edges / edges.max() * 255).astype(np.uint8)
        else:
            # Laplacian
            edges = cv2.Laplacian(gray, cv2.CV_64F)
            edges = np.absolute(edges)
            edges = (edges / edges.max() * 255).astype(np.uint8)

        return edges


class VisualizationTools:
    """
    Tools for visualizing medical images and annotations
    """

    def __init__(self):
        self.colors = self._generate_colors()

    def _generate_colors(self, n: int = 20) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for visualization"""
        colors = []
        for i in range(n):
            hue = i / n
            rgb = plt.cm.hsv(hue)[:3]
            colors.append(tuple(int(c * 255) for c in rgb))
        return colors

    def draw_bboxes(
            self,
            image: Union[np.ndarray, Image.Image],
            bboxes: List[List[float]],
            labels: Optional[List[str]] = None,
            confidences: Optional[List[float]] = None,
            thickness: int = 2
    ) -> Image.Image:
        """
        Draw bounding boxes on image

        Args:
            image: Input image
            bboxes: List of bounding boxes
            labels: Optional labels for each box
            confidences: Optional confidence scores
            thickness: Line thickness

        Returns:
            Annotated image
        """
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image.copy()

        draw = ImageDraw.Draw(img)

        # Try to load font
        try:
            font = ImageFont.truetype(self.font_path or "arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        for i, bbox in enumerate(bboxes):
            color = self.colors[i % len(self.colors)]
            x1, y1, x2, y2 = bbox

            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)

            # Draw label and confidence
            text_parts = []
            if labels and i < len(labels):
                text_parts.append(labels[i])
            if confidences and i < len(confidences):
                text_parts.append(f"{confidences[i]:.2f}")

            if text_parts:
                text = " ".join(text_parts)
                text_bbox = draw.textbbox((x1, y1), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                # Draw text background
                draw.rectangle(
                    [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
                    fill=color
                )
                draw.text((x1 + 2, y1 - text_height - 2), text, fill="white", font=font)

        return img

    def create_attention_heatmap(
            self,
            image: np.ndarray,
            attention_weights: np.ndarray,
            alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create attention heatmap overlay

        Args:
            image: Input image
            attention_weights: Attention weights
            alpha: Overlay transparency

        Returns:
            Image with heatmap overlay
        """
        # Ensure image is RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Resize attention weights to match image
        h, w = image.shape[:2]
        attention_resized = cv2.resize(attention_weights, (w, h))

        # Normalize attention
        attention_norm = (attention_resized - attention_resized.min()) / (
                    attention_resized.max() - attention_resized.min())

        # Create heatmap
        heatmap = cv2.applyColorMap((attention_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Overlay
        result = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

        return result

    def create_trajectory_visualization(
            self,
            image: np.ndarray,
            trajectory: List[Dict],
            save_path: Optional[Path] = None
    ) -> Union[np.ndarray, None]:
        """
        Visualize a complete reasoning trajectory

        Args:
            image: Input image
            trajectory: Reasoning trajectory
            save_path: Optional path to save visualization

        Returns:
            Visualization or None if saved
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        # Show original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # Visualize trajectory steps
        step_idx = 1
        for i, step in enumerate(trajectory[:5]):  # Show up to 5 steps
            if step.get('action', {}).get('action_type') == 'gaze':
                ax = axes[step_idx]
                ax.imshow(image)

                # Draw bbox
                bbox = step['action']['parameters']['coordinate']
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]),
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                    linewidth=2,
                    edgecolor='red',
                    facecolor='none'
                )
                ax.add_patch(rect)

                ax.set_title(f"Step {i + 1}: {step.get('action', {}).get('tool', 'gaze')}")
                ax.axis('off')
                step_idx += 1

        # Hide unused subplots
        for idx in range(step_idx, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return None
        else:
            # Convert to numpy array
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return buf