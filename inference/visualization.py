import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import List, Dict, Optional, Union, Tuple
from PIL import Image, ImageDraw, ImageFont
import cv2
from pathlib import Path
import io
import base64


class MedEyesVisualizer:
    """
    Visualization utilities for MedEyes predictions
    """

    def __init__(self):
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = self._generate_colors()
        self.font_path = self._find_font()

    def _generate_colors(self, n: int = 10) -> List[str]:
        """Generate distinct colors"""
        return plt.cm.tab10(np.linspace(0, 1, n))

    def _find_font(self) -> Optional[str]:
        """Find available font"""
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:\\Windows\\Fonts\\Arial.ttf"
        ]
        for path in font_paths:
            if Path(path).exists():
                return path
        return None

    def visualize_prediction(
            self,
            image: np.ndarray,
            prediction: Dict,
            save_path: Optional[Path] = None,
            show: bool = True
    ) -> Optional[np.ndarray]:
        """
        Visualize complete prediction with reasoning trajectory

        Args:
            image: Input image
            prediction: Prediction dictionary
            save_path: Optional path to save visualization
            show: Whether to display the visualization

        Returns:
            Visualization as numpy array if not saved
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, height_ratios=[2, 1, 1])

        # Main image with annotations
        ax_main = fig.add_subplot(gs[0, :])
        self._plot_main_image(ax_main, image, prediction)

        # Reasoning trajectory
        ax_trajectory = fig.add_subplot(gs[1, :])
        self._plot_reasoning_trajectory(ax_trajectory, prediction)

        # Answer and confidence
        ax_answer = fig.add_subplot(gs[2, 0])
        self._plot_answer(ax_answer, prediction)

        # Key regions
        ax_regions = fig.add_subplot(gs[2, 1])
        self._plot_key_regions(ax_regions, prediction)

        # Metrics
        ax_metrics = fig.add_subplot(gs[2, 2])
        self._plot_metrics(ax_metrics, prediction)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if not show:
                plt.close()
                return None

        if show:
            plt.show()

        # Convert to numpy array
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        if not show:
            plt.close()

        return buf

    def _plot_main_image(self, ax, image: np.ndarray, prediction: Dict):
        """Plot main image with region annotations"""
        ax.imshow(image)
        ax.set_title("Medical Image with Explored Regions", fontsize=14, weight='bold')
        ax.axis('off')

        # Extract and plot regions from trajectory
        if 'reasoning_chain' in prediction:
            regions = []
            for step in prediction['reasoning_chain']:
                if step.get('type') == 'tool_call' and step.get('tool') == 'gaze':
                    coord = step.get('parameters', {}).get('coordinate', [])
                    if len(coord) == 4:
                        regions.append(coord)

            # Plot regions with different colors
            for i, region in enumerate(regions):
                x1, y1, x2, y2 = region
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2,
                    edgecolor=self.colors[i % len(self.colors)],
                    facecolor='none',
                    alpha=0.8
                )
                ax.add_patch(rect)

                # Add region number
                ax.text(x1 + 5, y1 + 15, f"R{i + 1}",
                        color='white',
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor=self.colors[i % len(self.colors)],
                                  alpha=0.8),
                        fontsize=10,
                        weight='bold')

    def _plot_reasoning_trajectory(self, ax, prediction: Dict):
        """Plot reasoning trajectory timeline"""
        ax.set_title("Reasoning Trajectory", fontsize=12, weight='bold')

        if 'reasoning_chain' not in prediction:
            ax.text(0.5, 0.5, "No trajectory available",
                    ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return

        chain = prediction['reasoning_chain']

        # Create timeline
        y_pos = 0
        for i, step in enumerate(chain):
            x_pos = i

            if step['type'] == 'reasoning':
                marker = 'o'
                color = 'blue'
                label = 'Reasoning'
            elif step['type'] == 'tool_call':
                marker = 's'
                color = 'green'
                label = f"Tool: {step.get('tool', 'unknown')}"
            elif step['type'] == 'answer':
                marker = '*'
                color = 'red'
                label = 'Answer'
            else:
                marker = 'D'
                color = 'gray'
                label = step['type']

            ax.scatter(x_pos, y_pos, s=100, c=color, marker=marker, alpha=0.7)

            # Add text below
            text = step.get('content', '')[:50] + '...' if len(step.get('content', '')) > 50 else step.get('content',
                                                                                                           '')
            ax.text(x_pos, y_pos - 0.1, text,
                    ha='center', va='top', fontsize=8,
                    wrap=True, rotation=45)

        # Connect points
        if len(chain) > 1:
            x_coords = list(range(len(chain)))
            y_coords = [0] * len(chain)
            ax.plot(x_coords, y_coords, 'k--', alpha=0.3)

        ax.set_xlim(-0.5, len(chain) - 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel("Step")
        ax.set_yticks([])
        ax.grid(True, axis='x', alpha=0.3)

    def _plot_answer(self, ax, prediction: Dict):
        """Plot answer with confidence"""
        ax.set_title("Answer", fontsize=12, weight='bold')
        ax.axis('off')

        answer = prediction.get('answer', 'No answer')
        confidence = prediction.get('confidence', 0.0)

        # Wrap text
        wrapped_answer = self._wrap_text(answer, 30)

        ax.text(0.5, 0.7, wrapped_answer,
                ha='center', va='center',
                transform=ax.transAxes,
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7))

        # Add confidence bar
        ax.text(0.5, 0.3, f"Confidence: {confidence:.2%}",
                ha='center', va='center',
                transform=ax.transAxes,
                fontsize=10)

        # Confidence bar visualization
        bar_width = 0.6
        bar_x = 0.2
        bar_y = 0.15

        # Background
        ax.add_patch(patches.Rectangle(
            (bar_x, bar_y), bar_width, 0.1,
            transform=ax.transAxes,
            facecolor='lightgray',
            edgecolor='black'
        ))

        # Confidence fill
        ax.add_patch(patches.Rectangle(
            (bar_x, bar_y), bar_width * confidence, 0.1,
            transform=ax.transAxes,
            facecolor='green' if confidence > 0.7 else 'orange' if confidence > 0.4 else 'red',
            alpha=0.7
        ))

    def _plot_key_regions(self, ax, prediction: Dict):
        """Plot key regions summary"""
        ax.set_title("Key Regions", fontsize=12, weight='bold')
        ax.axis('off')

        if 'trajectory_summary' in prediction:
            regions = prediction['trajectory_summary'].get('regions_explored', [])

            if regions:
                text = f"Explored {len(regions)} regions\n"
                for i, region in enumerate(regions[:3]):  # Show first 3
                    text += f"R{i + 1}: [{region[0]:.0f}, {region[1]:.0f}, {region[2]:.0f}, {region[3]:.0f}]\n"
                if len(regions) > 3:
                    text += f"... and {len(regions) - 3} more"
            else:
                text = "No regions explored"
        else:
            text = "No trajectory summary"

        ax.text(0.1, 0.5, text,
                ha='left', va='center',
                transform=ax.transAxes,
                fontsize=9)

    def _plot_metrics(self, ax, prediction: Dict):
        """Plot performance metrics"""
        ax.set_title("Metrics", fontsize=12, weight='bold')
        ax.axis('off')

        metrics_text = []

        # Inference time
        if 'inference_time' in prediction:
            metrics_text.append(f"Inference: {prediction['inference_time']:.2f}s")

        # Trajectory length
        if 'reasoning_chain' in prediction:
            metrics_text.append(f"Steps: {len(prediction['reasoning_chain'])}")

        # Tools used
        if 'trajectory_summary' in prediction:
            tools = prediction['trajectory_summary'].get('tools_used', [])
            metrics_text.append(f"Tools: {', '.join(tools) if tools else 'None'}")

        text = '\n'.join(metrics_text) if metrics_text else 'No metrics available'

        ax.text(0.1, 0.5, text,
                ha='left', va='center',
                transform=ax.transAxes,
                fontsize=9)

    def _wrap_text(self, text: str, width: int) -> str:
        """Wrap text to specified width"""
        words = text.split()
        lines = []
        current_line = []

        for word in words:
            if len(' '.join(current_line + [word])) <= width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]

        if current_line:
            lines.append(' '.join(current_line))

        return '\n'.join(lines)

    def create_attention_heatmap(
            self,
            image: np.ndarray,
            attention_weights: np.ndarray,
            save_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        Create attention heatmap overlay

        Args:
            image: Input image
            attention_weights: Attention weights from model
            save_path: Optional save path

        Returns:
            Image with heatmap overlay
        """
        # Ensure image is RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Resize attention to match image
        h, w = image.shape[:2]
        attention_resized = cv2.resize(attention_weights, (w, h))

        # Normalize attention
        attention_norm = (attention_resized - attention_resized.min()) / (
                    attention_resized.max() - attention_resized.min() + 1e-8)

        # Create heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.imshow(attention_norm, cmap='jet', alpha=0.5)
        plt.colorbar(label='Attention Weight')
        plt.title('Attention Heatmap')
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)

        # Convert to array
        plt.tight_layout()
        plt.canvas.draw()
        buf = np.frombuffer(plt.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(plt.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return buf

    def create_reasoning_flow_diagram(
            self,
            reasoning_chain: List[Dict],
            save_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        Create flow diagram of reasoning process

        Args:
            reasoning_chain: Reasoning chain from prediction
            save_path: Optional save path

        Returns:
            Flow diagram as numpy array
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Node positions
        n_steps = len(reasoning_chain)
        x_positions = np.linspace(0.1, 0.9, n_steps)
        y_positions = [0.5] * n_steps

        # Draw nodes
        for i, (x, y, step) in enumerate(zip(x_positions, y_positions, reasoning_chain)):
            # Determine node style
            if step['type'] == 'reasoning':
                node_color = 'lightblue'
                node_shape = 'o'
                node_size = 1000
            elif step['type'] == 'tool_call':
                node_color = 'lightgreen'
                node_shape = 's'
                node_size = 1200
            else:  # answer
                node_color = 'lightcoral'
                node_shape = '*'
                node_size = 1500

            # Draw node
            ax.scatter(x, y, s=node_size, c=node_color,
                       marker=node_shape, edgecolors='black',
                       linewidths=2, transform=ax.transAxes)

            # Add label
            label = step['type'].replace('_', ' ').title()
            if step['type'] == 'tool_call':
                label += f"\n({step.get('tool', 'unknown')})"

            ax.text(x, y - 0.1, label,
                    ha='center', va='top',
                    transform=ax.transAxes,
                    fontsize=9,
                    weight='bold')

        # Draw arrows
        for i in range(n_steps - 1):
            ax.annotate('', xy=(x_positions[i + 1], y_positions[i + 1]),
                        xytext=(x_positions[i], y_positions[i]),
                        xycoords='axes fraction',
                        textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Reasoning Flow', fontsize=14, weight='bold')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)

        # Convert to array
        plt.tight_layout()
        plt.canvas.draw()
        buf = np.frombuffer(plt.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(plt.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return buf

    def create_comparison_visualization(
            self,
            image: np.ndarray,
            predictions: List[Dict],
            labels: List[str],
            save_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        Create comparison visualization for multiple predictions

        Args:
            image: Input image
            predictions: List of predictions to compare
            labels: Labels for each prediction
            save_path: Optional save path

        Returns:
            Comparison visualization
        """
        n_predictions = len(predictions)
        fig, axes = plt.subplots(1, n_predictions + 1, figsize=(5 * (n_predictions + 1), 5))

        if n_predictions == 1:
            axes = [axes]

        # Show original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image", fontsize=12, weight='bold')
        axes[0].axis('off')

        # Show each prediction
        for i, (pred, label) in enumerate(zip(predictions, labels)):
            ax = axes[i + 1]

            # Show image with regions
            ax.imshow(image)

            # Extract regions
            if 'reasoning_chain' in pred:
                for step in pred['reasoning_chain']:
                    if step.get('type') == 'tool_call' and step.get('tool') == 'gaze':
                        coord = step.get('parameters', {}).get('coordinate', [])
                        if len(coord) == 4:
                            x1, y1, x2, y2 = coord
                            rect = patches.Rectangle(
                                (x1, y1), x2 - x1, y2 - y1,
                                linewidth=2,
                                edgecolor='red',
                                facecolor='none'
                            )
                            ax.add_patch(rect)

            # Add answer
            answer = pred.get('answer', 'No answer')[:50] + '...' if len(pred.get('answer', '')) > 50 else pred.get(
                'answer', 'No answer')
            ax.set_title(f"{label}\n{answer}", fontsize=10, wrap=True)
            ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)

        # Convert to array
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return buf

def create_prediction_visualization(
        image: np.ndarray,
        reasoning_chain: List[Dict],
        answer: str,
        save_path: Optional[Path] = None
) -> np.ndarray:
    """
    Convenience function to create standard prediction visualization

    Args:
        image: Input image
        reasoning_chain: Reasoning chain
        answer: Final answer
        save_path: Optional save path

    Returns:
        Visualization as numpy array
    """
    visualizer = MedEyesVisualizer()

    prediction = {
        'answer': answer,
        'reasoning_chain': reasoning_chain,
        'confidence': 0.85,  # Placeholder
        'inference_time': 1.2,  # Placeholder
        'trajectory_summary': {
            'regions_explored': [],
            'tools_used': []
        }
    }

    # Extract regions and tools from chain
    for step in reasoning_chain:
        if step.get('type') == 'tool_call':
            if step.get('tool') == 'gaze' and 'parameters' in step:
                coord = step['parameters'].get('coordinate', [])
                if coord:
                    prediction['trajectory_summary']['regions_explored'].append(coord)

            tool = step.get('tool')
            if tool and tool not in prediction['trajectory_summary']['tools_used']:
                prediction['trajectory_summary']['tools_used'].append(tool)

    return visualizer.visualize_prediction(image, prediction, save_path, show=False)

def create_batch_report(
        results: List[Dict],
        output_path: Path,
        include_trajectories: bool = True
) -> None:
    """
    Create HTML report for batch predictions

    Args:
        results: List of prediction results
        output_path: Path to save HTML report
        include_trajectories: Whether to include reasoning trajectories
    """
    html_content = """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>MedEyes Batch Prediction Report</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; }
                        .prediction { border: 1px solid #ccc; margin: 20px 0; padding: 20px; }
                        .image { max-width: 400px; }
                        .answer { background-color: #f0f0f0; padding: 10px; margin: 10px 0; }
                        .trajectory { background-color: #f9f9f9; padding: 10px; margin: 10px 0; }
                        .metrics { color: #666; font-size: 0.9em; }
                        h1 { color: #333; }
                        h2 { color: #555; }
                    </style>
                </head>
                <body>
                    <h1>MedEyes Batch Prediction Report</h1>
                    <p>Generated on: {timestamp}</p>
                    <p>Total predictions: {total}</p>
                    <hr>
                """

    import datetime
    html_content = html_content.format(
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total=len(results)
    )

    for i, result in enumerate(results):
        pred_html = f"""
                    <div class="prediction">
                        <h2>Prediction {i + 1}</h2>
                        <p><strong>Question:</strong> {result.get('question', 'N/A')}</p>
                        <div class="answer">
                            <strong>Answer:</strong> {result.get('answer', 'N/A')}
                        </div>
                    """

        if include_trajectories and 'reasoning_chain' in result:
            pred_html += """
                        <div class="trajectory">
                            <strong>Reasoning Trajectory:</strong>
                            <ol>
                        """

            for step in result['reasoning_chain']:
                step_type = step.get('type', 'unknown')
                content = step.get('content', '')

                if step_type == 'tool_call':
                    content = f"Tool: {step.get('tool', 'unknown')}"

                pred_html += f"<li>{step_type}: {content[:100]}...</li>"

            pred_html += """
                            </ol>
                        </div>
                        """

        if 'inference_time' in result:
            pred_html += f"""
                        <div class="metrics">
                            <p>Inference time: {result['inference_time']:.2f}s</p>
                        </div>
                        """

        pred_html += "</div>"
        html_content += pred_html

    html_content += """
                </body>
                </html>
                """

    with open(output_path, 'w') as f:
        f.write(html_content)

