#!/usr/bin/env python3
"""
Interactive demo script for MedEyes
"""

import argparse
import logging
from pathlib import Path
import gradio as gr
import torch
import numpy as np
from PIL import Image
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from inference.predictor import MedEyesPredictor, PredictionConfig
from inference.visualization import MedEyesVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description='MedEyes Interactive Demo')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--port', type=int, default=7860,
                        help='Port for Gradio interface')
    parser.add_argument('--share', action='store_true',
                        help='Create public Gradio link')
    return parser.parse_args()


class MedEyesDemo:
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        # Load model
        config = PredictionConfig(
            model_path=checkpoint_path,
            device=device,
            return_trajectories=True,
            visualization=True
        )
        self.predictor = MedEyesPredictor(config)
        self.visualizer = MedEyesVisualizer()

        # Example questions
        self.example_questions = [
            "What abnormalities are visible in this image?",
            "Is there evidence of pneumonia?",
            "Describe the key findings in this chest X-ray.",
            "What is the most likely diagnosis?",
            "Are there any urgent findings requiring immediate attention?",
            "Compare the left and right lung fields.",
            "Is there any evidence of cardiomegaly?",
            "Describe any bone abnormalities visible."
        ]

    def predict(
            self,
            image,
            question: str,
            show_trajectory: bool = True,
            show_visualization: bool = True
    ):
        """Make prediction and return results"""
        if image is None:
            return "Please upload an image", None, None, None

        try:
            # Convert image if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            # Make prediction
            result = self.predictor.predict(
                image,
                question,
                return_visualization=show_visualization
            )

            # Format answer
            answer = result['answer']

            # Format trajectory
            trajectory_text = ""
            if show_trajectory and 'reasoning_chain' in result:
                trajectory_text = self._format_trajectory(result['reasoning_chain'])

            # Get visualization
            visualization = None
            if show_visualization and 'visualization' in result:
                visualization = result['visualization']

            # Format metrics
            metrics = f"Inference time: {result.get('inference_time', 0):.2f}s"
            if 'trajectory_summary' in result:
                summary = result['trajectory_summary']
                metrics += f"\nSteps: {summary['num_steps']}"
                metrics += f"\nRegions explored: {len(summary['regions_explored'])}"
                metrics += f"\nTools used: {', '.join(summary['tools_used']) if summary['tools_used'] else 'None'}"

            return answer, trajectory_text, visualization, metrics

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return f"Error: {str(e)}", None, None, None

    def _format_trajectory(self, reasoning_chain):
        """Format reasoning trajectory for display"""
        formatted = "### Reasoning Trajectory\n\n"

        for i, step in enumerate(reasoning_chain, 1):
            step_type = step['type']

            if step_type == 'reasoning':
                formatted += f"**Step {i} - Reasoning:**\n"
                formatted += f"{step.get('content', 'N/A')}\n\n"

            elif step_type == 'tool_call':
                formatted += f"**Step {i} - Tool Use ({step.get('tool', 'unknown')}):**\n"
                params = step.get('parameters', {})
                if 'coordinate' in params:
                    coord = params['coordinate']
                    formatted += f"Focus on region: [{coord[0]:.0f}, {coord[1]:.0f}, {coord[2]:.0f}, {coord[3]:.0f}]\n\n"
                else:
                    formatted += f"Parameters: {json.dumps(params, indent=2)}\n\n"

            elif step_type == 'answer':
                formatted += f"**Step {i} - Final Answer:**\n"
                formatted += f"{step.get('content', 'N/A')}\n\n"

        return formatted

    def analyze_image(self, image):
        """Comprehensive image analysis"""
        if image is None:
            return "Please upload an image", None

        try:
            # Convert image if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            # Perform comprehensive analysis
            analysis = self.predictor.analyze_image(image, 'comprehensive')

            # Format results
            formatted = "## Comprehensive Medical Image Analysis\n\n"

            # Synthesis
            synthesis = analysis['synthesis']
            formatted += "### Summary\n"
            formatted += f"- **Primary Findings**: {', '.join(synthesis['primary_findings']) if synthesis['primary_findings'] else 'None identified'}\n"
            formatted += f"- **Diagnostic Impression**: {', '.join(synthesis['diagnostic_impression']) if synthesis['diagnostic_impression'] else 'Inconclusive'}\n"
            formatted += f"- **Confidence Level**: {synthesis['confidence_level']:.1%}\n\n"

            # Detailed findings
            formatted += "### Detailed Analysis\n"
            for question, result in analysis['detailed_findings'].items():
                formatted += f"\n**{question}**\n"
                formatted += f"- Answer: {result['answer']}\n"
                formatted += f"- Confidence: {result['confidence']:.1%}\n"

                if result['key_regions']:
                    formatted += "- Key regions:\n"
                    for region in result['key_regions'][:3]:
                        formatted += f"  - {region['description'][:50]}...\n"

            # Recommendations
            formatted += "\n### Recommendations\n"
            for rec in analysis['recommendations']:
                formatted += f"- {rec}\n"

            # Create visualization showing all explored regions
            viz = self._create_analysis_visualization(image, analysis)

            return formatted, viz

        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            return f"Error: {str(e)}", None

    def _create_analysis_visualization(self, image, analysis):
        """Create visualization for comprehensive analysis"""
        # Extract all regions from analysis
        all_regions = []
        for result in analysis['detailed_findings'].values():
            all_regions.extend(result.get('key_regions', []))

        # Create dummy prediction for visualization
        prediction = {
            'answer': 'Comprehensive Analysis',
            'reasoning_chain': [],
            'trajectory_summary': {
                'regions_explored': [r['bbox'] for r in all_regions],
                'tools_used': ['gaze', 'analyze']
            }
        }

        # Add dummy tool calls for regions
        for region in all_regions:
            prediction['reasoning_chain'].append({
                'type': 'tool_call',
                'tool': 'gaze',
                'parameters': {'coordinate': region['bbox']}
            })

        return self.visualizer.visualize_prediction(
            np.array(image),
            prediction,
            show=False
        )


def create_gradio_interface(demo: MedEyesDemo):
    """Create Gradio interface"""

    with gr.Blocks(title="MedEyes Demo") as interface:
        gr.Markdown("""
        # MedEyes: Dynamic Visual Focus for Medical Diagnosis

        Upload a medical image and ask questions about it. The model will provide
        answers along with visual reasoning trajectories.
        """)

        with gr.Tab("Interactive Q&A"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="Medical Image",
                        type="pil"
                    )

                    question_input = gr.Textbox(
                        label="Question",
                        placeholder="Ask a question about the medical image...",
                        lines=2
                    )

                    # Example questions
                    gr.Examples(
                        examples=[[q] for q in demo.example_questions],
                        inputs=question_input,
                        label="Example Questions"
                    )

                    with gr.Row():
                        show_trajectory = gr.Checkbox(
                            label="Show Reasoning Trajectory",
                            value=True
                        )
                        show_viz = gr.Checkbox(
                            label="Show Visualization",
                            value=True
                        )

                    predict_btn = gr.Button("Ask Question", variant="primary")

                with gr.Column(scale=1):
                    answer_output = gr.Textbox(
                        label="Answer",
                        lines=4
                    )

                    metrics_output = gr.Textbox(
                        label="Metrics",
                        lines=4
                    )

            with gr.Row():
                trajectory_output = gr.Markdown(
                    label="Reasoning Trajectory",
                    visible=True
                )

            with gr.Row():
                viz_output = gr.Image(
                    label="Visual Analysis",
                    type="numpy"
                )

            # Connect predict button
            predict_btn.click(
                demo.predict,
                inputs=[image_input, question_input, show_trajectory, show_viz],
                outputs=[answer_output, trajectory_output, viz_output, metrics_output]
            )

        with gr.Tab("Comprehensive Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    analysis_image_input = gr.Image(
                        label="Medical Image",
                        type="pil"
                    )

                    analyze_btn = gr.Button("Analyze Image", variant="primary")

                with gr.Column(scale=1):
                    analysis_viz_output = gr.Image(
                        label="Analysis Visualization",
                        type="numpy"
                    )

            analysis_output = gr.Markdown(
                label="Analysis Results"
            )

            # Connect analyze button
            analyze_btn.click(
                demo.analyze_image,
                inputs=[analysis_image_input],
                outputs=[analysis_output, analysis_viz_output]
            )

        # Example images
        gr.Markdown("### Example Images")
        gr.Examples(
            examples=[
                ["examples/chest_xray_1.jpg"],
                ["examples/chest_xray_2.jpg"],
                ["examples/ct_scan_1.jpg"],
                ["examples/mri_brain_1.jpg"]
            ],
            inputs=image_input,
            label="Sample Medical Images"
        )

    return interface


def main():
    args = parse_args()

    # Create demo
    logger.info(f"Loading model from {args.checkpoint}")
    demo = MedEyesDemo(args.checkpoint, args.device)

    # Create interface
    logger.info("Creating Gradio interface...")
    interface = create_gradio_interface(demo)

    # Launch
    logger.info(f"Launching demo on port {args.port}")
    interface.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0"
    )


if __name__ == '__main__':
    main()