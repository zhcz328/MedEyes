import json
import random
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import re


class MedicalPromptGenerator:
    """
    Generate specialized prompts for medical VQA tasks
    """

    def __init__(self, templates_path: Optional[Path] = None):
        self.templates = self._load_templates(templates_path)
        self.medical_terms = self._load_medical_vocabulary()

    def _load_templates(self, templates_path: Optional[Path]) -> Dict[str, List[str]]:
        """Load prompt templates"""
        if templates_path and templates_path.exists():
            with open(templates_path, 'r') as f:
                return json.load(f)
        else:
            # Default templates
            return {
                "scanning": [
                    "Please identify and locate all abnormal or pathological regions in this medical image.",
                    "Scan the image systematically to detect any areas of clinical concern.",
                    "Examine the entire image and highlight regions that require further investigation.",
                    "Perform a comprehensive visual assessment to identify abnormalities."
                ],
                "drilling": [
                    "Analyze the highlighted region in detail for specific pathological features.",
                    "Provide a detailed assessment of the abnormality in the specified area.",
                    "Examine the region of interest and describe the clinical findings.",
                    "Focus on the marked area and evaluate its diagnostic significance."
                ],
                "diagnosis": [
                    "Based on the visual findings, what is the most likely diagnosis?",
                    "Considering the identified abnormalities, provide a differential diagnosis.",
                    "What clinical condition is suggested by these imaging findings?",
                    "Analyze the image features and suggest the probable pathology."
                ],
                "comparison": [
                    "Compare the identified regions and determine which shows more severe pathology.",
                    "Evaluate the relative clinical significance of the detected abnormalities.",
                    "Which region exhibits the most concerning features?",
                    "Assess the comparative severity of the identified findings."
                ]
            }

    def _load_medical_vocabulary(self) -> Dict[str, List[str]]:
        """Load medical terminology"""
        return {
            "anatomical_terms": [
                "anterior", "posterior", "superior", "inferior", "medial", "lateral",
                "proximal", "distal", "cranial", "caudal", "ventral", "dorsal"
            ],
            "pathological_terms": [
                "lesion", "mass", "nodule", "opacity", "consolidation", "infiltrate",
                "edema", "hemorrhage", "necrosis", "inflammation", "atrophy", "hypertrophy"
            ],
            "descriptive_terms": [
                "homogeneous", "heterogeneous", "well-defined", "ill-defined",
                "regular", "irregular", "focal", "diffuse", "bilateral", "unilateral"
            ],
            "severity_terms": [
                "mild", "moderate", "severe", "minimal", "extensive", "marked"
            ]
        }

    def generate_scanning_prompt(
            self,
            modality: Optional[str] = None,
            body_part: Optional[str] = None
    ) -> str:
        """Generate scanning mode prompt"""
        base_prompt = random.choice(self.templates["scanning"])

        # Add modality-specific context
        if modality:
            modality_context = {
                "xray": "Pay attention to bone structures and soft tissue densities.",
                "ct": "Examine all visible slices for abnormalities.",
                "mri": "Consider signal intensities across different sequences.",
                "ultrasound": "Evaluate echogenicity and acoustic shadows."
            }
            if modality.lower() in modality_context:
                base_prompt += f" {modality_context[modality.lower()]}"

        # Add body part specific guidance
        if body_part:
            base_prompt += f" Focus particularly on {body_part} anatomy."

        return base_prompt + " <SEG>"

    def generate_drilling_prompt(
            self,
            region_coords: List[float],
            suspected_pathology: Optional[str] = None
    ) -> str:
        """Generate drilling mode prompt"""
        base_prompt = random.choice(self.templates["drilling"])

        # Format coordinates
        coord_str = f"[{region_coords[0]:.1f}, {region_coords[1]:.1f}, {region_coords[2]:.1f}, {region_coords[3]:.1f}]"
        prompt = base_prompt.replace("the highlighted region", f"region {coord_str}")
        prompt = prompt.replace("the specified area", f"region {coord_str}")
        prompt = prompt.replace("the marked area", f"region {coord_str}")

        # Add suspected pathology context
        if suspected_pathology:
            prompt += f" Consider the possibility of {suspected_pathology}."

        return prompt + " <SEG>"

    def generate_diagnostic_prompt(
            self,
            findings: List[str],
            clinical_context: Optional[str] = None
    ) -> str:
        """Generate diagnostic reasoning prompt"""
        base_prompt = random.choice(self.templates["diagnosis"])

        # Add findings summary
        if findings:
            findings_text = "Key findings include: " + ", ".join(findings[:3])
            prompt = f"{findings_text} {base_prompt}"
        else:
            prompt = base_prompt

        # Add clinical context
        if clinical_context:
            prompt += f" Clinical context: {clinical_context}"

        return prompt

    def generate_comparison_prompt(
            self,
            regions: List[Dict[str, float]],
            comparison_aspect: str = "severity"
    ) -> str:
        """Generate comparison prompt for multiple regions"""
        base_prompt = random.choice(self.templates["comparison"])

        # Customize based on comparison aspect
        aspect_modifiers = {
            "severity": "in terms of clinical severity",
            "size": "based on size and extent",
            "characteristics": "regarding morphological characteristics",
            "urgency": "in terms of clinical urgency"
        }

        if comparison_aspect in aspect_modifiers:
            base_prompt += f" {aspect_modifiers[comparison_aspect]}"

        return base_prompt

    def generate_tool_instruction(
            self,
            tool_name: str,
            parameters: Optional[Dict] = None
    ) -> str:
        """Generate instruction for tool usage"""
        instructions = {
            "gaze": "Use the gaze tool to focus on specific regions: <action>{\"name\": \"gaze\", \"coordinate\": [x1, y1, x2, y2]}</action>",
            "zoom": "Apply zoom for detailed examination: <action>{\"name\": \"zoom\", \"coordinate\": [x1, y1, x2, y2], \"level\": 2.0}</action>",
            "segment": "Request segmentation of anatomical structures: <action>{\"name\": \"segment\", \"target\": \"organs\"}</action>",
            "measure": "Perform measurements: <action>{\"name\": \"measure\", \"type\": \"distance\", \"points\": [[x1, y1], [x2, y2]]}</action>"
        }

        if tool_name in instructions:
            instruction = instructions[tool_name]
            if parameters:
                # Customize with specific parameters
                for key, value in parameters.items():
                    instruction = instruction.replace(f'"{key}": ...', f'"{key}": {json.dumps(value)}')
            return instruction
        else:
            return f"Use the {tool_name} tool with appropriate parameters."

    def generate_report_template(
            self,
            findings: List[Dict],
            diagnosis: str,
            recommendations: Optional[List[str]] = None
    ) -> str:
        """Generate structured medical report template"""
        report = "MEDICAL IMAGE ANALYSIS REPORT\n"
        report += "=" * 40 + "\n\n"

        # Findings section
        report += "FINDINGS:\n"
        for i, finding in enumerate(findings, 1):
            location = finding.get('location', 'unspecified')
            description = finding.get('description', 'abnormality noted')
            severity = finding.get('severity', 'unspecified')
            report += f"{i}. {location}: {description} ({severity} severity)\n"

        # Impression section
        report += f"\nIMPRESSION:\n{diagnosis}\n"

        # Recommendations section
        if recommendations:
            report += "\nRECOMMENDATIONS:\n"
            for i, rec in enumerate(recommendations, 1):
                report += f"{i}. {rec}\n"

        return report

    def enhance_question_with_context(
            self,
            question: str,
            patient_info: Optional[Dict] = None,
            prior_findings: Optional[List[str]] = None
    ) -> str:
        """Enhance question with clinical context"""
        enhanced = question

        # Add patient information
        if patient_info:
            age = patient_info.get('age')
            gender = patient_info.get('gender')
            history = patient_info.get('history')

            context_parts = []
            if age:
                context_parts.append(f"{age}-year-old")
            if gender:
                context_parts.append(gender)
            if history:
                context_parts.append(f"with history of {history}")

            if context_parts:
                patient_context = " ".join(context_parts)
                enhanced = f"For a {patient_context}, {question.lower()}"

        # Add prior findings context
        if prior_findings:
            prior_context = "Previous imaging showed " + ", ".join(prior_findings[:2])
            enhanced += f" Note: {prior_context}."

        return enhanced


class ChainOfThoughtPromptBuilder:
    """
    Build chain-of-thought prompts for medical reasoning
    """

    def __init__(self):
        self.reasoning_steps = []
        self.current_findings = []

    def add_observation(self, observation: str, confidence: float = 1.0):
        """Add an observation to the reasoning chain"""
        step = {
            "type": "observation",
            "content": observation,
            "confidence": confidence
        }
        self.reasoning_steps.append(step)

    def add_reasoning(self, reasoning: str):
        """Add a reasoning step"""
        step = {
            "type": "reasoning",
            "content": reasoning
        }
        self.reasoning_steps.append(step)

    def add_tool_use(self, tool: str, parameters: Dict, result: str):
        """Add tool usage to the chain"""
        step = {
            "type": "tool_use",
            "tool": tool,
            "parameters": parameters,
            "result": result
        }
        self.reasoning_steps.append(step)

    def add_conclusion(self, conclusion: str):
        """Add conclusion to the chain"""
        step = {
            "type": "conclusion",
            "content": conclusion
        }
        self.reasoning_steps.append(step)

    def build_prompt(self) -> str:
        """Build the complete chain-of-thought prompt"""
        prompt_parts = []

        for i, step in enumerate(self.reasoning_steps):
            if step["type"] == "observation":
                conf_str = f" (confidence: {step['confidence']:.2f})" if step['confidence'] < 1.0 else ""
                prompt_parts.append(f"<observation>Step {i + 1}: {step['content']}{conf_str}</observation>")
            elif step["type"] == "reasoning":
                prompt_parts.append(f"<reasoning>Therefore: {step['content']}</reasoning>")
            elif step["type"] == "tool_use":
                prompt_parts.append(
                    f"<action>Using {step['tool']} with {step['parameters']}</action>\n"
                    f"<result>{step['result']}</result>"
                )
            elif step["type"] == "conclusion":
                prompt_parts.append(f"<conclusion>{step['content']}</conclusion>")

        return "\n".join(prompt_parts)

    def reset(self):
        """Reset the prompt builder"""
        self.reasoning_steps = []
        self.current_findings = []