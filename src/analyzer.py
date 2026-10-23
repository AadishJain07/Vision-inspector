import time
import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError
from src.schemas import InspectionReport

load_dotenv()

CANDIDATE_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
]

class VisionInspectorAgent:
    def __init__(self):
        self.client = genai.Client()

    def analyze_crop(self, crop_bgr: np.ndarray, inspection_context: str) -> tuple[InspectionReport, str]:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)

        prompt = f"""
        Perform a structured visual inspection of this image region.
        Context: {inspection_context}.
        Identify features, objects, structural state, or any defects.
        """

        for model_name in CANDIDATE_MODELS:
            for attempt in range(1, 3):
                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=[pil_img, prompt],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=InspectionReport,
                            temperature=0.1,
                        ),
                    )
                    return InspectionReport.model_validate_json(response.text), model_name
                except ServerError:
                    time.sleep(2)
                except ClientError:
                    break

        raise RuntimeError("All candidate Gemini models are currently unavailable.")