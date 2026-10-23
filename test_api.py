import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError
from PIL import Image
from pydantic import BaseModel, Field

load_dotenv()

class QuickCheck(BaseModel):
    subject: str = Field(description="Primary object, component, or region detected")
    condition: str = Field(description="Visual observation, anomalies, or defects")
    severity: str = Field(description="Severity rating: low, medium, high, or critical")
    needs_action: bool = Field(description="Whether maintenance or intervention is required")

client = genai.Client()

image_path = "crop_0.jpg"
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Missing {image_path}. Run test_cv.py first.")

crop_img = Image.open(image_path)

# Exact available models from your account, starting with Gemini 4 Argon
CANDIDATE_MODELS = [
    "gemini-4-argon",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
]

response = None

for model_name in CANDIDATE_MODELS:
    for attempt in range(1, 3):
        try:
            print(f"Attempting inspection with {model_name} (Attempt {attempt})...")
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    crop_img,
                    "Perform a visual inspection on this image slice. Identify the object and note any flaws or observations.",
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=QuickCheck,
                    temperature=0.1,
                ),
            )
            print(f"Success with {model_name}!")
            break
        except ServerError:
            print(f"  [503 Busy] {model_name} is under high demand. Waiting 2 seconds...")
            time.sleep(2)
        except ClientError as e:
            print(f"  [Skipped] {model_name} error ({e.code}): {e.message}")
            break
    if response is not None:
        break

if response is None:
    raise RuntimeError("All candidate models failed or are busy.")

result = QuickCheck.model_validate_json(response.text)
print("\n--- Structured Inspection Output ---")
print(result.model_dump_json(indent=2))