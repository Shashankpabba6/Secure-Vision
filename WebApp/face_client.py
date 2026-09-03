"""
Unified face verification client supporting multiple providers.
Supports: OpenRouter (vision models), Roboflow (specialized models)
"""
import base64
import os
import json
import re
import requests
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from openai import OpenAI

from config import (
    FACE_API_PROVIDER,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_VISION_MODEL,
    ROBOFLOW_API_KEY,
    ROBOFLOW_API_URL,
    ROBOFLOW_MODEL_ID,
)


class FaceVerificationClient(ABC):
    """Abstract base class for face verification providers."""
    
    @abstractmethod
    def verify_face(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Verify if a face in the image is real or spoof.
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            Dict with 'label' (real/spoof), 'confidence' (0-100), 'reasoning' (optional)
        """
        pass


def _extract_json(text: Optional[str]) -> Dict[str, Any]:
    """Extract a JSON object from model output (None, fences, or prose-safe)."""
    if not text:
        raise ValueError("model returned empty content")

    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Scan for the first balanced {...} block, tolerating prose around it
    start = cleaned.find("{")
    if start == -1:
        raise ValueError(f"no JSON object found in output: {cleaned[:120]}")
    depth = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(cleaned[start:i + 1])
    raise ValueError(f"unbalanced JSON in output: {cleaned[:120]}")


class OpenRouterFaceClient(FaceVerificationClient):
    """Face verification using OpenRouter vision models."""

    # Fallback chain, tried in order if the primary model errors.
    # Free-tier vision models verified against the OpenRouter catalog.
    FALLBACK_MODELS: List[str] = [
        "google/gemma-4-26b-a4b-it:free",
        "nvidia/nemotron-nano-12b-v2-vl:free",
    ]

    def __init__(self):
        if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_key_here":
            raise ValueError("OPENROUTER_API_KEY not configured in .env")

        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )
        self.model = OPENROUTER_VISION_MODEL

    def _candidate_models(self) -> List[str]:
        chain = [self.model] + self.FALLBACK_MODELS
        seen, out = set(), []
        for m in chain:
            if m and m not in seen:
                seen.add(m)
                out.append(m)
        return out

    def _encode_image(self, image: np.ndarray) -> str:
        """Encode image to base64."""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')

    def _query_model(self, model: str, base64_image: str, prompt: str) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300,
            temperature=0.1,
        )
        content = response.choices[0].message.content
        return _extract_json(content)

    def verify_face(self, image: np.ndarray) -> Dict[str, Any]:
        """Verify face liveness using vision model."""
        base64_image = self._encode_image(image)

        prompt = """Analyze this image for face anti-spoofing (liveness detection).

Determine if the face is REAL (live person) or SPOOF (photo, video replay, mask, deepfake, screen display).

Look for:
- Screen artifacts (moiré patterns, pixelation, reflections)
- Photo/paper edges, glare, unnatural lighting
- Mask boundaries, unnatural skin texture
- Video replay artifacts (flicker, compression)
- Lack of natural micro-movements, blinking
- Depth inconsistencies

Respond with ONLY a JSON object, no markdown, no explanation outside the JSON:
{
  "label": "real" or "spoof",
  "confidence": 0-100,
  "reasoning": "brief explanation"
}"""

        errors = []
        for model in self._candidate_models():
            try:
                result = self._query_model(model, base64_image, prompt)

                # Normalize label
                label = str(result.get('label', '')).lower()
                if label in ['real', 'live', 'genuine', 'authentic']:
                    label = 'real'
                elif label in ['spoof', 'fake', 'attack']:
                    label = 'spoof'

                return {
                    'label': label,
                    'confidence': float(result.get('confidence', 50)),
                    'reasoning': result.get('reasoning', ''),
                    'model_used': model,
                }
            except Exception as e:
                errors.append(f"{model}: {str(e)[:120]}")

        return {
            'label': 'error',
            'confidence': 0.0,
            'reasoning': "API error: " + " | ".join(errors),
        }


class RoboflowFaceClient(FaceVerificationClient):
    """Face verification using Roboflow hosted model."""
    
    def __init__(self):
        if not ROBOFLOW_API_KEY:
            raise ValueError("ROBOFLOW_API_KEY not configured in .env")
        
        self.api_url = ROBOFLOW_API_URL
        self.api_key = ROBOFLOW_API_KEY
        self.model_id = ROBOFLOW_MODEL_ID
    
    def verify_face(self, image: np.ndarray) -> Dict[str, Any]:
        """Verify face using Roboflow inference API."""
        # Convert to grayscale as expected by the model
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', gray)
        
        files = {'file': ('image.jpg', buffer.tobytes(), 'image/jpeg')}
        data = {'model_id': self.model_id}
        headers = {'Authorization': f'Bearer {self.api_key}'}
        
        try:
            response = requests.post(
                f"{self.api_url}/infer",
                files=files,
                data=data,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            predictions = result.get('predictions', [])
            if predictions:
                predictions.sort(key=lambda x: x['confidence'], reverse=True)
                top = predictions[0]
                label = top['class'].lower()
                confidence = top['confidence'] * 100
                
                return {
                    'label': label,
                    'confidence': confidence,
                    'reasoning': f"Roboflow {self.model_id} prediction"
                }
            
            return {
                'label': 'unknown',
                'confidence': 0.0,
                'reasoning': 'No predictions returned'
            }
            
        except Exception as e:
            return {
                'label': 'error',
                'confidence': 0.0,
                'reasoning': f"Roboflow API error: {str(e)}"
            }


# Provider registry
_PROVIDERS = {
    'openrouter': OpenRouterFaceClient,
    'roboflow': RoboflowFaceClient,
}


def get_face_client(provider: Optional[str] = None) -> FaceVerificationClient:
    """Get face verification client for specified provider."""
    if provider is None:
        provider = FACE_API_PROVIDER
    
    provider = provider.lower()
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(_PROVIDERS.keys())}")
    
    return _PROVIDERS[provider]()


def list_providers() -> list:
    """List available providers."""
    return list(_PROVIDERS.keys())