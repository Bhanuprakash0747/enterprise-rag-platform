from dotenv import load_dotenv
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

load_dotenv()

class GenerationService:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: int = 120,
    ):
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")

        self.model = model or os.getenv(
            "OLLAMA_MODEL", "llama3.2:3b"
        )
        self.base_url = (
            base_url
            or os.getenv(
                "OLLAMA_BASE_URL",
                "http://localhost:11434",
            )
        ).rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(
                f"Ollama returned HTTP {exc.code}."
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                "Could not connect to Ollama. Check that it is running."
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout} seconds."
            ) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError(
                "Ollama returned an invalid response."
            ) from exc

        if not isinstance(result, dict):
            raise RuntimeError("Ollama returned an unexpected response format.")

        answer = result.get("response")
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Ollama returned an empty or missing answer.")

        return answer.strip()
