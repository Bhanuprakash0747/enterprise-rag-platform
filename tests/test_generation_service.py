
import json
import unittest
from unittest.mock import patch
from urllib.error import URLError
from io import BytesIO

from app.rag.generation_service import GenerationService


class TestGenerationService(unittest.TestCase):

    def test_empty_prompt_is_rejected(self):
        service = GenerationService()

        with self.assertRaisesRegex(
            ValueError, "non-empty string"
        ):
            service.generate("   ")

    @patch("app.rag.generation_service.urlopen")
    def test_successful_generation(self, mock_urlopen):
        response = BytesIO(
            json.dumps({"response": "RAG is working"}).encode()
        )
        mock_urlopen.return_value.__enter__.return_value = response

        service = GenerationService()
        result = service.generate("Test prompt")

        self.assertEqual(result, "RAG is working")

    @patch(
        "app.rag.generation_service.urlopen",
        side_effect=URLError("connection refused"),
    )
    def test_connection_error(self, mock_urlopen):
        service = GenerationService()

        with self.assertRaisesRegex(
            RuntimeError, "Could not connect to Ollama"
        ):
            service.generate("Test prompt")

    @patch("app.rag.generation_service.urlopen")
    def test_empty_response_is_rejected(self, mock_urlopen):
        response = BytesIO(
            json.dumps({"response": "  "}).encode()
        )
        mock_urlopen.return_value.__enter__.return_value = response

        service = GenerationService()

        with self.assertRaisesRegex(
            RuntimeError, "empty or missing answer"
        ):
            service.generate("Test prompt")


if __name__ == "__main__":
    unittest.main()
