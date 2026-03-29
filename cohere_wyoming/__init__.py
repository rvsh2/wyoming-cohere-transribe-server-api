"""Shared runtime for Cohere Transcribe over HTTP and Wyoming."""

from .transcriber import CohereTranscriber, TranscriptionResult

__all__ = ["CohereTranscriber", "TranscriptionResult"]
