import asyncio
import unittest
from types import SimpleNamespace

import numpy as np

from cohere_wyoming.handler import CohereWyomingEventHandler
from cohere_wyoming.wyoming_protocol import Event


class CollectingHandler(CohereWyomingEventHandler):
    def __init__(self, transcriber, info_event):
        super().__init__(transcriber, info_event)
        self.events = []

    async def write_event(self, event):
        self.events.append(event)


class WyomingHandlerTests(unittest.TestCase):
    def run_async(self, coro):
        return asyncio.run(coro)

    def test_describe_returns_info_event(self):
        handler = CollectingHandler(transcriber=SimpleNamespace(), info_event=Event("describe", {"name": "info"}))
        self.run_async(handler.handle_event(Event("describe", {})))
        self.assertEqual(handler.events[0].type, "describe")

    def test_audio_stop_returns_transcript(self):
        class FakeTranscriber:
            def __init__(self):
                self.calls = []

            def transcribe_pcm(self, audio_data, *, sample_rate, language):
                self.calls.append((audio_data, sample_rate, language))
                return SimpleNamespace(text="hello wyoming", language="pl")

        transcriber = FakeTranscriber()
        handler = CollectingHandler(transcriber=transcriber, info_event=Event("describe", {}))

        pcm = (np.array([0, 1000, -1000, 500], dtype="<i2")).tobytes()
        self.run_async(handler.handle_event(Event("transcribe", {"language": "pl"})))
        self.run_async(handler.handle_event(Event("audio-start", {"rate": 16000, "width": 2, "channels": 1})))
        self.run_async(handler.handle_event(Event("audio-chunk", {"audio": pcm, "rate": 16000, "width": 2, "channels": 1})))
        self.run_async(handler.handle_event(Event("audio-stop", {})))

        self.assertEqual(handler.events[-1].type, "transcript")
        self.assertEqual(handler.events[-1].data["text"], "hello wyoming")
        _, sample_rate, language = transcriber.calls[0]
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(language, "pl")

    def test_empty_audio_returns_empty_transcript(self):
        handler = CollectingHandler(transcriber=SimpleNamespace(), info_event=Event("describe", {}))
        self.run_async(handler.handle_event(Event("transcribe", {"language": "en"})))
        self.run_async(handler.handle_event(Event("audio-stop", {})))
        self.assertEqual(handler.events[-1].data["text"], "")
