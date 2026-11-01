"""Grounded prompt composer (Sec. 3.2.1) and reflection prompt composer (Sec. 3.3)."""
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from google.genai import types

from .formation import FormationState, format_formation, number_word
from .prompts import (GENERATION_TEMPLATE, REFLECTION_TEMPLATE, TASK_CONTEXT_TEMPLATE,
                      VALIDATION_FEEDBACK_TEMPLATE)

_AUDIO_MIME = {".wav": "audio/wav", ".mp3": "audio/mp3", ".m4a": "audio/mp4", ".ogg": "audio/ogg",
               ".flac": "audio/flac"}


@dataclass
class TaskInput:
    """Unstructured multimodal task request I_k: text, speech, image, or any combination."""
    text: str = ""
    image_path: Optional[str] = None   # e.g. a sketch of the desired formation shape
    audio_path: Optional[str] = None   # e.g. a spoken request


def describe_task_context(state: FormationState) -> str:
    distribution, first, last = state.occupied_distribution()
    return TASK_CONTEXT_TEMPLATE.format(
        cav_number=number_word(state.num_cavs),
        lane_number=number_word(state.num_lanes, lowercase=True),
        distribution="(" + ",".join(map(str, distribution)) + ")",
        first_lane=first, last_lane=last,
        previous_formation=format_formation(state.previous_formation),
    )


def describe_user_input(task: TaskInput) -> str:
    parts = [task.text.strip()] if task.text.strip() else []
    if task.image_path:
        parts.append("(See the attached image, which depicts the desired formation shape.)")
    if task.audio_path:
        parts.append("(See the attached audio clip, which contains the spoken task request; "
                     "if it conflicts with the typed text, the audio takes priority.)")
    return " ".join(parts) or "None"


def _media_part(path: str, audio: bool) -> types.Part:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Task input file not found: {p.resolve()}")
    mime = _AUDIO_MIME.get(p.suffix.lower()) if audio else mimetypes.guess_type(p.name)[0]
    return types.Part.from_bytes(data=p.read_bytes(), mime_type=mime or "application/octet-stream")


class GroundedPromptComposer:
    def __init__(self, spacing_default: float = 2.0):
        self.spacing_default = spacing_default

    def generation_text(self, state: FormationState, task: TaskInput,
                        validation_errors: Optional[List[str]] = None) -> str:
        spacing = state.same_lane_spacing or self.spacing_default
        text = GENERATION_TEMPLATE.format(
            task_context=describe_task_context(state),
            user_task_input=describe_user_input(task),
            spacing=f"{spacing:g}",
        )
        if validation_errors:
            text += VALIDATION_FEEDBACK_TEMPLATE.format(errors="; ".join(validation_errors))
        return text

    def generation_parts(self, state: FormationState, task: TaskInput,
                         validation_errors: Optional[List[str]] = None) -> List[types.Part]:
        """Grounded input X_k (Eq. 8) before tool outputs / retrieved knowledge are added."""
        parts = [types.Part.from_text(text=self.generation_text(state, task, validation_errors))]
        if task.image_path:
            parts.append(_media_part(task.image_path, audio=False))
        if task.audio_path:
            parts.append(_media_part(task.audio_path, audio=True))
        return parts

    @staticmethod
    def reflection_text(record, outcome) -> str:
        """Reflection context C_k filled into template T_R (Fig. 3)."""
        none = lambda s: s.strip() if s and s.strip() else "None"
        return REFLECTION_TEMPLATE.format(
            task_context=record.task_context,
            user_task_input=record.user_task_input,
            retrieved_knowledge=none(record.retrieved_knowledge),
            retrieved_experience=none(record.retrieved_experience),
            invoked_tools=none(record.invoked_tools),
            tool_outputs=none(record.tool_outputs),
            generated_formation=format_formation(record.formation) if record.formation else "None",
            task_results=outcome.task_results,
            safety_checks=outcome.safety_checks,
            execution_performance=outcome.performance_text(),
        )
