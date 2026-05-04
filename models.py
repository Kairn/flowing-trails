from __future__ import annotations

from pydantic import BaseModel, Field


class MusicSpec(BaseModel):
    """Shared contract: query parser output → MusicGen prompt builder / spec refiner input."""

    description: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Core musical content prompt for MusicGen text conditioning.",
    )
    genre: str | None = Field(
        default=None,
        max_length=80,
        description="VGM genre or style label, e.g. 'JRPG battle theme', 'ambient exploration'.",
    )
    mood_tags: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Mood/atmosphere descriptors. Also concatenated for CLAP text embedding.",
    )
    instruments: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Instrumentation hints, e.g. 'orchestral strings', 'chiptune synth'.",
    )
    tempo_bpm: int | None = Field(
        default=None,
        ge=40,
        le=220,
        description="Tempo guidance in BPM.",
    )
    key: str | None = Field(
        default=None,
        max_length=20,
        description="Musical key, e.g. 'C minor', 'F# major'.",
    )
    energy: str | None = Field(
        default=None,
        description="Energy level: low, medium, or high.",
        pattern=r"^(low|medium|high)$",
    )
    duration_seconds: float = Field(
        default=10.0,
        ge=5.0,
        le=30.0,
        description="Target audio duration. MusicGen practical range is 5-30s.",
    )
    style_hint: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "VGM flavor anchor — composer style, game/era reference, franchise feel. "
            "Read-only across refiner attempts; not modified during retry loop."
        ),
    )

    def to_prompt(self) -> str:
        parts: list[str] = [self.description]
        if self.genre:
            parts.append(self.genre)
        if self.style_hint:
            parts.append(self.style_hint)
        if self.instruments:
            parts.append(", ".join(self.instruments))
        if self.mood_tags:
            parts.append(", ".join(self.mood_tags))
        if self.energy:
            parts.append(f"{self.energy} energy")
        if self.tempo_bpm:
            parts.append(f"{self.tempo_bpm} bpm")
        if self.key:
            parts.append(f"in {self.key}")
        return ". ".join(parts)

    def clap_text(self) -> str:
        parts: list[str] = [self.description]
        if self.mood_tags:
            parts.extend(self.mood_tags)
        if self.genre:
            parts.append(self.genre)
        return ", ".join(parts)
