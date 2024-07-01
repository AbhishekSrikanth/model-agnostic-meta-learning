from dataclasses import dataclass

@dataclass
class Range:
    min: float
    max: float

    def __post_init__(self):
        if not self.is_valid():
            raise ValueError(f"Invalid range: min={self.min}, max={self.max}")

    def is_valid(self) -> bool:
        return self.min < self.max
