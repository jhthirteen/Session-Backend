from typing import List, Optional
from enum import Enum
from pydantic import BaseModel
from datetime import datetime

class DifficultyScore(Enum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10

class Step(BaseModel):
    step_id: str
    title: str
    time_allocated: int
    completed: bool = False
    time_spent: Optional[int] = None
    difficulty: Optional[DifficultyScore] = None

class Milestones(BaseModel):
    milestone_id: str
    title: str
    completed: bool = False
    deadline: Optional[datetime] = None
    steps: List[Step] = None

class Module(BaseModel):
    module_id: str
    title: str
    completed: bool = False
    deadline: Optional[datetime] = None
    milestones: List[Milestones]

class Curriculum(BaseModel):
    curriculum_id: str
    title: str
    vault_id: str
    completed: bool = False
    deadline: Optional[datetime] = None
    modules: List[Module]
