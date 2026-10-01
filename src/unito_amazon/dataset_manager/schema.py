from typing import Optional

from pydantic import BaseModel


class DatasetRecord(BaseModel):
    id: str
    hate_speech: str
    counter_speech: Optional[str] = None
    edos_label: Optional[str] = None
    target: Optional[str] = None
    split: Optional[str] = None
    source: str
    label_sexist: Optional[str] = None
    label_category: Optional[str] = None
    version: Optional[str] = None
