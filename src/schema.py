"""Pydantic models that check a food order before the app accepts it.

Why: a language model can produce messy text, so an order is only trusted once
it passes these rules: every dish must exist in menu.json, and every quantity
must be at least 1. Pydantic reports exactly which field failed and why.
"""

import json
from functools import lru_cache

from pydantic import BaseModel, Field, field_validator

from src.config import MENU_PATH


@lru_cache(maxsize=1)
def _menu_ids() -> frozenset[str]:
    """Return the set of valid dish IDs, read once from menu.json."""
    with open(MENU_PATH, encoding="utf-8") as f:
        return frozenset(dish["id"] for dish in json.load(f))


class OrderItem(BaseModel):
    dish_id: str
    quantity: int = Field(ge=1)
    notes: str | None = None

    @field_validator("dish_id")
    @classmethod
    def dish_must_exist_on_menu(cls, value: str) -> str:
        if value not in _menu_ids():
            raise ValueError(f"'{value}' is not a dish on the menu. Valid IDs run from D01 to D20.")
        return value


class Order(BaseModel):
    items: list[OrderItem] = Field(min_length=1)
    delivery_address: str
