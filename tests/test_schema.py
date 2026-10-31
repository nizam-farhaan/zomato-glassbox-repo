"""Checks that orders are accepted or rejected correctly, plus the mock status tool.

No network, no API key: only the local menu file is read.
"""

import pytest
from pydantic import ValidationError

from src.schema import Order, OrderItem
from src.tools import get_order_status


def test_valid_order_passes():
    order = Order(items=[OrderItem(dish_id="D01", quantity=2)], delivery_address="12 MG Road")
    assert order.items[0].dish_id == "D01"


def test_unknown_dish_is_rejected():
    with pytest.raises(ValidationError, match="not a dish on the menu"):
        OrderItem(dish_id="D99", quantity=1)


def test_zero_quantity_is_rejected():
    with pytest.raises(ValidationError):
        OrderItem(dish_id="D01", quantity=0)


def test_empty_order_is_rejected():
    with pytest.raises(ValidationError):
        Order(items=[], delivery_address="12 MG Road")


def test_known_order_status():
    assert get_order_status("ORD1002")["status"] == "out_for_delivery"


def test_unknown_order_status_is_not_found():
    assert get_order_status("ORD9999")["status"] == "not_found"
