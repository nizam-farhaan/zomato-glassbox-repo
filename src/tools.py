"""A stub "order status" tool that the assistant can call.

Why: real restaurants have order systems. This one is a small in-memory
dictionary, so students can see how a tool call works without any setup.
It is a teaching stub, not a real order system.
"""

MOCK_ORDERS = {
    "ORD1001": {"status": "delivered", "eta_minutes": 0},
    "ORD1002": {"status": "out_for_delivery", "eta_minutes": 12},
    "ORD1003": {"status": "preparing", "eta_minutes": 30},
    "ORD1004": {"status": "preparing", "eta_minutes": 25},
    "ORD1005": {"status": "cancelled", "eta_minutes": 0},
}


def get_order_status(order_id: str) -> dict:
    """Return the mock status for `order_id`, or a "not_found" status if it is unknown."""
    order_id = order_id.strip().upper()
    order = MOCK_ORDERS.get(order_id)
    if order is None:
        return {"order_id": order_id, "status": "not_found", "eta_minutes": None}
    return {"order_id": order_id, **order}
