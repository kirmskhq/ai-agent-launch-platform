"""Scheduled run: every pending refund request, with its order."""

from agent_platform.tools import call

REQUEST_FIELDS = ("id", "requested_on", "amount", "message", "has_photo")
ORDER_FIELDS = ("id", "amount", "charges", "expected_by", "delivered_on", "refunded")


def collect() -> list[dict]:
    inputs = []
    for request in call("list_pending_refund_requests"):
        order = call("get_order", order_id=request["order_id"])
        if order is None:
            continue
        inputs.append(
            {
                "request": {k: request[k] for k in REQUEST_FIELDS},
                "order": {k: order[k] for k in ORDER_FIELDS},
            }
        )
    return inputs
