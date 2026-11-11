"""The refund policy as code. Same numbers as sandbox/refund_policy.md; a test keeps them
in step."""

from datetime import date

ACT_ALONE_UP_TO = 50.00
NOT_DELIVERED_AFTER_DAYS = 3
DAMAGED_WITHIN_DAYS = 30
CHANGED_MIND_WITHIN_DAYS = 14

REASONS = ("damaged", "duplicate_charge", "not_delivered", "changed_mind", "other")


def _days(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def precheck(request: dict, order: dict) -> dict | None:
    """Rules that need no reading of the customer's message."""
    left = round(order["amount"] - order["refunded"], 2)
    if left <= 0:
        return decline("already_refunded")
    if request["amount"] > left + 0.005:
        return decline("more_than_paid")
    return None


def decide(reason: str, request: dict, order: dict) -> dict:
    """Apply the policy once the reason for the request is known."""
    asked_on, delivered = request["requested_on"], order["delivered_on"]
    if reason == "duplicate_charge":
        same = [c for c in order["charges"] if abs(c["amount"] - order["amount"]) < 0.005]
        if len(same) < 2:
            return decline("no_duplicate_charge")
        return pay(order["amount"])
    if reason == "not_delivered":
        if delivered is not None:
            return decline("delivery_recorded")
        if _days(order["expected_by"], asked_on) < NOT_DELIVERED_AFTER_DAYS:
            return decline("not_yet_late")
        return pay(request["amount"])
    if reason in ("damaged", "changed_mind"):
        if delivered is None:
            return handover("asks about an item with no delivery recorded")
        window = DAMAGED_WITHIN_DAYS if reason == "damaged" else CHANGED_MIND_WITHIN_DAYS
        if _days(delivered, asked_on) > window:
            return decline("outside_window")
        if reason == "damaged" and not request["has_photo"]:
            return decline("photo_needed")
        return pay(request["amount"])
    return handover("not covered by the policy; needs a support specialist")


def pay(amount: float) -> dict:
    decision = "approve" if amount <= ACT_ALONE_UP_TO + 0.005 else "propose"
    return {"decision": decision, "amount": round(amount, 2), "reason": "in_policy"}


def decline(reason: str) -> dict:
    return {"decision": "decline", "amount": 0.0, "reason": reason}


def handover(why: str) -> dict:
    return {"handover": True, "reason": why}
