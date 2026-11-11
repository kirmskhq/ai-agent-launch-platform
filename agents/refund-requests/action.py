"""Turn a policy decision into a refund or decline request."""


def build_action(payload: dict, output: dict) -> dict | None:
    choice = output.get("decision")
    if choice in ("approve", "propose"):
        return {
            "tool": "issue_refund",
            "arguments": {"request_id": payload["request"]["id"], "amount": output["amount"]},
        }
    if choice == "decline":
        return {
            "tool": "decline_refund",
            "arguments": {"request_id": payload["request"]["id"], "reason": output["reason"]},
        }
    return None
