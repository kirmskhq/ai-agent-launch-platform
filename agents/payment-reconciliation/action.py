"""Turn the graph's checked match into a tool request."""


def build_action(payload: dict, output: dict) -> dict | None:
    if not output.get("invoice_ids"):
        return None
    return {
        "tool": "match_statement_line",
        "arguments": {
            "line_id": payload["line"]["id"],
            "invoice_ids": output["invoice_ids"],
            "fee": output.get("fee", 0),
        },
    }
