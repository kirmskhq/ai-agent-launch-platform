"""Scheduled run: every unmatched statement line, with the payer's open invoices."""

from agent_platform.tools import call


def collect() -> list[dict]:
    inputs = []
    for line in call("list_unmatched_statement_lines"):
        customer = call("find_customer", name=line["payer"])
        invoices = call("list_open_invoices", customer_id=customer["id"]) if customer else []
        inputs.append(
            {
                "line": {k: line[k] for k in ("id", "booked_on", "amount", "payer", "reference")},
                "open_invoices": [
                    {k: i[k] for k in ("id", "amount", "issued_on", "due_on")} for i in invoices
                ],
            }
        )
    return inputs
