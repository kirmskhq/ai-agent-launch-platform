"""In-memory data for Harbourline Payments: customers, invoices, bank statement lines, orders
and refund requests. `reset()` puts everything back to the seed state."""

import threading
from copy import deepcopy
from dataclasses import asdict, dataclass, field


@dataclass
class Customer:
    id: str
    name: str
    country: str
    email: str


@dataclass
class Invoice:
    id: str
    customer_id: str
    amount: float
    issued_on: str
    due_on: str
    status: str = "open"  # open | paid


@dataclass
class StatementLine:
    id: str
    booked_on: str
    amount: float
    payer: str
    reference: str
    matched_to: list[str] = field(default_factory=list)
    fee: float = 0.0


@dataclass
class Charge:
    amount: float
    charged_on: str


@dataclass
class Order:
    id: str
    customer_id: str
    item: str
    amount: float
    charges: list[Charge]
    expected_by: str
    delivered_on: str | None
    refunded: float = 0.0


@dataclass
class RefundRequest:
    id: str
    order_id: str
    requested_on: str
    amount: float
    message: str
    has_photo: bool
    status: str = "pending"  # pending | refunded | declined
    note: str = ""


CUSTOMERS = [
    Customer("C-101", "Okafor Logistics Ltd", "NG", "accounts@okafor-logistics.example"),
    Customer("C-102", "Lindqvist Design AB", "SE", "finance@lindqvist.example"),
    Customer("C-103", "Tanaka Foods KK", "JP", "ap@tanakafoods.example"),
    Customer("C-104", "Moreau et Fils SARL", "FR", "compta@moreau.example"),
    Customer("C-105", "Patel Electronics", "GB", "payables@patel-electronics.example"),
    Customer("C-106", "Garcia Textiles SL", "ES", "pagos@garciatextiles.example"),
    Customer("C-107", "Novak Engineering", "CZ", "billing@novak.example"),
    Customer("C-108", "Mwangi Coffee Traders", "KE", "accounts@mwangicoffee.example"),
]

INVOICES = [
    Invoice("INV-1201", "C-101", 1200.00, "2026-10-01", "2026-10-31"),
    Invoice("INV-1202", "C-101", 650.00, "2026-10-03", "2026-11-02"),
    Invoice("INV-1203", "C-102", 2480.00, "2026-10-02", "2026-11-01"),
    Invoice("INV-1204", "C-103", 315.40, "2026-10-04", "2026-11-03"),
    Invoice("INV-1205", "C-103", 315.40, "2026-10-11", "2026-11-10"),
    Invoice("INV-1206", "C-104", 990.00, "2026-10-05", "2026-11-04"),
    Invoice("INV-1207", "C-105", 4100.00, "2026-10-06", "2026-11-05"),
    Invoice("INV-1208", "C-106", 760.25, "2026-10-07", "2026-11-06"),
    Invoice("INV-1209", "C-106", 1239.75, "2026-10-09", "2026-11-08"),
    Invoice("INV-1210", "C-107", 5600.00, "2026-10-10", "2026-11-09"),
    Invoice("INV-1211", "C-108", 830.00, "2026-10-12", "2026-11-11"),
    Invoice("INV-1190", "C-102", 1500.00, "2026-09-01", "2026-10-01", status="paid"),
]

STATEMENT_LINES = [
    StatementLine("BSL-5001", "2026-10-20", 2480.00, "LINDQVIST DESIGN AB", "INV-1203"),
    StatementLine(
        "BSL-5002", "2026-10-21", 1837.50, "Okafor Logistics", "inv 1201 + 1202 minus fee"
    ),
    StatementLine("BSL-5003", "2026-10-21", 315.40, "Tanaka Foods K.K.", "October order"),
    StatementLine("BSL-5004", "2026-10-22", 975.00, "MOREAU ET FILS", "Facture 1206 frais 15"),
    StatementLine("BSL-5005", "2026-10-23", 2000.00, "Patel Electronics", "part payment 1207"),
    StatementLine("BSL-5006", "2026-10-24", 2000.00, "Garcia Textiles SL", "invoices 1208/1209"),
    StatementLine("BSL-5007", "2026-10-24", 120.00, "J. Smith", "thanks"),
    StatementLine("BSL-5008", "2026-10-25", 830.00, "Mwangi Coffee Traders", "INV1211"),
]

ORDERS = [
    Order(
        "ORD-7001",
        "C-105",
        "Card reader, 2 units",
        48.00,
        [Charge(48.00, "2026-10-01")],
        "2026-10-06",
        "2026-10-05",
    ),
    Order(
        "ORD-7002",
        "C-102",
        "Receipt printer",
        189.00,
        [Charge(189.00, "2026-10-02")],
        "2026-10-08",
        "2026-10-07",
    ),
    Order(
        "ORD-7003",
        "C-108",
        "Paper rolls, box of 50",
        36.50,
        [Charge(36.50, "2026-10-03"), Charge(36.50, "2026-10-03")],
        "2026-10-09",
        "2026-10-08",
    ),
    Order(
        "ORD-7004",
        "C-104",
        "Payment terminal",
        320.00,
        [Charge(320.00, "2026-10-04")],
        "2026-10-10",
        None,
    ),
    Order(
        "ORD-7005",
        "C-106",
        "Terminal stand",
        29.00,
        [Charge(29.00, "2026-09-10")],
        "2026-09-15",
        "2026-09-14",
    ),
    Order(
        "ORD-7006",
        "C-103",
        "Spare battery pack",
        22.00,
        [Charge(22.00, "2026-10-12")],
        "2026-10-16",
        "2026-10-15",
    ),
]

REFUND_REQUESTS = [
    RefundRequest(
        "RR-9001",
        "ORD-7001",
        "2026-10-09",
        48.00,
        "One of the card readers arrived with a cracked screen, photo attached.",
        True,
    ),
    RefundRequest(
        "RR-9002",
        "ORD-7003",
        "2026-10-10",
        36.50,
        "We were charged twice for the same box of paper rolls.",
        False,
    ),
    RefundRequest(
        "RR-9003",
        "ORD-7004",
        "2026-10-20",
        320.00,
        "The terminal never showed up and tracking has not moved in ten days.",
        False,
    ),
    RefundRequest(
        "RR-9004",
        "ORD-7005",
        "2026-10-25",
        29.00,
        "We no longer need the stand, please take it back.",
        False,
    ),
    RefundRequest(
        "RR-9005",
        "ORD-7002",
        "2026-10-12",
        189.00,
        "Printer works but the paper feed jams now and then.",
        False,
    ),
    RefundRequest(
        "RR-9006",
        "ORD-7006",
        "2026-10-20",
        22.00,
        "Changed our minds about the spare battery, unopened.",
        False,
    ),
]


class Store:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self.lock:
            self.customers = {c.id: deepcopy(c) for c in CUSTOMERS}
            self.invoices = {i.id: deepcopy(i) for i in INVOICES}
            self.lines = {s.id: deepcopy(s) for s in STATEMENT_LINES}
            self.orders = {o.id: deepcopy(o) for o in ORDERS}
            self.refunds = {r.id: deepcopy(r) for r in REFUND_REQUESTS}


STORE = Store()


def as_dict(obj) -> dict:
    return asdict(obj)
