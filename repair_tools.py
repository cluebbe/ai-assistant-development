"""Task 7: tools the assistant may call. Plain Python functions.

The model never runs code itself. It asks for a function call, the SDK runs
the function here on your computer, and sends the result back to the model.
"""

import re
from datetime import datetime
from zoneinfo import ZoneInfo

# Pretend database. In a real shop, this would come from the repair software.
REPAIRS = {
    "KW-1042": {"status": "ready for pickup", "work": "full service", "price_eur": 89},
    "KW-1043": {"status": "waiting for parts", "work": "new rear wheel",
                "expected_ready": "2026-11-06"},
    "KW-1044": {"status": "in progress", "work": "brake adjustment",
                "expected_ready": "2026-10-29"},
}


def get_repair_status(ticket_id: str) -> dict:
    """Look up the current status of a customer's repair.

    ticket_id is the ticket number from the customer's receipt, for example
    KW-1042. Ask the customer for it if they have not given it.
    """
    ticket = ticket_id.strip().upper()
    if not re.fullmatch(r"KW-\d{4}", ticket):
        return {"error": "Ticket numbers look like KW-1042. Ask the customer to check the receipt."}
    if ticket not in REPAIRS:
        return {"error": f"There is no repair with the ticket number {ticket}."}
    return {"ticket_id": ticket, **REPAIRS[ticket]}


def get_current_date_and_time() -> dict:
    """Return today's date, the weekday and the current time in Musterstadt.

    Use it for questions like "Are you open today?" or "Is it ready by Friday?".
    """
    now = datetime.now(ZoneInfo("Europe/Berlin"))
    return {"date": now.strftime("%Y-%m-%d"), "weekday": now.strftime("%A"),
            "time": now.strftime("%H:%M")}


TOOLS = [get_repair_status, get_current_date_and_time]
