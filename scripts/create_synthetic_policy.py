"""Run from the project root to generate an ignored, fictional test PDF."""

import argparse
from pathlib import Path

import pymupdf


def main():
    parser = argparse.ArgumentParser(description="Generate a fictional development policy.")
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--compliance", action="store_true", help="Generate a policy with explicit compliance test limits.")
    variants.add_argument("--approval-scope", action="store_true", help="Generate compliance rules plus separate general travel authorisation.")
    variants.add_argument("--premium-economy", action="store_true", help="Generate a Premium economy-only policy without cabin exceptions.")
    args = parser.parse_args()
    filename = ("synthetic_premium_economy_policy.pdf" if args.premium_economy else
                "synthetic_approval_scope_policy.pdf" if args.approval_scope else
                "synthetic_compliance_policy.pdf" if args.compliance else "synthetic_policy.pdf")
    output = Path("data") / filename
    if output.exists():
        parser.error(f"{output} already exists; refusing to overwrite an existing synthetic PDF.")
    output.parent.mkdir(exist_ok=True)
    texts = (
        "FICTIONAL TRAVEL POLICY - DEVELOPMENT ONLY\n\n"
        "Flights\nOfficers should book economy class for official travel.\n"
        "Business class requires written approval before booking.",
        "FICTIONAL TRAVEL POLICY - DEVELOPMENT ONLY\n\n"
        "Accommodation\nThe fictional hotel limit is SGD 180 per night.\n"
        "Exceptions require written approval from the approving officer.",
        "FICTIONAL TRAVEL POLICY - DEVELOPMENT ONLY\n\n"
        "Approvals\nTravel approval is required before committing expenses.\n"
        "Keep flight and accommodation receipts for reimbursement.",
    )
    if args.compliance or args.approval_scope or args.premium_economy:
        texts = (
            "FICTIONAL COMPLIANCE POLICY - DEVELOPMENT ONLY\n\n"
            "Flight requirements\nFlights must use Economy class.\n"
            "Total airfare must be at most SGD 900.\n"
            "These rules apply to every destination in this fictional policy.",
            "FICTIONAL COMPLIANCE POLICY - DEVELOPMENT ONLY\n\n"
            "Accommodation requirements\nHotel nightly rates must be at most SGD 180.\n"
            "Hotels must have at most 4 stars.\n"
            "These rules apply to every destination in this fictional policy.",
            "FICTIONAL COMPLIANCE POLICY - DEVELOPMENT ONLY\n\n"
            "Conditional cabin exception\nBusiness class may be used only with verified written approval.\n"
            "This exception changes only the cabin rule. The airfare limit still applies.\n"
            "Ordinary Economy bookings within all limits need no cabin exception.\n"
            "No exception to hotel cost or star limits is provided by this policy.",
        )
    if args.premium_economy:
        # Reuse the numeric requirements but replace the ordinary cabin rule and
        # omit the entire exception page, avoiding conflicting policy provisions.
        texts = (texts[0].replace("Flights must use Economy class.", "Flights must use Premium economy class only."), texts[1])
        texts = tuple(text.replace("DEVELOPMENT ONLY", "TESTING PURPOSES ONLY") for text in texts)
    if args.approval_scope:
        texts += (
            "FICTIONAL APPROVAL SCOPE POLICY - TESTING PURPOSES ONLY\n\n"
            "General Travel Authorisation\n"
            "All business travel requires prior written approval before booking.",
        )
    with pymupdf.open() as pdf:
        for text in texts:
            pdf.new_page().insert_text((50, 60), text)
        # Exclusive creation also protects an existing file if another process
        # creates it after the earlier check.
        content = pdf.tobytes()
        try:
            with output.open("xb") as handle:
                handle.write(content)
        except FileExistsError:
            parser.error(f"{output} already exists; refusing to overwrite an existing synthetic PDF.")
    print(f"Created fictional development policy: {output}")


if __name__ == "__main__":
    main()
