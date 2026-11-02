"""Run from the project root to generate an ignored, fictional test PDF."""

from pathlib import Path

import pymupdf


def main():
    output = Path("data") / "synthetic_policy.pdf"
    output.parent.mkdir(exist_ok=True)
    with pymupdf.open() as pdf:
        for text in (
            "FICTIONAL TRAVEL POLICY - DEVELOPMENT ONLY\n\n"
            "Flights\nOfficers should book economy class for official travel.\n"
            "Business class requires written approval before booking.",
            "FICTIONAL TRAVEL POLICY - DEVELOPMENT ONLY\n\n"
            "Accommodation\nThe fictional hotel limit is SGD 180 per night.\n"
            "Exceptions require written approval from the approving officer.",
            "FICTIONAL TRAVEL POLICY - DEVELOPMENT ONLY\n\n"
            "Approvals\nTravel approval is required before committing expenses.\n"
            "Keep flight and accommodation receipts for reimbursement.",
        ):
            pdf.new_page().insert_text((50, 60), text)
        pdf.save(output)
    print(f"Created fictional development policy: {output}")


if __name__ == "__main__":
    main()
