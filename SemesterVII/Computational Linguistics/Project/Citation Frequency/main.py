import pymupdf
import re
from collections import defaultdict
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

# ==================================================
# SETTINGS
# ==================================================

PDF_PATH = "input/testReport.pdf"
OUTPUT_PATH = "output/citation_report.pdf"

citation_pattern = r"\[(\d+(?:\s*[-,]\s*\d+)*)\]"
reference_pattern = r"\[(\d+)\]"


# ==================================================
# CITATION RANGE EXPANSION
# ==================================================

def expand_citation(citation):
    numbers = []

    for part in citation.split(","):
        part = part.strip()

        if "-" in part:
            start, end = part.split("-", 1)
            start = int(start.strip())
            end = int(end.strip())
            numbers.extend(range(start, end + 1))
        else:
            numbers.append(int(part))

    return numbers


# ==================================================
# FIND REFERENCES SECTION
# ==================================================

def find_reference_page(pdf):
    for page_number, page in enumerate(pdf):
        text = page.get_text()

        if re.search(r"\bREFERENCES\b", text, re.IGNORECASE):
            return page_number

    return None


# ==================================================
# EXTRACT REFERENCES
# ==================================================

def extract_references(pdf, reference_page):
    reference_text = ""

    for page_number in range(reference_page, len(pdf)):
        text = pdf[page_number].get_text()

        # Prevent content after the bibliography from being
        # swallowed into the final reference.
        if page_number > reference_page:
            if re.search(
                r"\b(Attention Visualizations|Appendix|Acknowledgments)\b",
                text,
                re.IGNORECASE,
            ):
                break

        reference_text += text + "\n"

    matches = list(re.finditer(reference_pattern, reference_text))
    references = {}

    for i, match in enumerate(matches):
        reference_number = int(match.group(1))

        start = match.end()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(reference_text)

        reference_content = reference_text[start:end]

        # Normalize PDF line breaks and whitespace.
        reference_content = " ".join(reference_content.split())

        # Remove stray PDF page numbers at the end.
        reference_content = re.sub(r"\s+\d+\s*$", "", reference_content)

        references[reference_number] = reference_content.strip()

    return references


# ==================================================
# EXTRACT PAPER TITLE
# ==================================================

def extract_title(reference):
    """
    Heuristic title extraction for common IEEE/bibliography formats.

    The complete reference is retained separately, so if the heuristic
    cannot confidently identify a title, the full reference is used.
    """

    text = reference.strip()

    # Normalize common PDF extraction artifacts.
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?i)arXivpreprint", "arXiv preprint", text)

    # Find the author -> title boundary.
    boundary = None

    # Multi-author references ending with "and <author>."
    matches = list(
        re.finditer(
            r"\b(?:and|&)\s+.*?([A-Za-zÀ-ÿ]{2,})\.\s+(?=[A-ZÀ-ÖØ-Þ])",
            text,
        )
    )

    if matches:
        boundary = matches[-1].end()

    # References ending with "et al.".
    if boundary is None:
        match = re.search(r"\bet al\.\s+(?=[A-ZÀ-ÖØ-Þ])", text)
        if match:
            boundary = match.end()

    # Single-author references such as "Alex Graves. Title..."
    if boundary is None:
        match = re.search(
            r"^[^.!?]+?\.\s+(?=[A-ZÀ-ÖØ-Þ])",
            text,
        )
        if match:
            boundary = match.end()

    if boundary is None:
        return text

    tail = text[boundary:].strip()

    # Find where the title ends and publication information begins.
    end = len(tail)

    publication_patterns = [
        r"\?\s+In\b",
        r"\.\s+In\b",
        r"\.\s+arXiv\b",
        r"\.\s+CoRR\b",
        r"\.\s+Journal\b",
        r"\.\s+Neural Computation\b",
        r"\.\s+Scientific reports\b",
        r"\.\s+Sensors\b",
        r"\.\s+PloS\b",
        r"\.\s+Applied Artificial Intelligence\b",
        r"\.\s+The International Journal\b",
        r"\.\s+International Journal\b",
        r"\.\s+Computational linguistics\b",
        r"\.\s+CS231A\b",
        r"\.\s+IEEE\b",
    ]

    for pattern in publication_patterns:
        match = re.search(pattern, tail, re.IGNORECASE)
        if match:
            end = min(end, match.start())

    # If publication information was not detected, use the year.
    year_match = re.search(r",\s*(?:19|20)\d{2}\b", tail)

    if year_match:
        # Only use the year if it occurs after a plausible title.
        end = min(end, year_match.start())

    title = tail[:end].strip(" .")

    # Remove accidental trailing punctuation.
    title = title.rstrip(" .")

    # A title should normally contain at least two words.
    if len(title.split()) >= 2:
        return title

    return text


# ==================================================
# FIND CITATION OCCURRENCES
# ==================================================

def find_citations(pdf, reference_page):
    citation_data = defaultdict(list)

    for page_number in range(reference_page):
        page = pdf[page_number]
        text = page.get_text()
        lines = text.splitlines()

        for line_number, line in enumerate(lines, start=1):
            matches = re.finditer(citation_pattern, line)

            for match in matches:
                raw_citation = match.group(1)
                citation_numbers = expand_citation(raw_citation)

                for citation_number in citation_numbers:
                    citation_data[citation_number].append(
                        {
                            "raw": f"[{raw_citation}]",
                            "page": page_number + 1,
                            "line": line_number,
                        }
                    )

    return citation_data


# ==================================================
# COMBINE DATA
# ==================================================

def build_analysis(references, citation_data):
    analysis = {}

    for reference_number in sorted(references):
        locations = citation_data.get(reference_number, [])

        analysis[reference_number] = {
            "title": extract_title(references[reference_number]),
            "reference": references[reference_number],
            "occurrences": len(locations),
            "locations": locations,
        }

    return analysis


# ==================================================
# PDF REPORT
# ==================================================

def generate_report(analysis, input_filename):
    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=9,
        textColor=colors.grey,
        spaceAfter=12,
    )

    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=8,
        spaceAfter=8,
    )

    cell_style = ParagraphStyle(
        "Cell",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
    )

    cell_center_style = ParagraphStyle(
        "CellCenter",
        parent=cell_style,
        alignment=TA_CENTER,
    )

    small_style = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        fontSize=7.5,
        leading=9,
    )

    story = []

    total_references = len(analysis)
    total_occurrences = sum(
        item["occurrences"] for item in analysis.values()
    )
    cited_references = sum(
        1 for item in analysis.values() if item["occurrences"] > 0
    )
    uncited_references = total_references - cited_references

    story.append(Paragraph("IEEE CITATION ANALYZER", title_style))
    story.append(
        Paragraph(
            f"Analysis report for <b>{input_filename}</b>",
            subtitle_style,
        )
    )

    summary_data = [
        [
            Paragraph("<b>Total References</b>", cell_center_style),
            Paragraph("<b>Cited References</b>", cell_center_style),
            Paragraph("<b>Uncited References</b>", cell_center_style),
            Paragraph("<b>Total Citations</b>", cell_center_style),
        ],
        [
            str(total_references),
            str(cited_references),
            str(uncited_references),
            str(total_occurrences),
        ],
    ]

    summary_table = Table(
        summary_data,
        colWidths=[42 * mm] * 4,
        repeatRows=1,
    )

    summary_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story.append(summary_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Citation Summary", section_style))

    table_data = [
        [
            Paragraph("<b>Reference No.</b>", cell_center_style),
            Paragraph("<b>Paper Name</b>", cell_style),
            Paragraph("<b>Occurrences</b>", cell_center_style),
        ]
    ]

    for number, data in analysis.items():
        table_data.append(
            [
                Paragraph(f"[{number}]", cell_center_style),
                Paragraph(data["title"], cell_style),
                Paragraph(str(data["occurrences"]), cell_center_style),
            ]
        )

    main_table = Table(
        table_data,
        colWidths=[25 * mm, 125 * mm, 25 * mm],
        repeatRows=1,
        splitByRow=1,
    )

    main_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E2F3")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("ALIGN", (2, 0), (2, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    story.append(main_table)

    # --------------------------------------------------
    # Detailed locations
    # --------------------------------------------------

    story.append(PageBreak())
    story.append(Paragraph("Citation Locations", section_style))

    for number, data in analysis.items():
        story.append(
            Paragraph(
                f"<b>[{number}] {data['title']}</b>",
                cell_style,
            )
        )

        if data["locations"]:
            location_rows = [
                [
                    Paragraph("<b>Page</b>", cell_center_style),
                    Paragraph("<b>Line</b>", cell_center_style),
                    Paragraph("<b>Citation</b>", cell_center_style),
                ]
            ]

            for location in data["locations"]:
                location_rows.append(
                    [
                        str(location["page"]),
                        str(location["line"]),
                        location["raw"],
                    ]
                )

            location_table = Table(
                location_rows,
                colWidths=[25 * mm, 25 * mm, 35 * mm],
                repeatRows=1,
            )

            location_table.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                        (
                            "BACKGROUND",
                            (0, 0),
                            (-1, 0),
                            colors.HexColor("#EEF2F7"),
                        ),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )

            story.append(location_table)
        else:
            story.append(
                Paragraph(
                    "No citation occurrence detected in the paper body.",
                    small_style,
                )
            )

        story.append(Spacer(1, 8))

    doc.build(story)


# ==================================================
# MAIN PROGRAM
# ==================================================

def main():
    pdf = pymupdf.open(PDF_PATH)

    reference_page = find_reference_page(pdf)

    if reference_page is None:
        print("Could not find REFERENCES section.")
        return

    print("References start on page:", reference_page + 1)

    references = extract_references(pdf, reference_page)
    citation_data = find_citations(pdf, reference_page)
    analysis = build_analysis(references, citation_data)

    print("\n========== FINAL ANALYSIS ==========\n")

    for number, data in analysis.items():
        print(
            f"[{number}] {data['title']} "
            f"→ {data['occurrences']} occurrences"
        )

    generate_report(
        analysis,
        PDF_PATH.split("/")[-1],
    )

    print("\n===================================")
    print("Report generated successfully!")
    print("Output:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
