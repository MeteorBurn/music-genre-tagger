#!/usr/bin/env python3

from pathlib import Path
from typing import Dict

from openpyxl import load_workbook
from openpyxl.styles import Alignment
from openpyxl.styles import Border
from openpyxl.styles import Font
from openpyxl.styles import PatternFill
from openpyxl.styles import Side

from report import summarize_excel_dataframe
from storage import build_excel_dataframe


def auto_adjust_excel_columns(excel_path: Path) -> None:
    workbook = load_workbook(excel_path)
    worksheet = workbook.active

    header_to_column: Dict[str, str] = {}
    for column_cells in worksheet.columns:
        header_value = column_cells[0].value
        if isinstance(header_value, str) and header_value.strip():
            header_to_column[header_value.strip()] = column_cells[0].column_letter

    for column_cells in worksheet.columns:
        max_len = 0
        column_letter = column_cells[0].column_letter
        for cell in column_cells:
            value = "" if cell.value is None else str(cell.value)
            if len(value) > max_len:
                max_len = len(value)
        worksheet.column_dimensions[column_letter].width = max_len + 2

    header_fill = PatternFill(fill_type="solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    header_alignment = Alignment(horizontal="center", vertical="center")
    header_border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )
    for cell in worksheet[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = header_alignment
        cell.border = header_border

    duration_column = header_to_column.get("duration")
    if duration_column:
        for row_index in range(2, worksheet.max_row + 1):
            worksheet[f"{duration_column}{row_index}"].number_format = "[h]:mm:ss"

    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions

    workbook.save(excel_path)


def create_excel_report(db_path: Path, output_excel: Path) -> Dict[str, object]:
    dataframe = build_excel_dataframe(db_path)
    if dataframe.empty:
        raise RuntimeError(f"No analyzed tracks found in database: {db_path}")

    output_excel.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_excel(output_excel, index=False, engine="openpyxl")
    auto_adjust_excel_columns(output_excel)

    library_summary = summarize_excel_dataframe(dataframe)
    return {
        "tracks_seen": len(dataframe),
        "rows_written": len(dataframe),
        "rows_total": len(dataframe),
        "library_summary": library_summary,
    }
