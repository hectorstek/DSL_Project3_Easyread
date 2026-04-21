import os
import unicodedata  # Added for character normalization
from datetime import datetime
from fpdf import FPDF
from evaluators.base_evaluator import BaseEvaluator
import config

class EasyReadPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Easy Read Document', 0, 1, 'C')
        self.ln(10)

class PDFEvaluator(BaseEvaluator):
    def evaluate(self, sentences: list[str], matched_filenames: list[list[str]]) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(config.OUTPUT_DIR, f"easy_read_{timestamp}.pdf")

        pdf = EasyReadPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Note: Standard Arial doesn't support Umlauts in FPDF without extra config.
        # If your SENTENCES have ä, ö, ü, you might need a Unicode font.
        pdf.set_font("Arial", size=24)

        PAIRS_PER_PAGE = 4
        row_height = 60

        for i, (sentence, top_k_list) in enumerate(zip(sentences, matched_filenames)):
            if i > 0 and i % PAIRS_PER_PAGE == 0:
                pdf.add_page()

            current_y = pdf.get_y()
            filename = top_k_list[0]

            # --- ENCODING FIX START ---
            # 1. Join path normally
            raw_path = os.path.join(config.IMAGE_FOLDER, filename)
            # 2. Normalize path to NFC (composed form) to match most file systems
            img_path = unicodedata.normalize('NFC', raw_path)
            # --- ENCODING FIX END ---

            if os.path.exists(img_path):
                pdf.image(img_path, x=10, y=current_y + 5, w=40)
            else:
                # Debugging print if file still not found
                print(f"CRITICAL: Image not found even after normalization: {img_path}")

            # Sanitizing the sentence for Arial (which is Latin-1 by default in FPDF)
            # This prevents the PDF from crashing if the sentence has an 'ä'
            safe_sentence = sentence.encode('latin-1', 'replace').decode('latin-1')

            pdf.set_xy(60, current_y + 15)
            pdf.multi_cell(0, 10, safe_sentence)
            pdf.set_y(current_y + row_height)

            if (i + 1) % PAIRS_PER_PAGE != 0 and (i + 1) != len(sentences):
                pdf.set_draw_color(200, 200, 200)
                pdf.line(10, pdf.get_y() - 5, 200, pdf.get_y() - 5)

        pdf.output(output_path)
        print(f"PDF successfully generated: {output_path}")
