import fitz
import os

def extract_to_md(pdf_path, output_path):
    print(f"Extracting {pdf_path} to {output_path} using fitz...")
    try:
        doc = fitz.open(pdf_path)
        text_content = f"# Original Document: {os.path.basename(pdf_path)}\n\n"
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_content += f"## Page {page_num + 1}\n\n"
            text_content += page.get_text() + "\n\n"
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text_content)
        
        print(f"Successfully extracted {len(text_content)} characters.")
        doc.close()
    except Exception as e:
        print(f"Error extracting {pdf_path}: {e}")

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    extract_to_md(os.path.join(data_dir, "BCTC.pdf"), os.path.join(data_dir, "sample_01.md"))
    extract_to_md(os.path.join(data_dir, "Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.pdf"), os.path.join(data_dir, "sample_02.md"))
