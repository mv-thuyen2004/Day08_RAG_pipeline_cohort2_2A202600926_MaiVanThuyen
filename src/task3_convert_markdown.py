"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

# Configure UTF-8 encoding support for Windows terminal prints
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def extract_docx_text_fallback(docx_path: Path) -> str:
    """Fallback docx extraction: parses word/document.xml inside docx zip."""
    try:
        with zipfile.ZipFile(docx_path) as docx:
            xml_content = docx.read('word/document.xml')
            root = ET.fromstring(xml_content)
            namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            paragraphs = []
            for p in root.findall('.//w:p', namespaces):
                texts = p.findall('.//w:t', namespaces)
                p_text = "".join([t.text for t in texts if t.text])
                if p_text.strip():
                    paragraphs.append(p_text.strip())
            return "\n\n".join(paragraphs)
    except Exception as e:
        print(f"[ERROR] Fail fallback docx read for {docx_path.name}: {e}")
        return ""


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Thử import MarkItDown
    md = None
    try:
        from markitdown import MarkItDown
        md = MarkItDown()
    except ImportError:
        print("[INFO] markitdown not installed, using fallback docx parser.")

    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            output_path = output_dir / f"{filepath.stem}.md"
            
            content = ""
            # Thử dùng markitdown nếu có
            if md is not None:
                try:
                    result = md.convert(str(filepath))
                    content = result.text_content
                except Exception as e:
                    print(f"[WARN] markitdown failed for {filepath.name}: {e}. Fallback to pure python docx parser.")
            
            # Nếu chưa có content và là file docx, dùng fallback
            if not content and filepath.suffix.lower() == ".docx":
                content = extract_docx_text_fallback(filepath)
                
            # Nếu vẫn trống, ghi nhận lỗi
            if not content:
                print(f"[ERROR] Failed to convert {filepath.name}")
                continue
                
            output_path.write_text(content, encoding="utf-8")
            print(f"[OK] Saved: {output_path}")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in news_dir.iterdir():
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
                output_path = output_dir / f"{filepath.stem}.md"
                
                # Thêm metadata header
                header = f"# {data.get('title', 'Unknown')}\n\n"
                header += f"**Source:** {data.get('url', 'N/A')}\n"
                header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"
                
                content = header + data.get("content_markdown", "")
                output_path.write_text(content, encoding="utf-8")
                print(f"[OK] Saved: {output_path}")
            except Exception as e:
                print(f"[ERROR] Failed to convert JSON {filepath.name}: {e}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n[OK] Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
