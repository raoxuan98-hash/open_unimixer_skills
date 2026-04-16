---
name: pdf-to-images
description: Convert PDF documents to images page by page to preserve layout, formatting, equations, tables, and special symbols that may be lost or distorted in text extraction. Use when analyzing PDFs with complex formatting, mathematical equations, tables, figures, multi-column layouts, or when text extraction produces poor results due to OCR issues or encoding problems.
---

# PDF to Images Converter

Convert PDF pages to high-resolution images for visual analysis. This approach preserves the original layout, formatting, mathematical equations, tables, and special symbols that may be lost or corrupted during text extraction.

## When to Use This Skill

### Primary Use Cases

- **Complex mathematical content**: Papers with extensive equations, formulas, or mathematical notation
- **Table-heavy documents**: Financial reports, research tables, or structured data presentations
- **Multi-column layouts**: Academic papers, magazines, or newsletters with complex formatting
- **Figure-dependent analysis**: Documents where visual elements (graphs, diagrams, charts) are crucial
- **OCR-problematic PDFs**: Scanned documents, image-based PDFs, or files with encoding issues
- **Preservation of formatting**: When exact text positioning, fonts, or styling matters

### When NOT to Use

- Simple text documents where text extraction works perfectly
- Very large PDFs (>100 pages) where image storage becomes impractical
- Batch processing scenarios where text extraction is sufficient

## Quick Start

### Basic Usage

```bash
# Convert PDF to images (default: 200 DPI)
python skills/pdf-to-images/scripts/pdf_to_images.py document.pdf

# Specify output directory and resolution
python skills/pdf-to-images/scripts/pdf_to_images.py document.pdf ./output 300
```

### Output Structure

```
document.pdf
└── document_images/
    ├── page_001.png
    ├── page_002.png
    ├── page_003.png
    └── ...
```

## Installation Requirements

The script supports two conversion methods. At least one must be available:

### Option 1: pdf2image (Recommended)

Best quality, fastest for batch conversion.

```bash
# Install dependencies
pip install pdf2image
brew install poppler  # macOS
# apt-get install poppler-utils  # Ubuntu/Debian
```

### Option 2: pdfplumber (Fallback)

Pure Python, no system dependencies, but slower.

```bash
pip install pdfplumber Pillow
```

## Using with Kimi

### Workflow

1. **Convert PDF to images**
   ```bash
   python skills/pdf-to-images/scripts/pdf_to_images.py paper.pdf
   ```

2. **Read specific pages with Kimi**
   - Use `ReadMediaFile` to view individual pages
   - Analyze layout, equations, tables visually
   - Cross-reference with extracted text when needed

3. **Selective analysis**
   - Focus on problematic pages (complex tables, equations)
   - Use text extraction for simple text pages
   - Combine both approaches for comprehensive review

### Example Session

```bash
# Convert a research paper
python skills/pdf-to-images/scripts/pdf_to_images.py research_paper.pdf ./paper_images 300

# Kimi can now read specific pages
ReadMediaFile(path="./paper_images/page_001.png")  # Title page
ReadMediaFile(path="./paper_images/page_005.png")  # Complex equation
ReadMediaFile(path="./paper_images/page_012.png")  # Data table
```

## Resolution Guidelines

| Use Case | Recommended DPI | Notes |
|----------|-----------------|-------|
| Text reading | 150-200 | Fast, sufficient for most text |
| Figure analysis | 200-300 | Good balance of quality and size |
| Equations/formulas | 300+ | High detail for complex math |
| Publication quality | 300-400 | Maximum fidelity |

## Comparison: Text Extraction vs. Image Conversion

| Aspect | Text Extraction | Image Conversion |
|--------|-----------------|------------------|
| **Speed** | Fast | Slower (file I/O) |
| **Storage** | Small (text files) | Large (PNG files) |
| **Layout preservation** | Lost | Perfect |
| **Equations** | Often garbled | Preserved |
| **Tables** | May lose structure | Visual intact |
| **Searchability** | Yes | No (use OCR if needed) |
| **Copy-paste** | Easy | Requires OCR |

## Best Practices

### 1. Hybrid Approach

For most documents, combine both methods:
- Extract text for the majority of content
- Convert to images for problematic pages only

```python
# Example: Convert only pages with known issues
python pdf_to_images.py paper.pdf ./problem_pages 300
# Then analyze page_005.png, page_012.png, etc.
```

### 2. Storage Management

Images can consume significant space:
- 10-page PDF @ 300 DPI ≈ 5-15 MB of PNGs
- Clean up temporary image files after analysis
- Use lower DPI (150) for draft analysis, higher (300+) for final review

### 3. Page Selection

Instead of converting entire documents:
- Use `pdfplumber` or `PyPDF2` to identify problematic pages first
- Convert only those pages to images
- Process remaining pages with text extraction

## Troubleshooting

### Issue: "poppler not found"

**Solution**: Install poppler system package
```bash
# macOS
brew install poppler

# Ubuntu/Debian
sudo apt-get install poppler-utils

# Windows
# Download from: https://github.com/oschwartz10612/poppler-windows/releases
```

### Issue: "ImportError: No module named pdf2image"

**Solution**: Install Python dependencies
```bash
pip install pdf2image pdfplumber Pillow
```

### Issue: Images are blurry

**Solution**: Increase DPI resolution
```bash
python pdf_to_images.py document.pdf ./output 400
```

### Issue: File size too large

**Solution**: Reduce DPI or compress images
```bash
# Lower resolution
python pdf_to_images.py document.pdf ./output 150

# Or compress existing images (requires additional tools)
# pngquant --quality=70-90 page_*.png
```

## Integration with Other Skills

### Conference Reviewer Workflow

1. Convert PDF pages to images
2. Read both text extraction AND visual pages
3. Write comprehensive review with accurate figure/table references

### Document Analysis Workflow

1. Convert PDF to images
2. Use visual analysis for layout understanding
3. Extract text from images using OCR if needed
4. Combine structured data from both sources

## Technical Details

### Supported Input

- PDF version 1.0 through 2.0
- Single and multi-page documents
- Password-protected PDFs (if password provided)
- Scanned/image-based PDFs

### Output Format

- PNG (Portable Network Graphics)
- RGB color mode
- Preserves original page dimensions
- DPI metadata embedded in image

### Performance Notes

- Conversion speed: ~1-3 seconds per page (depending on DPI)
- Memory usage: ~50-200 MB per page at 300 DPI
- Temporary files: None (direct output to destination)

## Script Reference

### pdf_to_images.py

**Location**: `skills/pdf-to-images/scripts/pdf_to_images.py`

**Arguments**:
1. `input_pdf` (required): Path to input PDF file
2. `output_dir` (optional): Output directory for images
3. `dpi` (optional): Resolution in dots per inch (default: 200)

**Returns**: List of generated image file paths

**Exit Codes**:
- 0: Success
- 1: Error (file not found, conversion failed, etc.)

## Limitations

- Does not extract text (use OCR tools like pytesseract if needed)
- Output files can be large at high DPI
- No built-in image compression
- Password-protected PDFs require password input (not supported in batch mode)
- Very large PDFs (>1000 pages) may require significant memory
