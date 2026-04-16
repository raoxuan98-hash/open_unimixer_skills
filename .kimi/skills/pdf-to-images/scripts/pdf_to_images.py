#!/usr/bin/env python3
"""
PDF to Images Converter

Convert PDF pages to images for visual inspection and analysis.
This preserves layout, formatting, and special symbols that may be lost in text extraction.

Usage:
    python pdf_to_images.py <input_pdf> [output_dir] [dpi]

Arguments:
    input_pdf   Path to the input PDF file
    output_dir  Directory to save images (default: same as input, suffixed with "_images")
    dpi         Resolution in dots per inch (default: 200)

Example:
    python pdf_to_images.py paper.pdf
    python pdf_to_images.py paper.pdf ./output 300
"""

import sys
import os
from pathlib import Path

def convert_pdf_to_images(input_pdf, output_dir=None, dpi=200):
    """
    Convert PDF pages to PNG images.
    
    Args:
        input_pdf: Path to input PDF file
        output_dir: Output directory for images (None = auto-generate)
        dpi: Resolution in dots per inch
        
    Returns:
        List of paths to generated images
    """
    input_path = Path(input_pdf)
    
    if not input_path.exists():
        raise FileNotFoundError(f"PDF not found: {input_pdf}")
    
    if output_dir is None:
        output_dir = input_path.parent / f"{input_path.stem}_images"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try different methods in order of preference
    generated_files = []
    
    # Method 1: pdf2image (requires poppler)
    try:
        from pdf2image import convert_from_path
        print(f"Using pdf2image with dpi={dpi}...")
        
        images = convert_from_path(input_pdf, dpi=dpi)
        for i, image in enumerate(images, 1):
            output_file = output_dir / f"page_{i:03d}.png"
            image.save(output_file, "PNG")
            generated_files.append(output_file)
            print(f"  Saved: {output_file}")
        
        print(f"\n✓ Successfully converted {len(images)} pages to {output_dir}")
        return generated_files
        
    except ImportError:
        print("pdf2image not available, trying pdfplumber...")
    except Exception as e:
        print(f"pdf2image failed: {e}")
        print("Trying pdfplumber...")
    
    # Method 2: pdfplumber (fallback)
    try:
        import pdfplumber
        print(f"Using pdfplumber with resolution={dpi}...")
        
        with pdfplumber.open(input_pdf) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                output_file = output_dir / f"page_{i:03d}.png"
                
                # Convert page to image
                im = page.to_image(resolution=dpi)
                im.save(output_file)
                
                generated_files.append(output_file)
                print(f"  Saved: {output_file}")
        
        print(f"\n✓ Successfully converted {len(pdf.pages)} pages to {output_dir}")
        return generated_files
        
    except ImportError:
        raise ImportError(
            "No PDF conversion library available.\n"
            "Please install either:\n"
            "  - pdf2image: pip install pdf2image (also requires: brew install poppler)\n"
            "  - pdfplumber: pip install pdfplumber"
        )
    except Exception as e:
        raise RuntimeError(f"Conversion failed: {e}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    input_pdf = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    
    try:
        files = convert_pdf_to_images(input_pdf, output_dir, dpi)
        print(f"\nGenerated {len(files)} images:")
        for f in files:
            print(f"  - {f}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
