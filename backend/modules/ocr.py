"""
OCR Module - Handles image upload and text extraction
"""
from flask import Blueprint, request, jsonify
import os
import pytesseract
from PIL import Image
import config
from werkzeug.utils import secure_filename

# Try to import pdf2image, but handle gracefully if not available
try:
    import pdf2image
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Warning: pdf2image not available. PDF support disabled. Install poppler and pdf2image for PDF support.")

ocr_bp = Blueprint('ocr', __name__)

def allowed_image_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_IMAGE_EXTENSIONS

@ocr_bp.route('/upload', methods=['POST'])
def upload_and_extract():
    """Upload image(s) and extract text using OCR"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    results = []
    
    for file in files:
        if file.filename == '':
            continue
            
        if not allowed_image_file(file.filename):
            results.append({
                'filename': file.filename,
                'error': 'Unsupported file format. Please upload PNG/JPG/JPEG/PDF.'
            })
            continue
        
        try:
            # Save uploaded file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(config.UPLOADS_DIR, filename)
            file.save(filepath)
            
            # Process based on file type
            if filename.lower().endswith('.pdf'):
                if not PDF_SUPPORT:
                    raise Exception("PDF support not available. Please install poppler and pdf2image.")
                # Convert ALL PDF pages to images and extract text
                images = pdf2image.convert_from_path(filepath, dpi=200)
                text_parts = []
                for page_num, image in enumerate(images, 1):
                    page_text = pytesseract.image_to_string(image)
                    if page_text.strip():
                        text_parts.append(f"--- Page {page_num} ---\n{page_text}")
                text = "\n\n".join(text_parts) if text_parts else ""
            else:
                # Process image file
                image = Image.open(filepath)
                text = pytesseract.image_to_string(image, config='--psm 3')
            
            # Clean up temporary file
            os.remove(filepath)
            
            # Get character and word count
            text_stripped = text.strip()
            word_count = len(text_stripped.split())
            char_count = len(text_stripped)
            
            results.append({
                'filename': filename,
                'text': text_stripped,
                'word_count': word_count,
                'char_count': char_count,
                'success': True
            })
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(filepath):
                os.remove(filepath)
            results.append({
                'filename': file.filename,
                'error': str(e),
                'success': False
            })
    
    return jsonify({'results': results})

