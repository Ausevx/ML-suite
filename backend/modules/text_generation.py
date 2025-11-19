"""
Text Generation Module - Rephrases text based on style selection
"""
from flask import Blueprint, request, jsonify
from transformers import pipeline
import config

text_gen_bp = Blueprint('text_generation', __name__)

# Initialize text generation pipeline (lazy loading)
_text_pipeline = None

def get_text_pipeline():
    """Lazy load the text generation pipeline"""
    global _text_pipeline
    if _text_pipeline is None:
        try:
            print(f"Loading text generation model: {config.TEXT_MODEL_NAME}")
            print("This may take a few minutes on first run (downloading model)...")
            _text_pipeline = pipeline(
                "text2text-generation",
                model=config.TEXT_MODEL_NAME,
                device=-1,  # CPU
                max_length=512
            )
            print("✓ Text generation model loaded successfully!")
        except Exception as e:
            print(f"⚠ Warning: Could not load text generation model: {e}")
            print("→ Falling back to rule-based text generation")
            _text_pipeline = "FAILED"  # Mark as failed
    return _text_pipeline if _text_pipeline != "FAILED" else None

def rule_based_rephrase(text, style):
    """Fallback rule-based rephrasing if model unavailable"""
    import re
    
    # Simple transformations based on style
    if style == 'concise':
        # Remove filler words and redundancies
        text = re.sub(r'\b(very|really|actually|basically|literally)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text)  # Clean up extra spaces
        return text.strip()
    
    elif style == 'expanded':
        # Add context and detail
        sentences = text.split('. ')
        expanded = []
        for sent in sentences:
            if sent.strip():
                expanded.append(f"{sent}. This point is important to consider")
        return '. '.join(expanded)
    
    elif style == 'formal':
        # Make more formal
        text = text.replace("don't", "do not")
        text = text.replace("can't", "cannot")
        text = text.replace("won't", "will not")
        text = text.replace("it's", "it is")
        text = text.replace("I'm", "I am")
        text = text.replace("you're", "you are")
        text = text.replace("they're", "they are")
        # Remove casual phrases
        text = text.replace("kind of", "somewhat")
        text = text.replace("a lot of", "many")
        return text
    
    elif style == 'casual':
        # Make more casual
        text = text.replace("do not", "don't")
        text = text.replace("cannot", "can't")
        text = text.replace("will not", "won't")
        # Add casual intro if short
        if len(text.split()) < 20:
            return f"So, {text.lower()}"
        return text
    
    else:  # professional
        # Clean and professional
        text = ' '.join(text.split())  # Normalize whitespace
        return text.capitalize() if text else text

@text_gen_bp.route('/generate', methods=['POST'])
def generate_text():
    """Generate rephrased text based on input and style"""
    data = request.get_json()
    
    if not data or 'text' not in data or 'style' not in data:
        return jsonify({'error': 'Missing text or style parameter'}), 400
    
    input_text = data['text']
    style = data['style']
    
    # Validate input length (rough token estimate)
    if len(input_text.split()) > config.MAX_INPUT_TOKENS:
        return jsonify({'error': f'Input too long. Maximum {config.MAX_INPUT_TOKENS} tokens allowed.'}), 400
    
    # Valid styles
    valid_styles = ['professional', 'casual', 'formal', 'concise', 'expanded']
    if style not in valid_styles:
        return jsonify({'error': f'Invalid style. Must be one of: {", ".join(valid_styles)}'}), 400
    
    try:
        pipeline = get_text_pipeline()
        
        if pipeline:
            # Use transformer model
            try:
                prompt = f"Rephrase the following text in a {style} style: {input_text}"
                max_len = min(len(input_text.split()) + 100, 512)
                result = pipeline(prompt, max_length=max_len, min_length=10, do_sample=False, num_beams=2)
                generated_text = result[0]['generated_text']
                using_ai = True
            except Exception as e:
                print(f"AI generation failed: {e}, falling back to rule-based")
                generated_text = rule_based_rephrase(input_text, style)
                using_ai = False
        else:
            # Fallback to rule-based
            generated_text = rule_based_rephrase(input_text, style)
            using_ai = False
        
        return jsonify({
            'original': input_text,
            'generated': generated_text,
            'style': style,
            'using_ai': using_ai,
            'success': True
        })
        
    except Exception as e:
        # Last resort fallback
        return jsonify({
            'original': input_text,
            'generated': rule_based_rephrase(input_text, style),
            'style': style,
            'using_ai': False,
            'success': True,
            'warning': f'AI generation unavailable: {str(e)}'
        })

