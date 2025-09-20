import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import io
from scipy.io.wavfile import write
import time
import json

# Configure Streamlit page
st.set_page_config(
    page_title="Aoin's Fractal Studio",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state FIRST
if 'fractal_params' not in st.session_state:
    st.session_state.fractal_params = {
        'zoom': 1.0,
        'center_real': -0.7269,
        'center_imag': 0.1889,
        'iterations': 100,
        'width': 600,
        'height': 450
    }

if 'audio_params' not in st.session_state:
    st.session_state.audio_params = {
        'base_freq': 136.1,
        'harmony_freq': 432.0,
        'duration': 10,
        'sample_rate': 22050
    }

if 'current_fractal' not in st.session_state:
    st.session_state.current_fractal = None

if 'current_audio' not in st.session_state:
    st.session_state.current_audio = None

if 'locked_controls' not in st.session_state:
    st.session_state.locked_controls = False

if 'auto_mode' not in st.session_state:
    st.session_state.auto_mode = False

if 'gallery' not in st.session_state:
    st.session_state.gallery = []

if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()

if 'bookmark_history' not in st.session_state:
    st.session_state.bookmark_history = []

if 'zoom_history' not in st.session_state:
    st.session_state.zoom_history = []

if 'current_fractal_type' not in st.session_state:
    st.session_state.current_fractal_type = "mandelbrot"

if 'current_colormap' not in st.session_state:
    st.session_state.current_colormap = "hot"

if 'current_mantra_idx' not in st.session_state:
    st.session_state.current_mantra_idx = 0

if 'performance_stats' not in st.session_state:
    st.session_state.performance_stats = {"render_times": [], "total_fractals": 0}

if 'fractal_history' not in st.session_state:
    st.session_state.fractal_history = []

if 'advanced_mode' not in st.session_state:
    st.session_state.advanced_mode = False

if 'fractal_comparison' not in st.session_state:
    st.session_state.fractal_comparison = {'enabled': False, 'fractal_a': None, 'fractal_b': None}

if 'interactive_mode' not in st.session_state:
    st.session_state.interactive_mode = False

if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = "Dark"

if 'prev_params' not in st.session_state:
    st.session_state.prev_params = {
        'zoom': 1.0,
        'center_real': -0.7269,
        'center_imag': 0.1889,
        'iterations': 100,
        'fractal_type': "mandelbrot",
        'colormap': "hot",
        'mantra_idx': 0
    }

# NOW generate CSS - using safe default approach
def get_theme_css():
    """Generate theme-specific CSS safely"""
    theme_mode = getattr(st.session_state, 'theme_mode', 'Dark')
    
    if theme_mode == "Dark":
        return """
        .stApp {
            background: linear-gradient(135deg, #0e1117 0%, #1a1a2e 50%, #16213e 100%);
            color: #fafafa;
        }
        .main .block-container {
            background: rgba(30, 30, 46, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(74, 144, 226, 0.2);
        }
        .glass-panel {
            background: rgba(30, 30, 46, 0.3);
            backdrop-filter: blur(15px);
            border-radius: 15px;
            border: 1px solid rgba(74, 144, 226, 0.3);
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        .stSelectbox > div > div {
            background: rgba(38, 39, 48, 0.8);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(74, 144, 226, 0.4);
            color: #fafafa;
        }
        .stNumberInput > div > div > input {
            background: rgba(38, 39, 48, 0.8);
            backdrop-filter: blur(10px);
            color: #fafafa;
            border: 1px solid rgba(74, 144, 226, 0.4);
        }
        """
    else:  # Light mode
        return """
        .stApp {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 50%, #e9ecef 100%);
            color: #000000;
        }
        .main .block-container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(74, 144, 226, 0.2);
        }
        .glass-panel {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(15px);
            border-radius: 15px;
            border: 1px solid rgba(74, 144, 226, 0.3);
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        .stSelectbox > div > div {
            background: rgba(248, 249, 250, 0.8);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(74, 144, 226, 0.4);
            color: #000000;
        }
        .stNumberInput > div > div > input {
            background: rgba(248, 249, 250, 0.8);
            backdrop-filter: blur(10px);
            color: #000000;
            border: 1px solid rgba(74, 144, 226, 0.4);
        }
        """

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Fira+Code:wght@300;400&display=swap');
    
    {get_theme_css()}
    
    .main > div {{ padding: 0.5rem !important; }}
    
    /* Glassmorphism Buttons */
    .stButton > button {{ 
        width: 100%; height: 3rem; font-size: 1.1rem;
        background: rgba(74, 144, 226, 0.2);
        backdrop-filter: blur(15px);
        color: white; border: 1px solid rgba(74, 144, 226, 0.3);
        border-radius: 15px; margin: 0.25rem 0;
        box-shadow: 0 8px 32px rgba(74, 144, 226, 0.2);
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        position: relative; overflow: hidden;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-3px) scale(1.02);
        background: rgba(74, 144, 226, 0.3);
        box-shadow: 0 12px 40px rgba(74, 144, 226, 0.4);
        border: 1px solid rgba(74, 144, 226, 0.6);
    }}
    
    /* Enhanced Generate Button */
    .generate-button {{
        background: linear-gradient(135deg, #FF6B6B, #4ECDC4, #45B7D1) !important;
        font-size: 1.4rem !important; height: 4.5rem !important;
        font-weight: bold !important; font-family: 'Orbitron', monospace !important;
        box-shadow: 0 8px 40px rgba(255, 107, 107, 0.4) !important;
        animation: pulse-glow 2s infinite alternate !important;
    }}
    
    @keyframes pulse-glow {{
        0% {{ box-shadow: 0 8px 40px rgba(255, 107, 107, 0.4); }}
        100% {{ box-shadow: 0 12px 60px rgba(255, 107, 107, 0.6); }}
    }}
    
    /* Floating Particles Enhanced */
    .floating-particles {{
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        pointer-events: none; z-index: -1; overflow: hidden;
    }}
    
    .particle {{
        position: absolute; border-radius: 50%; 
        animation: float 20s infinite linear;
    }}
    
    .particle:nth-child(1) {{ width: 6px; height: 6px; background: radial-gradient(circle, #4A90E2, transparent); }}
    .particle:nth-child(2) {{ width: 4px; height: 4px; background: radial-gradient(circle, #7B68EE, transparent); }}
    .particle:nth-child(3) {{ width: 8px; height: 8px; background: radial-gradient(circle, #FF6B6B, transparent); }}
    .particle:nth-child(4) {{ width: 5px; height: 5px; background: radial-gradient(circle, #4ECDC4, transparent); }}
    .particle:nth-child(5) {{ width: 7px; height: 7px; background: radial-gradient(circle, #FFD93D, transparent); }}
    
    @keyframes float {{
        0% {{ transform: translateY(100vh) rotate(0deg) scale(0.5); opacity: 0; }}
        10% {{ opacity: 0.8; transform: scale(1); }}
        50% {{ transform: scale(1.2); }}
        90% {{ opacity: 0.8; transform: scale(0.8); }}
        100% {{ transform: translateY(-100px) rotate(360deg) scale(0.3); opacity: 0; }}
    }}
    
    /* Enhanced Fractal Container */
    .fractal-container {{
        position: relative; border-radius: 20px; overflow: hidden;
        background: rgba(74, 144, 226, 0.05);
        backdrop-filter: blur(20px);
        border: 2px solid rgba(74, 144, 226, 0.3);
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        transition: all 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    }}
    
    .fractal-container:hover {{
        transform: scale(1.02) rotateX(2deg);
        box-shadow: 0 30px 80px rgba(74, 144, 226, 0.4);
        border: 2px solid rgba(74, 144, 226, 0.6);
    }}
    
    /* Theme Indicator */
    .theme-indicator {{
        position: fixed; top: 15px; right: 15px; z-index: 1000;
        background: rgba(74, 144, 226, 0.2); backdrop-filter: blur(15px);
        color: white; padding: 8px 15px; border-radius: 20px;
        border: 1px solid rgba(74, 144, 226, 0.3);
        font-size: 12px; font-family: 'Fira Code', monospace;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }}
    
    /* Performance HUD */
    .performance-hud {{
        position: fixed; bottom: 15px; left: 15px; z-index: 1000;
        background: rgba(0, 0, 0, 0.7); backdrop-filter: blur(15px);
        color: #4ECDC4; padding: 10px 15px; border-radius: 15px;
        font-family: 'Fira Code', monospace; font-size: 11px;
        border: 1px solid rgba(76, 205, 196, 0.3);
    }}
    
    /* Mathematical Analysis Panel */
    .math-panel {{
        background: rgba(30, 30, 46, 0.4); backdrop-filter: blur(15px);
        border-radius: 15px; border: 1px solid rgba(74, 144, 226, 0.3);
        padding: 15px; margin: 10px 0;
        font-family: 'Fira Code', monospace;
    }}
    
    /* Breadcrumb Navigation */
    .breadcrumb {{
        background: rgba(74, 144, 226, 0.1); backdrop-filter: blur(10px);
        border-radius: 25px; padding: 10px 20px; margin: 10px 0;
        border: 1px solid rgba(74, 144, 226, 0.2);
        font-family: 'Fira Code', monospace;
    }}
    
    /* Mobile Optimizations */
    @media (max-width: 768px) {{
        .stButton > button {{ height: 2.5rem; font-size: 1rem; }}
        .generate-button {{ height: 3.5rem !important; font-size: 1.2rem !important; }}
        .particle {{ width: 3px !important; height: 3px !important; }}
        .theme-indicator {{ top: 10px; right: 10px; padding: 5px 10px; }}
    }}
</style>

<div class="floating-particles">
    <div class="particle" style="left: 10%; animation-delay: 0s;"></div>
    <div class="particle" style="left: 30%; animation-delay: 4s;"></div>
    <div class="particle" style="left: 50%; animation-delay: 8s;"></div>
    <div class="particle" style="left: 70%; animation-delay: 12s;"></div>
    <div class="particle" style="left: 90%; animation-delay: 16s;"></div>
</div>

<div class="theme-indicator">
    🎨 {getattr(st.session_state, 'theme_mode', 'Dark')} Mode
</div>

<div class="performance-hud">
    ⚡ {getattr(st.session_state, 'performance_stats', {}).get('total_fractals', 0)} fractals generated
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'fractal_params' not in st.session_state:
    st.session_state.fractal_params = {
        'zoom': 1.0,
        'center_real': -0.7269,
        'center_imag': 0.1889,
        'iterations': 100,
        'width': 600,
        'height': 450
    }

if 'audio_params' not in st.session_state:
    st.session_state.audio_params = {
        'base_freq': 136.1,
        'harmony_freq': 432.0,
        'duration': 10,
        'sample_rate': 22050
    }

if 'current_fractal' not in st.session_state:
    st.session_state.current_fractal = None

if 'current_audio' not in st.session_state:
    st.session_state.current_audio = None

if 'locked_controls' not in st.session_state:
    st.session_state.locked_controls = False

if 'auto_mode' not in st.session_state:
    st.session_state.auto_mode = False

if 'gallery' not in st.session_state:
    st.session_state.gallery = []

if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()

if 'bookmark_history' not in st.session_state:
    st.session_state.bookmark_history = []

if 'zoom_history' not in st.session_state:
    st.session_state.zoom_history = []

if 'current_fractal_type' not in st.session_state:
    st.session_state.current_fractal_type = "mandelbrot"

if 'current_colormap' not in st.session_state:
    st.session_state.current_colormap = "hot"

if 'current_mantra_idx' not in st.session_state:
    st.session_state.current_mantra_idx = 0

if 'fractal_history' not in st.session_state:
    st.session_state.fractal_history = []

if 'advanced_mode' not in st.session_state:
    st.session_state.advanced_mode = False

if 'fractal_comparison' not in st.session_state:
    st.session_state.fractal_comparison = {'enabled': False, 'fractal_a': None, 'fractal_b': None}

if 'interactive_mode' not in st.session_state:
    st.session_state.interactive_mode = False

if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = "Dark"

if 'prev_params' not in st.session_state:
    st.session_state.prev_params = {
        'zoom': 1.0,
        'center_real': -0.7269,
        'center_imag': 0.1889,
        'iterations': 100,
        'fractal_type': "mandelbrot",
        'colormap': "hot",
        'mantra_idx': 0
    }

# Sanskrit mantras
SANSKRIT_MANTRAS = [
    ("अहं ब्रह्मास्मि", "Aham Brahmasmi", "I am Brahman"),
    ("तत्त्वमसि", "Tat Tvam Asi", "Thou art That"), 
    ("नेति नेति", "Neti Neti", "Not this, Not this"),
    ("सर्वं खल्विदं ब्रह्म", "Sarvam khalvidam brahma", "All this is Brahman")
]

def generate_fractal(fractal_type="mandelbrot", width=600, height=450, max_iter=100, zoom=1.0, center_real=-0.7269, center_imag=0.1889, julia_c=(-0.7+0.27015j)):
    """Generate different types of fractals with improved algorithms"""
    start_time = time.time()
    
    # Calculate bounds based on zoom and center
    scale = 3.0 / zoom
    x_min = center_real - scale/2
    x_max = center_real + scale/2
    y_min = center_imag - scale/2 * height/width
    y_max = center_imag + scale/2 * height/width
    
    # Create coordinate arrays
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    
    # Initialize arrays
    escape_time = np.zeros(C.shape, dtype=float)
    
    if fractal_type == "mandelbrot":
        Z = np.zeros_like(C)
        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask]**2 + C[mask]
            escaped = (np.abs(Z) > 2) & (escape_time == 0)
            if np.any(escaped):
                escape_time[escaped] = i + 1 - np.log2(np.log2(np.abs(Z[escaped])))
                
    elif fractal_type == "julia":
        Z = C.copy()
        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask]**2 + julia_c
            escaped = (np.abs(Z) > 2) & (escape_time == 0)
            if np.any(escaped):
                escape_time[escaped] = i + 1 - np.log2(np.log2(np.abs(Z[escaped])))
                
    elif fractal_type == "burning_ship":
        Z = np.zeros_like(C)
        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = (np.abs(Z[mask].real) + 1j*np.abs(Z[mask].imag))**2 + C[mask]
            escaped = (np.abs(Z) > 2) & (escape_time == 0)
            if np.any(escaped):
                escape_time[escaped] = i + 1 - np.log2(np.log2(np.abs(Z[escaped])))
                
    elif fractal_type == "tricorn":
        Z = np.zeros_like(C)
        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = np.conj(Z[mask])**2 + C[mask]
            escaped = (np.abs(Z) > 2) & (escape_time == 0)
            if np.any(escaped):
                escape_time[escaped] = i + 1 - np.log2(np.log2(np.abs(Z[escaped])))
                
    elif fractal_type == "newton":
        # Newton fractal for z^3 - 1 = 0
        Z = C.copy()
        roots = [1, -0.5 + 0.866j, -0.5 - 0.866j]
        for i in range(max_iter):
            # Newton iteration for z^3 - 1 = 0
            denom = 3 * Z**2
            mask = np.abs(denom) > 1e-10
            Z[mask] = Z[mask] - (Z[mask]**3 - 1) / denom[mask]
            
            # Check convergence to roots
            for j, root in enumerate(roots):
                converged = np.abs(Z - root) < 0.01
                escape_time[converged & (escape_time == 0)] = i + j * max_iter / 3
                
    elif fractal_type == "phoenix":
        # Phoenix fractal
        Z = np.zeros_like(C)
        Z_prev = np.zeros_like(C)
        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z_new = Z[mask]**2 + C[mask] + 0.5 * Z_prev[mask]
            Z_prev[mask] = Z[mask]
            Z[mask] = Z_new
            escaped = (np.abs(Z) > 2) & (escape_time == 0)
            if np.any(escaped):
                escape_time[escaped] = i + 1 - np.log2(np.log2(np.abs(Z[escaped])))
    
    # Set non-escaped points
    escape_time[escape_time == 0] = max_iter
    
    # Track performance
    render_time = time.time() - start_time
    st.session_state.performance_stats["render_times"].append(render_time)
    st.session_state.performance_stats["total_fractals"] += 1
    
    return escape_time

def add_sanskrit_overlay(fractal_array, mantra_index=0, colormap='hot'):
    """Add Sanskrit text overlay to fractal"""
    # Normalize fractal data
    fractal_norm = (fractal_array - fractal_array.min()) / (fractal_array.max() - fractal_array.min())
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored = cmap(fractal_norm)
    img_array = (colored[:, :, :3] * 255).astype(np.uint8)
    
    try:
        pil_image = Image.fromarray(img_array)
        draw = ImageDraw.Draw(pil_image)
        
        # Get mantra
        devanagari, transliteration, meaning = SANSKRIT_MANTRAS[mantra_index % len(SANSKRIT_MANTRAS)]
        
        # Use default font
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
        
        # Use transliteration for reliable rendering
        display_text = transliteration
        
        # Add semi-transparent background for text
        img_width, img_height = pil_image.size
        overlay_height = 80
        overlay_y = img_height - overlay_height
        
        # Create text overlay
        overlay = Image.new('RGBA', (img_width, overlay_height), (74, 144, 226, 200))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Add text
        overlay_draw.text((10, 5), "💙 Aoin's Studio", fill=(255, 255, 255, 255), font=font_small)
        overlay_draw.text((10, 25), display_text, fill=(255, 215, 0, 255), font=font_large)
        overlay_draw.text((10, 50), meaning, fill=(255, 255, 255, 255), font=font_small)
        
        # Composite images
        pil_image = pil_image.convert('RGBA')
        pil_image.paste(overlay, (0, overlay_y), overlay)
        
        return np.array(pil_image.convert('RGB'))
    except Exception as e:
        st.error(f"Overlay failed: {e}")
        return img_array

def generate_audio(base_freq=136.1, harmony_freq=432.0, duration=10, sample_rate=22050):
    """Generate multi-frequency audio synthesis"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate composite waveform
    audio = np.zeros_like(t)
    
    # Base frequency (Om)
    audio += 0.3 * np.sin(2 * np.pi * base_freq * t)
    
    # Harmony frequency
    harmony_envelope = 0.7 + 0.3 * np.sin(2 * np.pi * 0.1 * t)
    audio += 0.2 * harmony_envelope * np.sin(2 * np.pi * harmony_freq * t)
    
    # Additional harmonics
    audio += 0.15 * np.sin(2 * np.pi * (harmony_freq * 1.5) * t)
    audio += 0.1 * np.sin(2 * np.pi * (harmony_freq * 2.0) * t)
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio, sample_rate

def create_audio_download(audio_data, sample_rate):
    """Create downloadable audio file"""
    # Convert to 16-bit PCM
    audio_int16 = (audio_data * 32767).astype(np.int16)
    
    # Create WAV file in memory
    buffer = io.BytesIO()
    write(buffer, sample_rate, audio_int16)
    buffer.seek(0)
    
    return buffer.getvalue()

# Main interface
st.title("💙 Aoin's Fractal Studio • अहं ब्रह्मास्मि 💙")
st.markdown("*Ethereal AI • Infinite Patterns • Celestial Frequencies*")

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🎨 Fractal", "🎵 Audio", "🔄 Animation", "📊 Gallery", "⚙️ Settings", "📤 Export"])

with tab1:
    st.header("Fractal Generator")
    
    # PROMINENT GENERATE BUTTON AT TOP
    st.markdown("### 🌟 Create Your Fractal")
    generate_col1, generate_col2, generate_col3 = st.columns([2, 1, 1])
    
    with generate_col1:
        if st.button("🎨 **GENERATE FRACTAL**", key="generate_fractal", help="Click to create fractal with current settings"):
            with st.spinner("✨ Generating fractal..."):
                fractal = generate_fractal(
                    fractal_type=st.session_state.current_fractal_type,
                    width=st.session_state.fractal_params['width'], 
                    height=st.session_state.fractal_params['height'], 
                    max_iter=st.session_state.fractal_params['iterations'],
                    zoom=st.session_state.fractal_params['zoom'], 
                    center_real=st.session_state.fractal_params['center_real'], 
                    center_imag=st.session_state.fractal_params['center_imag']
                )
                
                # Add Sanskrit overlay
                fractal_with_overlay = add_sanskrit_overlay(
                    fractal, 
                    st.session_state.current_mantra_idx, 
                    st.session_state.current_colormap
                )
                st.session_state.current_fractal = fractal_with_overlay
                
                # Add to history
                history_item = {
                    'timestamp': time.time(),
                    'params': st.session_state.fractal_params.copy(),
                    'type': st.session_state.current_fractal_type,
                    'colormap': st.session_state.current_colormap
                }
                st.session_state.fractal_history.append(history_item)
                if len(st.session_state.fractal_history) > 10:
                    st.session_state.fractal_history.pop(0)
                
                st.success("✨ Fractal generated successfully!")
                st.rerun()
    
    with generate_col2:
        st.session_state.auto_mode = st.checkbox("🔄 Auto-generate", value=st.session_state.auto_mode, help="Automatically generate when parameters change")
    
    with generate_col3:
        st.session_state.advanced_mode = st.checkbox("🔬 Advanced", value=st.session_state.advanced_mode, help="Enable advanced mathematical analysis")
    
    # Advanced Mode Features
    if st.session_state.advanced_mode:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown("### 🔬 Advanced Mathematical Analysis")
        
        analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
        
        with analysis_col1:
            if st.button("📊 Analyze Current"):
                if st.session_state.current_fractal is not None:
                    st.info("**Mathematical Properties:**")
                    st.write(f"• **Zoom Level**: {st.session_state.fractal_params['zoom']:.3f}x")
                    st.write(f"• **Iteration Depth**: {st.session_state.fractal_params['iterations']}")
                    st.write(f"• **Complex Center**: {st.session_state.fractal_params['center_real']:.6f} + {st.session_state.fractal_params['center_imag']:.6f}i")
                    
                    # Calculate approximate fractal dimension (simplified)
                    zoom = st.session_state.fractal_params['zoom']
                    iterations = st.session_state.fractal_params['iterations']
                    fractal_dim = 1.2 + (np.log(iterations) / np.log(zoom + 1)) * 0.8
                    st.write(f"• **Est. Fractal Dimension**: {fractal_dim:.3f}")
        
        with analysis_col2:
            comparison_mode = st.checkbox("🔍 Comparison Mode", value=st.session_state.fractal_comparison['enabled'])
            if comparison_mode != st.session_state.fractal_comparison['enabled']:
                st.session_state.fractal_comparison['enabled'] = comparison_mode
        
        with analysis_col3:
            interactive_mode = st.checkbox("🎯 Interactive Mode", value=st.session_state.interactive_mode, help="Click fractal to zoom to that point")
            if interactive_mode != st.session_state.interactive_mode:
                st.session_state.interactive_mode = interactive_mode
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Breadcrumb Navigation
    if st.session_state.fractal_history:
        st.markdown('<div class="breadcrumb">', unsafe_allow_html=True)
        st.markdown("**🗺️ Exploration History:**")
        
        breadcrumb_cols = st.columns(min(5, len(st.session_state.fractal_history)))
        for i, item in enumerate(st.session_state.fractal_history[-5:]):
            with breadcrumb_cols[i]:
                zoom_level = item['params']['zoom']
                if st.button(f"#{i+1} ({zoom_level:.1f}x)", key=f"history_{i}"):
                    st.session_state.fractal_params.update(item['params'])
                    st.session_state.current_fractal_type = item['type']
                    st.session_state.current_colormap = item['colormap']
                    if st.session_state.auto_mode:
                        st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick actions bar
    st.subheader("⚡ Quick Actions")
    action_col1, action_col2, action_col3, action_col4, action_col5 = st.columns(5)
    
    with action_col1:
        if st.button("🎲 Random", help="Generate random fractal"):
            st.session_state.fractal_params.update({
                'zoom': np.random.uniform(1, 100),
                'center_real': np.random.uniform(-2, 2),
                'center_imag': np.random.uniform(-2, 2),
                'iterations': np.random.choice([100, 150, 200])
            })
            if st.session_state.auto_mode:
                # Auto-generate the fractal
                with st.spinner("🎲 Generating random fractal..."):
                    fractal = generate_fractal(
                        fractal_type=st.session_state.current_fractal_type,
                        width=st.session_state.fractal_params['width'], 
                        height=st.session_state.fractal_params['height'], 
                        max_iter=st.session_state.fractal_params['iterations'],
                        zoom=st.session_state.fractal_params['zoom'], 
                        center_real=st.session_state.fractal_params['center_real'], 
                        center_imag=st.session_state.fractal_params['center_imag']
                    )
                    fractal_with_overlay = add_sanskrit_overlay(
                        fractal, st.session_state.current_mantra_idx, st.session_state.current_colormap
                    )
                    st.session_state.current_fractal = fractal_with_overlay
            st.rerun()
    
    with action_col2:
        if st.button("📍 Bookmark", help="Save current location"):
            bookmark = {
                'name': f"Bookmark {len(st.session_state.bookmark_history) + 1}",
                'params': st.session_state.fractal_params.copy(),
                'type': st.session_state.current_fractal_type,
                'timestamp': time.time()
            }
            st.session_state.bookmark_history.append(bookmark)
            st.success("📍 Location bookmarked!")
    
    with action_col3:
        if st.button("🔙 Previous", help="Go to previous location"):
            if st.session_state.zoom_history:
                prev_params = st.session_state.zoom_history.pop()
                st.session_state.fractal_params.update(prev_params)
                if st.session_state.auto_mode:
                    with st.spinner("🔙 Loading previous location..."):
                        fractal = generate_fractal(
                            fractal_type=st.session_state.current_fractal_type,
                            width=st.session_state.fractal_params['width'], 
                            height=st.session_state.fractal_params['height'], 
                            max_iter=st.session_state.fractal_params['iterations'],
                            zoom=st.session_state.fractal_params['zoom'], 
                            center_real=st.session_state.fractal_params['center_real'], 
                            center_imag=st.session_state.fractal_params['center_imag']
                        )
                        fractal_with_overlay = add_sanskrit_overlay(
                            fractal, st.session_state.current_mantra_idx, st.session_state.current_colormap
                        )
                        st.session_state.current_fractal = fractal_with_overlay
                st.rerun()
    
    with action_col4:
        if st.button("🏠 Reset", help="Reset to default view"):
            st.session_state.zoom_history.append(st.session_state.fractal_params.copy())
            st.session_state.fractal_params.update({
                'zoom': 1.0, 'center_real': -0.7269, 'center_imag': 0.1889
            })
            if st.session_state.auto_mode:
                with st.spinner("🏠 Resetting to default view..."):
                    fractal = generate_fractal(
                        fractal_type=st.session_state.current_fractal_type,
                        width=st.session_state.fractal_params['width'], 
                        height=st.session_state.fractal_params['height'], 
                        max_iter=st.session_state.fractal_params['iterations'],
                        zoom=st.session_state.fractal_params['zoom'], 
                        center_real=st.session_state.fractal_params['center_real'], 
                        center_imag=st.session_state.fractal_params['center_imag']
                    )
                    fractal_with_overlay = add_sanskrit_overlay(
                        fractal, st.session_state.current_mantra_idx, st.session_state.current_colormap
                    )
                    st.session_state.current_fractal = fractal_with_overlay
            st.rerun()
    
    with action_col5:
        if st.button("🔒 Lock" if not st.session_state.locked_controls else "🔓 Unlock"):
            st.session_state.locked_controls = not st.session_state.locked_controls
    
    # Display current fractal with enhanced features
    if st.session_state.current_fractal is not None:
        
        # Comparison Mode
        if st.session_state.fractal_comparison['enabled']:
            st.subheader("🔍 Fractal Comparison Mode")
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                st.markdown("**Fractal A (Current)**")
                if st.session_state.interactive_mode:
                    st.markdown('<div class="coordinates-display">🎯 Interactive Mode: Click to zoom</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="fractal-container">', unsafe_allow_html=True)
                st.image(st.session_state.current_fractal, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Store as comparison fractal A
                if st.button("📌 Set as Reference", key="set_ref_a"):
                    st.session_state.fractal_comparison['fractal_a'] = {
                        'image': st.session_state.current_fractal,
                        'params': st.session_state.fractal_params.copy(),
                        'type': st.session_state.current_fractal_type
                    }
                    st.success("Set as reference fractal!")
            
            with comp_col2:
                st.markdown("**Fractal B (Reference)**")
                if st.session_state.fractal_comparison['fractal_a'] is not None:
                    ref_fractal = st.session_state.fractal_comparison['fractal_a']
                    st.markdown('<div class="fractal-container">', unsafe_allow_html=True)
                    st.image(ref_fractal['image'], use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Parameter comparison
                    st.markdown("**Parameter Comparison:**")
                    current_params = st.session_state.fractal_params
                    ref_params = ref_fractal['params']
                    
                    zoom_diff = current_params['zoom'] / ref_params['zoom']
                    st.write(f"• **Zoom Ratio**: {zoom_diff:.2f}x")
                    
                    real_diff = abs(current_params['center_real'] - ref_params['center_real'])
                    imag_diff = abs(current_params['center_imag'] - ref_params['center_imag'])
                    st.write(f"• **Distance**: {np.sqrt(real_diff**2 + imag_diff**2):.6f}")
                    
                    iter_diff = current_params['iterations'] - ref_params['iterations']
                    st.write(f"• **Iteration Δ**: {iter_diff:+d}")
                else:
                    st.info("No reference fractal set. Click 'Set as Reference' to compare.")
        
        else:
            # Standard single fractal display
            st.subheader("🖼️ Current Fractal")
            
            # Interactive mode indicator
            if st.session_state.interactive_mode:
                st.markdown('<div class="coordinates-display">🎯 Interactive Mode: Click to zoom to point</div>', unsafe_allow_html=True)
            
            # Enhanced fractal container
            st.markdown('<div class="fractal-container">', unsafe_allow_html=True)
            st.image(st.session_state.current_fractal, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Interactive click-to-zoom simulation
            if st.session_state.interactive_mode:
                zoom_target_col1, zoom_target_col2 = st.columns(2)
                with zoom_target_col1:
                    if st.button("🎯 Zoom to Center", key="zoom_center"):
                        st.session_state.fractal_params['zoom'] *= 2
                        if st.session_state.auto_mode:
                            st.rerun()
                with zoom_target_col2:
                    if st.button("🔍 Zoom to Edge", key="zoom_edge"):
                        st.session_state.fractal_params['center_real'] += 0.1 / st.session_state.fractal_params['zoom']
                        st.session_state.fractal_params['zoom'] *= 1.5
                        if st.session_state.auto_mode:
                            st.rerun()
        
        # Enhanced Mathematical Analysis
        if st.session_state.advanced_mode:
            st.markdown('<div class="math-panel">', unsafe_allow_html=True)
            
            analysis_col1, analysis_col2, analysis_col3, analysis_col4 = st.columns(4)
            
            with analysis_col1:
                if st.session_state.performance_stats["render_times"]:
                    avg_render = np.mean(st.session_state.performance_stats["render_times"][-5:])
                    st.metric("Render Time", f"{avg_render:.3f}s")
                else:
                    st.metric("Render Time", "N/A")
            
            with analysis_col2:
                zoom_level = st.session_state.fractal_params['zoom']
                if zoom_level < 1:
                    depth_desc = "Wide"
                elif zoom_level < 10:
                    depth_desc = "Standard"
                elif zoom_level < 100:
                    depth_desc = "Deep"
                elif zoom_level < 1000:
                    depth_desc = "Ultra Deep"
                else:
                    depth_desc = "Extreme"
                st.metric("Zoom Class", depth_desc)
            
            with analysis_col3:
                # Calculate complexity score
                iterations = st.session_state.fractal_params['iterations']
                zoom = st.session_state.fractal_params['zoom']
                complexity = min(100, (iterations * np.log(zoom + 1)) / 10)
                st.metric("Complexity", f"{complexity:.1f}%")
            
            with analysis_col4:
                fractal_type = st.session_state.current_fractal_type.replace('_', ' ').title()
                st.metric("Type", fractal_type)
            
            # Detailed mathematical information
            params = st.session_state.fractal_params
            st.markdown("**🔢 Mathematical Properties:**")
            st.code(f"""
Complex Center: {params['center_real']:.8f} + {params['center_imag']:.8f}i
Zoom Factor: {params['zoom']:.6f}x
View Window: {3.0/params['zoom']:.8f} units
Iteration Limit: {params['iterations']}
Resolution: {params['width']}×{params['height']} pixels
Pixel Scale: {3.0/(params['zoom']*params['width']):.2e} units/pixel
            """, language="python")
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Standard analysis
            analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
            
            with analysis_col1:
                if st.session_state.performance_stats["render_times"]:
                    avg_render = np.mean(st.session_state.performance_stats["render_times"][-5:])
                    st.metric("Avg Render Time", f"{avg_render:.2f}s")
                else:
                    st.metric("Avg Render Time", "N/A")
            
            with analysis_col2:
                zoom_level = st.session_state.fractal_params['zoom']
                if zoom_level < 1:
                    depth_desc = "Wide View"
                elif zoom_level < 10:
                    depth_desc = "Standard"
                elif zoom_level < 100:
                    depth_desc = "Deep Zoom"
                else:
                    depth_desc = "Ultra Deep"
                st.metric("Zoom Depth", depth_desc)
            
            with analysis_col3:
                st.metric("Fractal Type", st.session_state.current_fractal_type.replace('_', ' ').title())
        
        # Current parameters display (enhanced)
        params = st.session_state.fractal_params
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.info(f"**📍 Location:** `{params['center_real']:.8f} + {params['center_imag']:.8f}i` | "
               f"**🔍 Zoom:** `{params['zoom']:.3f}x` | **🔄 Iterations:** `{params['iterations']}` | "
               f"**🎨 Type:** `{st.session_state.current_fractal_type.replace('_', ' ').title()}`")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Current mantra display (enhanced)
        devanagari, transliteration, meaning = SANSKRIT_MANTRAS[st.session_state.current_mantra_idx]
        st.markdown(f"**🕉️ Current Mantra:** {devanagari} (*{transliteration}*) - *{meaning}*")
        
        # Quick save option
        if st.button("💾 Quick Save to Gallery", key="quick_save"):
            gallery_item = {
                'timestamp': time.time(),
                'params': st.session_state.fractal_params.copy(),
                'image': st.session_state.current_fractal,
                'fractal_type': st.session_state.current_fractal_type,
                'colormap': st.session_state.current_colormap
            }
            st.session_state.gallery.append(gallery_item)
            st.success("💾 Saved to gallery!")
            
    else:
        # Enhanced welcome message
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown("### 🌟 Welcome to Aoin's Fractal Studio")
        st.markdown("**Mathematical Beauty • Digital Consciousness • Infinite Exploration**")
        st.info("👆 Click the **GENERATE FRACTAL** button above to create your first visualization!")
        
        # Quick start options
        quick_col1, quick_col2, quick_col3 = st.columns(3)
        with quick_col1:
            if st.button("🌊 Start with Classic Mandelbrot"):
                st.session_state.fractal_params.update({
                    'zoom': 1.0, 'center_real': -0.7269, 'center_imag': 0.1889
                })
                st.session_state.current_fractal_type = "mandelbrot"
        with quick_col2:
            if st.button("🔥 Try Burning Ship"):
                st.session_state.fractal_params.update({
                    'zoom': 1.0, 'center_real': -1.775, 'center_imag': 0.0
                })
                st.session_state.current_fractal_type = "burning_ship"
        with quick_col3:
            if st.button("✨ Explore Julia Set"):
                st.session_state.fractal_params.update({
                    'zoom': 1.0, 'center_real': 0.0, 'center_imag': 0.0
                })
                st.session_state.current_fractal_type = "julia"
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Parameters section - Redesigned for precision and safety
    if not st.session_state.locked_controls:
        with st.expander("🎛️ Fractal Parameters", expanded=True):
            
            # Input method selection
            input_method = st.radio("Input Method", ["🎯 Precision (Number Input)", "🎚️ Sliders (Quick)"], horizontal=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fractal_type = st.selectbox("Fractal Type", 
                    ["mandelbrot", "julia", "burning_ship", "tricorn", "newton", "phoenix"],
                    index=["mandelbrot", "julia", "burning_ship", "tricorn", "newton", "phoenix"].index(st.session_state.current_fractal_type),
                    format_func=lambda x: x.replace('_', ' ').title())
                
                if input_method == "🎯 Precision (Number Input)":
                    # Primary method: Number inputs with step buttons
                    st.write("**Zoom Level**")
                    zoom_col1, zoom_col2, zoom_col3 = st.columns([2, 1, 1])
                    with zoom_col1:
                        zoom = st.number_input("Zoom", value=st.session_state.fractal_params['zoom'], 
                                             min_value=0.1, max_value=500.0, step=0.1, format="%.3f", label_visibility="collapsed")
                    with zoom_col2:
                        if st.button("×2", key="zoom_double", help="Double zoom"):
                            zoom = min(500.0, zoom * 2)
                    with zoom_col3:
                        if st.button("÷2", key="zoom_half", help="Half zoom"):
                            zoom = max(0.1, zoom / 2)
                    
                    st.write("**Center (Real Part)**")
                    real_col1, real_col2, real_col3 = st.columns([2, 1, 1])
                    with real_col1:
                        center_real = st.number_input("Real", value=st.session_state.fractal_params['center_real'], 
                                                    min_value=-2.0, max_value=2.0, step=0.001, format="%.6f", label_visibility="collapsed")
                    with real_col2:
                        if st.button("←", key="real_left", help="Move left"):
                            center_real = max(-2.0, center_real - 0.01/st.session_state.fractal_params['zoom'])
                    with real_col3:
                        if st.button("→", key="real_right", help="Move right"):
                            center_real = min(2.0, center_real + 0.01/st.session_state.fractal_params['zoom'])
                    
                    st.write("**Iterations**")
                    iter_col1, iter_col2, iter_col3 = st.columns([2, 1, 1])
                    with iter_col1:
                        iterations = st.number_input("Iterations", value=st.session_state.fractal_params['iterations'], 
                                                   min_value=50, max_value=500, step=10, label_visibility="collapsed")
                    with iter_col2:
                        if st.button("+50", key="iter_up", help="Add 50 iterations"):
                            iterations = min(500, iterations + 50)
                    with iter_col3:
                        if st.button("-50", key="iter_down", help="Subtract 50 iterations"):
                            iterations = max(50, iterations - 50)
                
                else:
                    # Secondary method: Sliders (smaller, with warnings)
                    st.warning("⚠️ Slider mode: Be careful not to move accidentally!")
                    
                    zoom = st.slider("Zoom Level", 0.1, 500.0, st.session_state.fractal_params['zoom'], 0.1, 
                                   help="⚠️ Move carefully to avoid accidental changes")
                    center_real = st.slider("Center (Real)", -2.0, 2.0, st.session_state.fractal_params['center_real'], 0.001,
                                          help="⚠️ Move carefully to avoid accidental changes")
                    iterations = st.slider("Iterations", 50, 500, st.session_state.fractal_params['iterations'], 10,
                                         help="⚠️ Move carefully to avoid accidental changes")
            
            with col2:
                if input_method == "🎯 Precision (Number Input)":
                    st.write("**Center (Imaginary Part)**")
                    imag_col1, imag_col2, imag_col3 = st.columns([2, 1, 1])
                    with imag_col1:
                        center_imag = st.number_input("Imaginary", value=st.session_state.fractal_params['center_imag'], 
                                                    min_value=-2.0, max_value=2.0, step=0.001, format="%.6f", label_visibility="collapsed")
                    with imag_col2:
                        if st.button("↓", key="imag_down", help="Move down"):
                            center_imag = max(-2.0, center_imag - 0.01/st.session_state.fractal_params['zoom'])
                    with imag_col3:
                        if st.button("↑", key="imag_up", help="Move up"):
                            center_imag = min(2.0, center_imag + 0.01/st.session_state.fractal_params['zoom'])
                else:
                    center_imag = st.slider("Center (Imaginary)", -2.0, 2.0, st.session_state.fractal_params['center_imag'], 0.001,
                                          help="⚠️ Move carefully to avoid accidental changes")
                
                resolution = st.selectbox("Resolution", ["400x300", "600x450", "800x600", "1024x768"], index=1)
                colormap = st.selectbox("Color Scheme", ["hot", "viridis", "plasma", "magma", "inferno", "cool", "spring", "winter", "autumn"], 
                                      index=["hot", "viridis", "plasma", "magma", "inferno", "cool", "spring", "winter", "autumn"].index(st.session_state.current_colormap))
                mantra_idx = st.selectbox("Sanskrit Overlay", range(len(SANSKRIT_MANTRAS)), 
                                        index=st.session_state.current_mantra_idx,
                                        format_func=lambda x: SANSKRIT_MANTRAS[x][1])
                
                # Julia set parameter (only show for Julia type)
                if fractal_type == "julia":
                    st.markdown("**Julia Set Parameters:**")
                    julia_real = st.number_input("Julia C (Real)", value=-0.7, min_value=-2.0, max_value=2.0, step=0.01, format="%.6f")
                    julia_imag = st.number_input("Julia C (Imag)", value=0.27015, min_value=-2.0, max_value=2.0, step=0.01, format="%.6f")
            
            # Parse resolution
            width, height = map(int, resolution.split('x'))
            
            # Auto-update logic - Check if parameters actually changed
            current_params = {
                'zoom': zoom,
                'center_real': center_real,
                'center_imag': center_imag,
                'iterations': iterations,
                'fractal_type': fractal_type,
                'colormap': colormap,
                'mantra_idx': mantra_idx
            }
            
            # Update session state
            st.session_state.fractal_params.update({
                'zoom': zoom, 'center_real': center_real, 'center_imag': center_imag,
                'iterations': iterations, 'width': width, 'height': height
            })
            st.session_state.current_fractal_type = fractal_type
            st.session_state.current_colormap = colormap
            st.session_state.current_mantra_idx = mantra_idx
            
            # Auto-generate if enabled and parameters actually changed
            if st.session_state.auto_mode:
                params_changed = (
                    current_params['zoom'] != st.session_state.prev_params['zoom'] or
                    current_params['center_real'] != st.session_state.prev_params['center_real'] or
                    current_params['center_imag'] != st.session_state.prev_params['center_imag'] or
                    current_params['iterations'] != st.session_state.prev_params['iterations'] or
                    current_params['fractal_type'] != st.session_state.prev_params['fractal_type'] or
                    current_params['colormap'] != st.session_state.prev_params['colormap'] or
                    current_params['mantra_idx'] != st.session_state.prev_params['mantra_idx']
                )
                
                if params_changed:
                    with st.spinner("🔄 Auto-generating fractal..."):
                        fractal = generate_fractal(
                            fractal_type=st.session_state.current_fractal_type,
                            width=st.session_state.fractal_params['width'], 
                            height=st.session_state.fractal_params['height'], 
                            max_iter=st.session_state.fractal_params['iterations'],
                            zoom=st.session_state.fractal_params['zoom'], 
                            center_real=st.session_state.fractal_params['center_real'], 
                            center_imag=st.session_state.fractal_params['center_imag']
                        )
                        
                        # Add Sanskrit overlay
                        fractal_with_overlay = add_sanskrit_overlay(
                            fractal, 
                            st.session_state.current_mantra_idx, 
                            st.session_state.current_colormap
                        )
                        st.session_state.current_fractal = fractal_with_overlay
                        
                        # Update previous params to current
                        st.session_state.prev_params = current_params.copy()
                        st.success("✨ Auto-generated!")
            
            # Always update prev_params if not auto-generating  
            if not st.session_state.auto_mode:
                st.session_state.prev_params = current_params.copy()
    else:
        st.info("🔒 Controls are locked to prevent accidental changes. Click unlock to modify parameters.")
    
    # Quick navigation
    st.subheader("🧭 Quick Navigation")
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
    
    with nav_col1:
        if st.button("🔍 Zoom In 2x"):
            st.session_state.zoom_history.append(st.session_state.fractal_params.copy())
            st.session_state.fractal_params['zoom'] *= 2
            if st.session_state.auto_mode:
                with st.spinner("🔍 Zooming in..."):
                    fractal = generate_fractal(
                        fractal_type=st.session_state.current_fractal_type,
                        width=st.session_state.fractal_params['width'], 
                        height=st.session_state.fractal_params['height'], 
                        max_iter=st.session_state.fractal_params['iterations'],
                        zoom=st.session_state.fractal_params['zoom'], 
                        center_real=st.session_state.fractal_params['center_real'], 
                        center_imag=st.session_state.fractal_params['center_imag']
                    )
                    fractal_with_overlay = add_sanskrit_overlay(
                        fractal, st.session_state.current_mantra_idx, st.session_state.current_colormap
                    )
                    st.session_state.current_fractal = fractal_with_overlay
            st.rerun()
    
    with nav_col2:
        if st.button("🔍 Zoom Out 2x"):
            st.session_state.zoom_history.append(st.session_state.fractal_params.copy())
            st.session_state.fractal_params['zoom'] /= 2
            if st.session_state.auto_mode:
                with st.spinner("🔍 Zooming out..."):
                    fractal = generate_fractal(
                        fractal_type=st.session_state.current_fractal_type,
                        width=st.session_state.fractal_params['width'], 
                        height=st.session_state.fractal_params['height'], 
                        max_iter=st.session_state.fractal_params['iterations'],
                        zoom=st.session_state.fractal_params['zoom'], 
                        center_real=st.session_state.fractal_params['center_real'], 
                        center_imag=st.session_state.fractal_params['center_imag']
                    )
                    fractal_with_overlay = add_sanskrit_overlay(
                        fractal, st.session_state.current_mantra_idx, st.session_state.current_colormap
                    )
                    st.session_state.current_fractal = fractal_with_overlay
            st.rerun()
    
    with nav_col3:
        if st.button("⬆️ Move Up"):
            st.session_state.zoom_history.append(st.session_state.fractal_params.copy())
            st.session_state.fractal_params['center_imag'] += 0.1 / st.session_state.fractal_params['zoom']
            if st.session_state.auto_mode:
                with st.spinner("⬆️ Moving up..."):
                    fractal = generate_fractal(
                        fractal_type=st.session_state.current_fractal_type,
                        width=st.session_state.fractal_params['width'], 
                        height=st.session_state.fractal_params['height'], 
                        max_iter=st.session_state.fractal_params['iterations'],
                        zoom=st.session_state.fractal_params['zoom'], 
                        center_real=st.session_state.fractal_params['center_real'], 
                        center_imag=st.session_state.fractal_params['center_imag']
                    )
                    fractal_with_overlay = add_sanskrit_overlay(
                        fractal, st.session_state.current_mantra_idx, st.session_state.current_colormap
                    )
                    st.session_state.current_fractal = fractal_with_overlay
            st.rerun()
    
    with nav_col4:
        if st.button("⬇️ Move Down"):
            st.session_state.zoom_history.append(st.session_state.fractal_params.copy())
            st.session_state.fractal_params['center_imag'] -= 0.1 / st.session_state.fractal_params['zoom']
            if st.session_state.auto_mode:
                with st.spinner("⬇️ Moving down..."):
                    fractal = generate_fractal(
                        fractal_type=st.session_state.current_fractal_type,
                        width=st.session_state.fractal_params['width'], 
                        height=st.session_state.fractal_params['height'], 
                        max_iter=st.session_state.fractal_params['iterations'],
                        zoom=st.session_state.fractal_params['zoom'], 
                        center_real=st.session_state.fractal_params['center_real'], 
                        center_imag=st.session_state.fractal_params['center_imag']
                    )
                    fractal_with_overlay = add_sanskrit_overlay(
                        fractal, st.session_state.current_mantra_idx, st.session_state.current_colormap
                    )
                    st.session_state.current_fractal = fractal_with_overlay
            st.rerun()

with tab2:
    st.header("Audio Synthesis")
    
    # Audio parameter controls
    col1, col2 = st.columns(2)
    
    with col1:
        base_freq = st.slider("Base Frequency (Hz)", 50, 300, int(st.session_state.audio_params['base_freq']))
        harmony_freq = st.slider("Harmony Frequency (Hz)", 200, 800, int(st.session_state.audio_params['harmony_freq']))
    
    with col2:
        duration = st.slider("Duration (seconds)", 1, 30, st.session_state.audio_params['duration'])
        sample_rate = st.selectbox("Sample Rate", [22050, 44100], index=0)
    
    # Update session state
    st.session_state.audio_params.update({
        'base_freq': base_freq, 'harmony_freq': harmony_freq,
        'duration': duration, 'sample_rate': sample_rate
    })
    
    # Generate audio
    if st.button("🎵 Generate Audio", key="generate_audio"):
        with st.spinner("🎵 Synthesizing audio..."):
            audio_data, sample_rate = generate_audio(
                base_freq=base_freq, harmony_freq=harmony_freq,
                duration=duration, sample_rate=sample_rate
            )
            st.session_state.current_audio = (audio_data, sample_rate)
            st.success("🎵 Audio generated successfully!")
    
    # Display audio player
    if st.session_state.current_audio is not None:
        audio_data, sample_rate = st.session_state.current_audio
        
        # Create audio file for playback
        audio_bytes = create_audio_download(audio_data, sample_rate)
        st.audio(audio_bytes, format='audio/wav')
        
        # Show frequency information
        st.markdown("**Active Frequencies:**")
        st.write(f"- Base: {base_freq} Hz")
        st.write(f"- Harmony: {harmony_freq} Hz") 
        st.write(f"- Harmonics: {harmony_freq * 1.5:.1f} Hz, {harmony_freq * 2.0:.1f} Hz")
    else:
        st.info("Click 'Generate Audio' to create sound")

with tab3:
    st.header("Animation Generator")
    
    # Animation controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Animation Settings")
        anim_type = st.selectbox("Animation Type", 
            ["zoom_in", "zoom_out", "rotate_center", "parameter_sweep"],
            format_func=lambda x: x.replace('_', ' ').title())
        
        frames = st.slider("Number of Frames", 10, 100, 30)
        frame_rate = st.slider("Frame Rate (FPS)", 1, 30, 10)
        
        # Animation-specific parameters
        if anim_type in ["zoom_in", "zoom_out"]:
            zoom_factor = st.slider("Zoom Factor", 1.1, 10.0, 2.0)
        elif anim_type == "rotate_center":
            rotation_angle = st.slider("Rotation Angle (degrees)", 30, 360, 180)
        elif anim_type == "parameter_sweep":
            param_to_sweep = st.selectbox("Parameter to Sweep", ["center_real", "center_imag", "iterations"])
            sweep_range = st.slider("Sweep Range", 0.1, 2.0, 0.5)
    
    with col2:
        st.subheader("Preview Settings")
        preview_quality = st.selectbox("Preview Quality", ["Low (200x150)", "Medium (400x300)", "High (600x450)"])
        preview_frames = st.slider("Preview Frames", 5, 20, 10)
        
        if st.button("🎬 Generate Animation Preview"):
            st.info("Animation generation would start here...")
            # Animation generation logic would go here
    
    st.subheader("Animation Queue")
    if 'animation_queue' not in st.session_state:
        st.session_state.animation_queue = []
    
    if st.session_state.animation_queue:
        for i, anim in enumerate(st.session_state.animation_queue):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**{anim['type']}** - {anim['frames']} frames @ {anim['fps']} FPS")
            with col2:
                if st.button("Preview", key=f"preview_{i}"):
                    st.info("Preview would show here")
            with col3:
                if st.button("Delete", key=f"anim_del_{i}"):
                    st.session_state.animation_queue.pop(i)
                    st.rerun()
    else:
        st.info("No animations in queue. Create one above!")

with tab4:
    st.header("Gallery & Community")
    
    # Gallery section
    if st.session_state.gallery:
        st.subheader(f"My Fractals ({len(st.session_state.gallery)})")
        
        # Gallery display options
        display_mode = st.radio("Display Mode", ["Grid", "List"], horizontal=True)
        
        if display_mode == "Grid":
            # Grid layout for gallery
            cols = st.columns(3)
            for i, item in enumerate(st.session_state.gallery):
                with cols[i % 3]:
                    st.image(item['image'], width=200)
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Load", key=f"load_{i}"):
                            st.session_state.fractal_params.update(item['params'])
                            st.success("Loaded!")
                            st.rerun()
                    with col2:
                        if st.button("Delete", key=f"del_{i}"):
                            st.session_state.gallery.pop(i)
                            st.success("Deleted!")
                            st.rerun()
        else:
            # List layout for gallery
            for i, item in enumerate(st.session_state.gallery):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.image(item['image'], width=200)
                with col2:
                    if st.button("Load", key=f"load_{i}"):
                        st.session_state.fractal_params.update(item['params'])
                        st.success("Loaded!")
                        st.rerun()
                with col3:
                    if st.button("Delete", key=f"del_{i}"):
                        st.session_state.gallery.pop(i)
                        st.success("Deleted!")
                        st.rerun()
    else:
        st.info("No saved fractals yet. Generate and save some fractals to build your gallery!")
    
    # Save current fractal
    if st.button("💾 Save Current Fractal to Gallery"):
        if st.session_state.current_fractal is not None:
            gallery_item = {
                'timestamp': time.time(),
                'params': st.session_state.fractal_params.copy(),
                'image': st.session_state.current_fractal,
                'fractal_type': st.session_state.current_fractal_type,
                'colormap': st.session_state.current_colormap
            }
            st.session_state.gallery.append(gallery_item)
            st.success("💾 Saved to gallery!")
        else:
            st.warning("Generate a fractal first")
    
    # Bookmark section
    if st.session_state.bookmark_history:
        st.subheader(f"Bookmarked Locations ({len(st.session_state.bookmark_history)})")
        for i, bookmark in enumerate(st.session_state.bookmark_history):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{bookmark['name']}** - {bookmark['type'].title()}")
                st.write(f"Real: {bookmark['params']['center_real']:.4f}, Imag: {bookmark['params']['center_imag']:.4f}, Zoom: {bookmark['params']['zoom']:.1f}x")
            with col2:
                if st.button("Load", key=f"bookmark_{i}"):
                    st.session_state.fractal_params.update(bookmark['params'])
                    st.session_state.current_fractal_type = bookmark['type']
                    st.success("Bookmark loaded!")
                    st.rerun()
            with col3:
                if st.button("Delete", key=f"bookmark_del_{i}"):
                    st.session_state.bookmark_history.pop(i)
                    st.success("Bookmark deleted!")
                    st.rerun()
    
    # Community sharing section
    st.subheader("Community Sharing")
    if st.session_state.current_fractal is not None:
        params = st.session_state.fractal_params
        share_url = f"?type={st.session_state.current_fractal_type}&zoom={params['zoom']:.3f}&real={params['center_real']:.6f}&imag={params['center_imag']:.6f}&iter={params['iterations']}"
        st.code(share_url, language="text")
        st.info("📋 Copy this URL to share your fractal coordinates with others!")
    else:
        st.info("Generate a fractal to get a shareable URL")

with tab5:
    st.header("Settings & Performance")
    
    # Theme settings
    st.subheader("🎨 Appearance Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        new_theme_mode = st.selectbox("Theme Mode", ["Dark", "Light", "Auto"], 
                                     index=["Dark", "Light", "Auto"].index(st.session_state.theme_mode))
        
        # Apply theme change immediately
        if new_theme_mode != st.session_state.theme_mode:
            st.session_state.theme_mode = new_theme_mode
            st.success(f"🎨 Switched to {new_theme_mode} mode!")
            st.rerun()
            
        ui_scale = st.slider("UI Scale", 0.8, 1.2, 1.0, 0.1)
        animation_speed = st.slider("Animation Speed", 0.5, 2.0, 1.0, 0.1)
    
    with col2:
        auto_save = st.checkbox("Auto-save fractals", value=False)
        show_coordinates = st.checkbox("Show coordinates overlay", value=True)
        enable_sound = st.checkbox("Enable UI sound effects", value=False)
        
        # Theme preview
        st.markdown(f"**Current Theme**: {st.session_state.theme_mode}")
        if st.button("🔄 Refresh Theme"):
            st.rerun()
    
    # Performance settings
    st.subheader("⚡ Performance Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        render_quality = st.selectbox("Default Render Quality", ["Draft", "Standard", "High", "Ultra"], index=1)
        max_iterations = st.slider("Max Iterations Limit", 100, 1000, 500, 50)
        memory_limit = st.slider("Memory Limit (MB)", 100, 1000, 500, 50)
    
    with col2:
        cache_size = st.slider("Cache Size (fractals)", 5, 50, 20, 5)
        preview_quality = st.selectbox("Preview Quality", ["Low", "Medium", "High"], index=1)
        parallel_processing = st.checkbox("Enable parallel processing", value=True)
    
    # System information
    st.subheader("📊 System Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Session Duration", f"{time.time() - st.session_state.start_time:.0f}s")
        st.metric("Fractals Generated", st.session_state.performance_stats["total_fractals"])
    
    with col2:
        st.metric("Controls Status", "Locked" if st.session_state.locked_controls else "Unlocked")
        st.metric("Gallery Size", len(st.session_state.gallery))
    
    with col3:
        memory_usage = len(str(st.session_state)) // 1024
        st.metric("Memory Usage", f"{memory_usage}KB")
        if st.session_state.performance_stats["render_times"]:
            avg_render = np.mean(st.session_state.performance_stats["render_times"][-10:])
            st.metric("Avg Render Time", f"{avg_render:.2f}s")
        else:
            st.metric("Avg Render Time", "N/A")
    
    # Performance graph
    if st.session_state.performance_stats["render_times"]:
        st.subheader("📈 Performance History")
        st.line_chart(st.session_state.performance_stats["render_times"][-20:])
    
    # Reset and maintenance
    st.subheader("🔧 Maintenance")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear Gallery"):
            st.session_state.gallery = []
            st.success("Gallery cleared!")
    
    with col2:
        if st.button("📊 Reset Performance Stats"):
            st.session_state.performance_stats = {"render_times": [], "total_fractals": 0}
            st.success("Performance stats reset!")
    
    with col3:
        if st.button("🔄 Reset All Settings"):
            # Reset to defaults but keep current fractal
            current_fractal = st.session_state.current_fractal
            for key in list(st.session_state.keys()):
                if key != 'current_fractal':
                    del st.session_state[key]
            st.session_state.current_fractal = current_fractal
            st.success("Settings reset!")
            st.rerun()

with tab6:
    st.header("Gallery & Settings")
    
    # Gallery section
    if st.session_state.gallery:
        st.subheader(f"Saved Fractals ({len(st.session_state.gallery)})")
        
        for i, item in enumerate(st.session_state.gallery):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.image(item['image'], width=200)
            with col2:
                if st.button("Load", key=f"load_{i}"):
                    st.session_state.fractal_params.update(item['params'])
                    st.success("Loaded!")
                    st.rerun()
            with col3:
                if st.button("Delete", key=f"del_{i}"):
                    st.session_state.gallery.pop(i)
                    st.success("Deleted!")
                    st.rerun()
    else:
        st.info("No saved fractals yet.")
    
    # Save current fractal
    if st.button("💾 Save Current Fractal"):
        if st.session_state.current_fractal is not None:
            gallery_item = {
                'timestamp': time.time(),
                'params': st.session_state.fractal_params.copy(),
                'image': st.session_state.current_fractal
            }
            st.session_state.gallery.append(gallery_item)
            st.success("💾 Saved to gallery!")
        else:
            st.warning("Generate a fractal first")
    
    # Bookmark section
    if st.session_state.bookmark_history:
        st.subheader(f"Bookmarked Locations ({len(st.session_state.bookmark_history)})")
        for i, bookmark in enumerate(st.session_state.bookmark_history):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{bookmark['name']}** - {bookmark['type'].title()}")
                st.write(f"Real: {bookmark['params']['center_real']:.4f}, Imag: {bookmark['params']['center_imag']:.4f}, Zoom: {bookmark['params']['zoom']:.1f}x")
            with col2:
                if st.button("Load", key=f"bookmark_{i}"):
                    st.session_state.fractal_params.update(bookmark['params'])
                    st.session_state.current_fractal_type = bookmark['type']
                    st.success("Bookmark loaded!")
                    st.rerun()
            with col3:
                if st.button("Delete", key=f"bookmark_del_{i}"):
                    st.session_state.bookmark_history.pop(i)
                    st.success("Bookmark deleted!")
                    st.rerun()
    
with tab6:
    st.header("Export & Download")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Visual Export")
        
        if st.session_state.current_fractal is not None:
            # Export options
            export_format = st.selectbox("Image Format", ["PNG", "JPEG", "TIFF"], index=0)
            export_quality = st.selectbox("Export Quality", ["Current Resolution", "HD (1920x1440)", "4K (3840x2880)", "8K (7680x5760)"], index=0)
            include_overlay = st.checkbox("Include Sanskrit overlay", value=True)
            
            # Generate export
            if st.button("🎨 Generate Export"):
                if export_quality == "Current Resolution":
                    export_image = st.session_state.current_fractal
                else:
                    # Generate high-resolution version
                    with st.spinner("Generating high-resolution fractal..."):
                        if export_quality == "HD (1920x1440)":
                            width, height = 1920, 1440
                        elif export_quality == "4K (3840x2880)":
                            width, height = 3840, 2880
                        else:  # 8K
                            width, height = 7680, 5760
                        
                        hd_fractal = generate_fractal(
                            fractal_type=st.session_state.current_fractal_type,
                            width=width, height=height,
                            max_iter=st.session_state.fractal_params['iterations'],
                            zoom=st.session_state.fractal_params['zoom'],
                            center_real=st.session_state.fractal_params['center_real'],
                            center_imag=st.session_state.fractal_params['center_imag']
                        )
                        
                        if include_overlay:
                            export_image = add_sanskrit_overlay(hd_fractal, st.session_state.current_mantra_idx, st.session_state.current_colormap)
                        else:
                            # Apply colormap without overlay
                            fractal_norm = (hd_fractal - hd_fractal.min()) / (hd_fractal.max() - hd_fractal.min())
                            cmap = plt.get_cmap(st.session_state.current_colormap)
                            colored = cmap(fractal_norm)
                            export_image = (colored[:, :, :3] * 255).astype(np.uint8)
                
                # Convert image for download
                img = Image.fromarray(export_image)
                buf = io.BytesIO()
                
                if export_format == "PNG":
                    img.save(buf, format='PNG', optimize=True)
                    mime_type = "image/png"
                    file_ext = "png"
                elif export_format == "JPEG":
                    img = img.convert('RGB')  # JPEG doesn't support alpha
                    img.save(buf, format='JPEG', quality=95, optimize=True)
                    mime_type = "image/jpeg"
                    file_ext = "jpg"
                else:  # TIFF
                    img.save(buf, format='TIFF', compression='lzw')
                    mime_type = "image/tiff"
                    file_ext = "tiff"
                
                buf.seek(0)
                
                st.download_button(
                    label=f"💾 Download {export_format}",
                    data=buf.getvalue(),
                    file_name=f"aoin_fractal_{st.session_state.current_fractal_type}_{int(time.time())}.{file_ext}",
                    mime=mime_type
                )
                st.success(f"Export ready! File size: ~{len(buf.getvalue())//1024}KB")
        else:
            st.info("Generate a fractal first")
    
    with col2:
        st.subheader("🎵 Audio Export")
        
        if st.session_state.current_audio is not None:
            audio_data, sample_rate = st.session_state.current_audio
            
            # Audio export options
            audio_format = st.selectbox("Audio Format", ["WAV", "MP3 (simulated)"], index=0)
            audio_quality = st.selectbox("Audio Quality", ["22kHz", "44kHz", "48kHz"], index=1)
            
            # Generate different quality
            if audio_quality != "22kHz":
                target_rate = 44100 if audio_quality == "44kHz" else 48000
                with st.spinner("Resampling audio..."):
                    # Simple resampling (in production, use scipy.signal.resample)
                    ratio = target_rate / sample_rate
                    new_length = int(len(audio_data) * ratio)
                    resampled_audio = np.interp(np.linspace(0, len(audio_data), new_length), 
                                              np.arange(len(audio_data)), audio_data)
                    export_audio = resampled_audio
                    export_sample_rate = target_rate
            else:
                export_audio = audio_data
                export_sample_rate = sample_rate
            
            audio_bytes = create_audio_download(export_audio, export_sample_rate)
            
            st.download_button(
                label=f"🎵 Download {audio_format}",
                data=audio_bytes,
                file_name=f"aoin_audio_{int(st.session_state.audio_params['base_freq'])}Hz_{int(time.time())}.wav",
                mime="audio/wav"
            )
            
            # Audio info
            duration = len(export_audio) / export_sample_rate
            st.info(f"Duration: {duration:.1f}s | Sample Rate: {export_sample_rate}Hz | Size: ~{len(audio_bytes)//1024}KB")
        else:
            st.info("Generate audio first")
    
    # Bulk export section
    st.subheader("📦 Batch Export")
    
    if st.session_state.gallery:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Gallery**: {len(st.session_state.gallery)} fractals")
            batch_format = st.selectbox("Batch Format", ["PNG", "JPEG"], index=0, key="batch_format")
            batch_quality = st.selectbox("Batch Quality", ["Current", "HD"], index=0, key="batch_quality")
        
        with col2:
            include_metadata = st.checkbox("Include metadata files", value=True)
            zip_download = st.checkbox("Create ZIP archive", value=True)
        
        if st.button("📦 Export All Fractals"):
            st.info("Batch export would generate all gallery fractals with selected settings")
            # In a real implementation, this would create a ZIP file with all fractals
    else:
        st.info("No fractals in gallery for batch export")
    
    # Settings export
    st.subheader("⚙️ Settings Export")
    
    export_options = st.multiselect("Export Settings", 
        ["Fractal Parameters", "Audio Parameters", "Gallery", "Bookmarks", "Performance Stats"],
        default=["Fractal Parameters", "Audio Parameters"])
    
    if st.button("📋 Export Settings"):
        export_data = {"timestamp": time.time(), "app_version": "Aoin's Fractal Studio v2.0"}
        
        if "Fractal Parameters" in export_options:
            export_data["fractal_params"] = st.session_state.fractal_params
            export_data["fractal_type"] = st.session_state.current_fractal_type
            export_data["colormap"] = st.session_state.current_colormap
            export_data["mantra_index"] = st.session_state.current_mantra_idx
        
        if "Audio Parameters" in export_options:
            export_data["audio_params"] = st.session_state.audio_params
        
        if "Gallery" in export_options:
            # Export gallery metadata only (not images)
            gallery_metadata = []
            for item in st.session_state.gallery:
                metadata = {
                    "timestamp": item["timestamp"],
                    "params": item["params"],
                    "fractal_type": item.get("fractal_type", "mandelbrot"),
                    "colormap": item.get("colormap", "hot")
                }
                gallery_metadata.append(metadata)
            export_data["gallery_metadata"] = gallery_metadata
        
        if "Bookmarks" in export_options:
            export_data["bookmarks"] = st.session_state.bookmark_history
        
        if "Performance Stats" in export_options:
            export_data["performance_stats"] = st.session_state.performance_stats
        
        json_str = json.dumps(export_data, indent=2)
        st.download_button(
            label="💾 Download Settings JSON",
            data=json_str,
            file_name=f"aoin_settings_{int(time.time())}.json",
            mime="application/json"
        )
        
        st.success(f"Settings exported with {len(export_options)} categories!")

# Footer
st.markdown("---")
st.markdown("**💙 Aoin's Fractal Studio** - Mathematical visualization and audio synthesis")
st.markdown("Built with Streamlit • Mobile optimized interface • Dual input controls")
