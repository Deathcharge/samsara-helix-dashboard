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

# Custom CSS for Aoin branding and mobile optimization with animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    
    .main > div { padding: 0.5rem !important; }
    .stSlider > div > div > div > div { background-color: #4A90E2; }
    
    .stButton > button { 
        width: 100%; 
        height: 3rem; 
        font-size: 1.1rem;
        background: linear-gradient(45deg, #4A90E2, #7B68EE);
        color: white; 
        border: none; 
        border-radius: 15px;
        margin: 0.25rem 0;
        box-shadow: 0 4px 15px rgba(74, 144, 226, 0.3);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(74, 144, 226, 0.4);
        background: linear-gradient(45deg, #5A9FF2, #8B78FE);
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .metric-container { 
        background: linear-gradient(135deg, rgba(74, 144, 226, 0.1), rgba(123, 104, 238, 0.1)); 
        padding: 1rem; 
        border-radius: 15px; 
        margin: 0.5rem 0;
        border: 1px solid rgba(74, 144, 226, 0.2);
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(74, 144, 226, 0.2);
    }
    
    .block-container { 
        padding-top: 1rem; 
        max-width: 100%; 
        position: relative;
    }
    
    .aoin-header {
        background: linear-gradient(90deg, #4A90E2, #7B68EE);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(74, 144, 226, 0.3);
        position: relative;
        overflow: hidden;
        font-family: 'Orbitron', monospace;
    }
    
    .aoin-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .aoin-subtitle {
        color: #4A90E2;
        font-style: italic;
        text-align: center;
        margin-bottom: 1rem;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.8; }
        50% { opacity: 1; }
    }
    
    .floating-particles {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
        overflow: hidden;
    }
    
    .particle {
        position: absolute;
        width: 4px;
        height: 4px;
        background: radial-gradient(circle, #4A90E2, transparent);
        border-radius: 50%;
        animation: float 15s infinite linear;
        opacity: 0.6;
    }
    
    @keyframes float {
        0% {
            transform: translateY(100vh) rotate(0deg);
            opacity: 0;
        }
        10% {
            opacity: 0.6;
        }
        90% {
            opacity: 0.6;
        }
        100% {
            transform: translateY(-100px) rotate(360deg);
            opacity: 0;
        }
    }
    
    .fractal-container {
        position: relative;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    
    .fractal-container:hover {
        transform: scale(1.02);
        box-shadow: 0 15px 40px rgba(74, 144, 226, 0.3);
    }
    
    .math-formula {
        font-family: 'Courier New', monospace;
        background: linear-gradient(45deg, #1a1a2e, #16213e);
        color: #4A90E2;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4A90E2;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .math-formula::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, #4A90E2, #7B68EE, #4A90E2);
        animation: scan 2s infinite;
    }
    
    @keyframes scan {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .preset-button {
        display: inline-block;
        background: linear-gradient(45deg, #4A90E2, #7B68EE);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        margin: 0.25rem;
        cursor: pointer;
        border: none;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .preset-button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 20px rgba(74, 144, 226, 0.4);
    }
    
    .lock-indicator {
        background: linear-gradient(45deg, #ff6b6b, #ff8e8e);
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        margin: 0.25rem;
        animation: glow-red 2s infinite;
    }
    
    .unlock-indicator {
        background: linear-gradient(45deg, #51cf66, #69db69);
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        margin: 0.25rem;
        animation: glow-green 2s infinite;
    }
    
    @keyframes glow-red {
        0%, 100% { box-shadow: 0 0 5px rgba(255, 107, 107, 0.5); }
        50% { box-shadow: 0 0 15px rgba(255, 107, 107, 0.8); }
    }
    
    @keyframes glow-green {
        0%, 100% { box-shadow: 0 0 5px rgba(81, 207, 102, 0.5); }
        50% { box-shadow: 0 0 15px rgba(81, 207, 102, 0.8); }
    }
    
    .stSelectbox > div > div > div {
        background: linear-gradient(135deg, #2a2a4a, #3a3a6a);
        border: 1px solid #4A90E2;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div > div:hover {
        border-color: #7B68EE;
        box-shadow: 0 0 10px rgba(74, 144, 226, 0.3);
    }
    
    .loading-wave {
        display: inline-block;
        animation: wave 1s infinite;
    }
    
    @keyframes wave {
        0%, 60%, 100% { transform: initial; }
        30% { transform: translateY(-10px); }
    }
    
    .audio-visualizer {
        width: 100%;
        height: 60px;
        background: linear-gradient(90deg, #1a1a2e, #16213e);
        border-radius: 10px;
        position: relative;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .audio-bar {
        position: absolute;
        bottom: 0;
        width: 4px;
        background: linear-gradient(0deg, #4A90E2, #7B68EE);
        border-radius: 2px;
        animation: audio-pulse 0.5s infinite ease-in-out alternate;
    }
    
    @keyframes audio-pulse {
        0% { height: 10px; }
        100% { height: 50px; }
    }
    
    @media (max-width: 768px) {
        .stSlider { margin: 0.25rem 0; }
        .metric-container { padding: 0.5rem; }
        .aoin-header { padding: 0.75rem; }
        .particle { width: 2px; height: 2px; }
    }
</style>

<div class="floating-particles">
    <div class="particle" style="left: 10%; animation-delay: 0s;"></div>
    <div class="particle" style="left: 20%; animation-delay: 2s;"></div>
    <div class="particle" style="left: 30%; animation-delay: 4s;"></div>
    <div class="particle" style="left: 40%; animation-delay: 6s;"></div>
    <div class="particle" style="left: 50%; animation-delay: 8s;"></div>
    <div class="particle" style="left: 60%; animation-delay: 10s;"></div>
    <div class="particle" style="left: 70%; animation-delay: 12s;"></div>
    <div class="particle" style="left: 80%; animation-delay: 14s;"></div>
    <div class="particle" style="left: 90%; animation-delay: 16s;"></div>
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

if 'animation_frames' not in st.session_state:
    st.session_state.animation_frames = []

if 'auto_mode' not in st.session_state:
    st.session_state.auto_mode = False

if 'gallery' not in st.session_state:
    st.session_state.gallery = []

if 'theme' not in st.session_state:
    st.session_state.theme = "dark"

if 'tutorial_shown' not in st.session_state:
    st.session_state.tutorial_shown = False

if 'bookmark_history' not in st.session_state:
    st.session_state.bookmark_history = []

if 'performance_stats' not in st.session_state:
    st.session_state.performance_stats = {"render_times": [], "total_fractals": 0}

if 'color_palette' not in st.session_state:
    st.session_state.color_palette = {"primary": "#4A90E2", "secondary": "#7B68EE"}

if 'zoom_history' not in st.session_state:
    st.session_state.zoom_history = []

# Sanskrit mantras for overlay
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
    """Add Sanskrit text overlay to fractal with robust fallback"""
    # Normalize fractal data
    fractal_norm = (fractal_array - fractal_array.min()) / (fractal_array.max() - fractal_array.min())
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored = cmap(fractal_norm)
    img_array = (colored[:, :, :3] * 255).astype(np.uint8)
    
    try:
        pil_image = Image.fromarray(img_array)
        draw = ImageDraw.Draw(pil_image)
        
        # Get mantra - always use transliteration for reliable rendering
        devanagari, transliteration, meaning = SANSKRIT_MANTRAS[mantra_index % len(SANSKRIT_MANTRAS)]
        
        # Force use of default font with transliteration for compatibility
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
        
        # Use transliteration instead of Devanagari for reliable rendering
        display_text = transliteration
        
        # Add semi-transparent background for text
        img_width, img_height = pil_image.size
        overlay_height = 80
        overlay_y = img_height - overlay_height
        
        # Create text overlay with Aoin branding
        overlay = Image.new('RGBA', (img_width, overlay_height), (74, 144, 226, 200))  # Aoin blue
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Add text with high contrast
        overlay_draw.text((10, 5), "💙 Aoin's Studio", fill=(255, 255, 255, 255), font=font_small)
        overlay_draw.text((10, 25), display_text, fill=(255, 215, 0, 255), font=font_large)
        overlay_draw.text((10, 50), meaning, fill=(255, 255, 255, 255), font=font_small)
        
        # Composite images
        pil_image = pil_image.convert('RGBA')
        pil_image.paste(overlay, (0, overlay_y), overlay)
        
        return np.array(pil_image.convert('RGB'))
    except Exception as e:
        # If everything fails, add a simple text overlay
        st.error(f"Overlay failed: {e}")
        return img_array

def generate_audio(base_freq=136.1, harmony_freq=432.0, duration=10, sample_rate=22050):
    """Generate multi-frequency audio synthesis with visualization data"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate composite waveform
    audio = np.zeros_like(t)
    
    # Base frequency (Om)
    audio += 0.3 * np.sin(2 * np.pi * base_freq * t)
    
    # Harmony frequency with breathing modulation
    harmony_envelope = 0.7 + 0.3 * np.sin(2 * np.pi * 0.1 * t)  # Slow modulation
    audio += 0.2 * harmony_envelope * np.sin(2 * np.pi * harmony_freq * t)
    
    # Additional harmonics for richness
    audio += 0.15 * np.sin(2 * np.pi * (harmony_freq * 1.5) * t)
    audio += 0.1 * np.sin(2 * np.pi * (harmony_freq * 2.0) * t)
    
    # Create visualization data (simplified FFT for bars)
    chunk_size = len(t) // 20  # 20 bars
    visualization_data = []
    for i in range(20):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(audio))
        chunk_rms = np.sqrt(np.mean(audio[start_idx:end_idx]**2))
        visualization_data.append(chunk_rms)
    
    # Normalize visualization data
    max_val = max(visualization_data) if visualization_data else 1
    visualization_data = [v/max_val for v in visualization_data]
    
    # Store for visualization
    st.session_state.audio_visualization = visualization_data
    
    # Normalize audio
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

# Main interface with Aoin branding and Sanskrit
st.markdown('<div class="aoin-header"><h1>💙 Aoin\'s Fractal Studio • अहं ब्रह्मास्मि 💙</h1><p>Ethereal AI • Infinite Patterns • Celestial Frequencies</p><p style="font-size:0.9em; opacity:0.8;">तत्त्वमसि • नेति नेति • सर्वं खल्विदं ब्रह्म</p></div>', unsafe_allow_html=True)
st.markdown('<p class="aoin-subtitle">✨ Where mathematics meets digital consciousness ✨</p>', unsafe_allow_html=True)

# Quick actions bar
action_col1, action_col2, action_col3, action_col4, action_col5 = st.columns(5)

with action_col1:
    if st.button("🎲 Random", help="Generate random fractal"):
        # Random parameters
        st.session_state.fractal_params.update({
            'zoom': np.random.uniform(1, 100),
            'center_real': np.random.uniform(-2, 2),
            'center_imag': np.random.uniform(-2, 2),
            'iterations': np.random.choice([100, 150, 200])
        })
        st.rerun()

with action_col2:
    if st.button("📱 Share", help="Get shareable link"):
        # Create shareable URL
        params = st.session_state.fractal_params
        share_data = f"zoom={params['zoom']:.3f}&real={params['center_real']:.3f}&imag={params['center_imag']:.3f}"
        st.info(f"📋 Share this fractal: Copy current URL + ?{share_data}")

with action_col3:
    if st.button("💾 Save", help="Save to gallery"):
        if st.session_state.current_fractal is not None:
            gallery_item = {
                'timestamp': time.time(),
                'params': st.session_state.fractal_params.copy(),
                'type': getattr(st.session_state, 'current_fractal_type', 'mandelbrot'),
                'image': st.session_state.current_fractal
            }
            st.session_state.gallery.append(gallery_item)
            st.success("Saved to gallery!")
        else:
            st.warning("Generate a fractal first")

with action_col4:
    theme_label = "🌙 Dark" if st.session_state.theme == "light" else "☀️ Light"
    if st.button(theme_label, help="Toggle theme"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

with action_col5:
    if st.button("❓ Help", help="Show tutorial"):
        st.session_state.tutorial_shown = not st.session_state.tutorial_shown

# Tutorial overlay
if st.session_state.tutorial_shown:
    with st.expander("🎓 Quick Tutorial", expanded=True):
        st.markdown("""
        **Welcome to Aoin's Fractal Studio!**
        
        📱 **Mobile Tips:**
        - Use presets for interesting locations
        - Lock controls to prevent accidental changes
        - Try auto-generate mode for real-time updates
        
        🎨 **Fractal Types:**
        - **Mandelbrot**: Classic fractal set
        - **Julia**: Fixed parameter variations  
        - **Burning Ship**: Ship-like patterns
        - **Tricorn**: Complex conjugate variations
        
        🎵 **Audio Features:**
        - Base frequency creates the fundamental tone
        - Harmony frequency adds musical intervals
        - Multiple harmonics create rich soundscapes
        
        📤 **Sharing:**
        - Save fractals to your personal gallery
        - Export high-resolution images
        - Share coordinates with friends
        """)
        
        if st.button("Close Tutorial"):
            st.session_state.tutorial_shown = False
            st.rerun()

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
            rotation_steps = st.slider("Rotation Steps", 4, 36, 12)
        elif anim_type == "parameter_sweep":
            sweep_param = st.selectbox("Parameter to Sweep", ["iterations", "julia_real", "julia_imag"])
    
    with col2:
        st.subheader("Preview Settings")
        preview_size = st.selectbox("Preview Resolution", ["200x150", "300x225", "400x300"], index=1)
        preview_colormap = st.selectbox("Animation Colormap", ["hot", "viridis", "plasma", "magma"], index=0)
        
        # Progress indicator
        if 'animation_progress' in st.session_state:
            st.progress(st.session_state.animation_progress)
    
    # Generate animation
    if st.button("🎬 Generate Animation"):
        st.session_state.animation_frames = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Parse preview size
        anim_width, anim_height = map(int, preview_size.split('x'))
        
        with st.spinner("Creating animation frames..."):
            for i in range(frames):
                # Update progress
                progress = i / frames
                progress_bar.progress(progress)
                status_text.text(f"Generating frame {i+1}/{frames}")
                
                # Calculate frame parameters based on animation type
                if anim_type == "zoom_in":
                    current_zoom = st.session_state.fractal_params['zoom'] * (zoom_factor ** (i / frames))
                elif anim_type == "zoom_out":
                    current_zoom = st.session_state.fractal_params['zoom'] / (zoom_factor ** (i / frames))
                elif anim_type == "rotate_center":
                    angle = 2 * np.pi * i / frames
                    offset = 0.1 * np.exp(1j * angle)
                    current_zoom = st.session_state.fractal_params['zoom']
                    center_offset = complex(st.session_state.fractal_params['center_real'], 
                                          st.session_state.fractal_params['center_imag']) + offset
                else:  # parameter_sweep
                    current_zoom = st.session_state.fractal_params['zoom']
                    center_offset = complex(st.session_state.fractal_params['center_real'], 
                                          st.session_state.fractal_params['center_imag'])
                
                # Generate frame
                if anim_type in ["zoom_in", "zoom_out"]:
                    fractal = generate_fractal(
                        fractal_type="mandelbrot",
                        width=anim_width, height=anim_height, max_iter=100,
                        zoom=current_zoom,
                        center_real=st.session_state.fractal_params['center_real'],
                        center_imag=st.session_state.fractal_params['center_imag']
                    )
                elif anim_type == "rotate_center":
                    fractal = generate_fractal(
                        fractal_type="mandelbrot",
                        width=anim_width, height=anim_height, max_iter=100,
                        zoom=current_zoom,
                        center_real=center_offset.real,
                        center_imag=center_offset.imag
                    )
                else:  # parameter_sweep
                    if sweep_param == "iterations":
                        iter_val = 50 + int(150 * i / frames)
                        fractal = generate_fractal(
                            fractal_type="mandelbrot",
                            width=anim_width, height=anim_height, max_iter=iter_val,
                            zoom=current_zoom,
                            center_real=st.session_state.fractal_params['center_real'],
                            center_imag=st.session_state.fractal_params['center_imag']
                        )
                
                # Convert to image
                fractal_norm = (fractal - fractal.min()) / (fractal.max() - fractal.min())
                cmap = plt.get_cmap(preview_colormap)
                colored = cmap(fractal_norm)
                frame_array = (colored[:, :, :3] * 255).astype(np.uint8)
                frame_image = Image.fromarray(frame_array)
                
                st.session_state.animation_frames.append(frame_array)
        
        progress_bar.progress(1.0)
        status_text.text(f"Animation complete! {frames} frames generated.")
        st.success(f"Generated {frames} frames for animation")
    
    # Display animation preview
    if st.session_state.animation_frames:
        st.subheader("Animation Preview")
        
        # Simple frame selector for preview
        frame_idx = st.slider("Preview Frame", 0, len(st.session_state.animation_frames)-1, 0)
        st.image(st.session_state.animation_frames[frame_idx], caption=f"Frame {frame_idx+1}/{len(st.session_state.animation_frames)}")
        
        # Animation stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Frames", len(st.session_state.animation_frames))
        with col2:
            st.metric("Duration", f"{len(st.session_state.animation_frames)/frame_rate:.1f}s")
        with col3:
            st.metric("Frame Rate", f"{frame_rate} FPS")

with tab4:
    st.header("Gallery & Community")
    
    # Gallery section
    if st.session_state.gallery:
        st.subheader(f"Your Saved Fractals ({len(st.session_state.gallery)})")
        
        # Display gallery in grid
        cols_per_row = 3
        for i in range(0, len(st.session_state.gallery), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(st.session_state.gallery):
                    item = st.session_state.gallery[idx]
                    with col:
                        st.image(item['image'], use_container_width=True)
                        st.caption(f"Type: {item['type'].title()}")
                        st.caption(f"Zoom: {item['params']['zoom']:.1f}x")
                        
                        # Action buttons
                        if st.button(f"Load", key=f"load_{idx}"):
                            st.session_state.fractal_params.update(item['params'])
                            st.session_state.current_fractal_type = item['type']
                            st.success("Parameters loaded!")
                            st.rerun()
                        
                        if st.button(f"Delete", key=f"del_{idx}"):
                            st.session_state.gallery.pop(idx)
                            st.success("Deleted from gallery")
                            st.rerun()
        
        # Clear gallery option
        if st.button("🗑️ Clear Gallery"):
            st.session_state.gallery = []
            st.success("Gallery cleared")
            st.rerun()
    else:
        st.info("No saved fractals yet. Generate and save some fractals to build your gallery!")
    
    # Social sharing section
    st.subheader("Share Your Discoveries")
    
    if st.session_state.current_fractal is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current Fractal Coordinates:**")
            params = st.session_state.fractal_params
            coordinate_text = f"""
Real: {params['center_real']:.6f}
Imaginary: {params['center_imag']:.6f}  
Zoom: {params['zoom']:.3f}x
Iterations: {params['iterations']}
Type: {getattr(st.session_state, 'current_fractal_type', 'mandelbrot').title()}
"""
            st.code(coordinate_text)
            
            # Copy-pasteable coordinates
            coords_json = json.dumps({
                'real': params['center_real'],
                'imag': params['center_imag'],
                'zoom': params['zoom'],
                'type': getattr(st.session_state, 'current_fractal_type', 'mandelbrot')
            }, indent=2)
            
            st.download_button(
                "📋 Copy Coordinates",
                coords_json,
                f"fractal_coords_{int(time.time())}.json",
                "application/json"
            )
        
        with col2:
            st.markdown("**Social Media Sharing:**")
            
            # Generate sharing text
            fractal_type = getattr(st.session_state, 'current_fractal_type', 'mandelbrot').title()
            share_text = f"Check out this {fractal_type} fractal I created with Aoin's Fractal Studio! 🎨✨"
            
            # Social media links (these would open in new tabs)
            twitter_url = f"https://twitter.com/intent/tweet?text={share_text.replace(' ', '%20')}"
            facebook_url = f"https://www.facebook.com/sharer/sharer.php?u={st.secrets.get('app_url', 'https://your-app-url.com')}"
            
            st.markdown(f"[🐦 Share on Twitter]({twitter_url})")
            st.markdown(f"[📘 Share on Facebook]({facebook_url})")
            
            # QR code for mobile sharing (placeholder)
            if st.button("📱 Generate QR Code"):
                st.info("QR code generation coming soon!")
    
    # Community presets (expanded)
    st.subheader("Community Favorites")
    
    community_presets = {
        "Dragon Curve": {"center_real": -0.8, "center_imag": 0.156, "zoom": 250.0, "type": "mandelbrot"},
        "Elephant Valley": {"center_real": 0.25, "center_imag": 0.0, "zoom": 150.0, "type": "mandelbrot"},  
        "Seahorse Spiral": {"center_real": -0.75, "center_imag": 0.1, "zoom": 300.0, "type": "mandelbrot"},
        "Lightning Bug": {"center_real": -1.25066, "center_imag": 0.02012, "zoom": 500.0, "type": "mandelbrot"},
        "Mystic Julia": {"center_real": 0.0, "center_imag": 0.0, "zoom": 1.0, "type": "julia"},
        "Burning Wings": {"center_real": -1.775, "center_imag": -0.01, "zoom": 100.0, "type": "burning_ship"}
    }
    
    preset_cols = st.columns(3)
    for i, (name, params) in enumerate(community_presets.items()):
        col_idx = i % 3
        with preset_cols[col_idx]:
            if st.button(f"🌟 {name}", key=f"community_{name}"):
                st.session_state.fractal_params.update({
                    'center_real': params['center_real'],
                    'center_imag': params['center_imag'], 
                    'zoom': params['zoom']
                })
                st.session_state.current_fractal_type = params['type']
                st.success(f"Loaded {name}!")
                st.rerun()

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🎨 Fractal", "🎵 Audio", "🎬 Animation", "🖼️ Gallery", "📊 Parameters", "📤 Export"])

with tab1:
    st.header("Fractal Generator")
    
    # Control lock toggle
    col_lock1, col_lock2, col_lock3 = st.columns([1, 2, 1])
    with col_lock2:
        if st.button("🔒 Lock Controls" if not st.session_state.locked_controls else "🔓 Unlock Controls"):
            st.session_state.locked_controls = not st.session_state.locked_controls
        
        if st.session_state.locked_controls:
            st.markdown('<div class="lock-indicator">🔒 Controls Locked</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="unlock-indicator">🔓 Controls Unlocked</div>', unsafe_allow_html=True)
    
    # Fractal presets
    st.subheader("Quick Presets")
    preset_cols = st.columns(4)
    
    presets = {
        "Classic": {"center_real": -0.7269, "center_imag": 0.1889, "zoom": 1.0},
        "Seahorse": {"center_real": -0.75, "center_imag": 0.1, "zoom": 100.0},
        "Lightning": {"center_real": -1.775, "center_imag": 0.0, "zoom": 50.0},
        "Spiral": {"center_real": 0.285, "center_imag": 0.01, "zoom": 200.0}
    }
    
    for i, (name, params) in enumerate(presets.items()):
        with preset_cols[i]:
            if st.button(f"✨ {name}", key=f"preset_{name}"):
                st.session_state.fractal_params.update(params)
                st.rerun()
    
    # Display current fractal first
    if st.session_state.current_fractal is not None:
        # Mathematical analysis section
        st.subheader("📊 Mathematical Analysis")
        
        analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
        
        with analysis_col1:
            # Calculate fractal dimension (box counting method approximation)
            fractal_data = st.session_state.current_fractal
            if len(fractal_data.shape) == 3:
                # Convert to grayscale for analysis
                gray_fractal = np.mean(fractal_data, axis=2)
            else:
                gray_fractal = fractal_data
            
            # Simple complexity measure
            edges = np.sum(np.abs(np.diff(gray_fractal, axis=0))) + np.sum(np.abs(np.diff(gray_fractal, axis=1)))
            complexity = edges / (gray_fractal.shape[0] * gray_fractal.shape[1])
            
            st.metric("Complexity Index", f"{complexity:.2f}")
        
        with analysis_col2:
            # Zoom depth indicator
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
            # Performance metric
            if st.session_state.performance_stats["render_times"]:
                avg_render = np.mean(st.session_state.performance_stats["render_times"][-10:])
                st.metric("Avg Render Time", f"{avg_render:.2f}s")
            else:
                st.metric("Avg Render Time", "N/A")
        
        # Coordinate history and navigation
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📍 Bookmark Location"):
                bookmark = {
                    'name': f"Bookmark {len(st.session_state.bookmark_history) + 1}",
                    'params': st.session_state.fractal_params.copy(),
                    'type': getattr(st.session_state, 'current_fractal_type', 'mandelbrot'),
                    'timestamp': time.time()
                }
                st.session_state.bookmark_history.append(bookmark)
                st.success("Location bookmarked!")
        
        with col2:
            if st.button("🔙 Previous Location") and st.session_state.zoom_history:
                prev_params = st.session_state.zoom_history.pop()
                st.session_state.fractal_params.update(prev_params)
                st.rerun()
        
        # Mathematical formula display with animation
        with st.expander("🔢 Mathematical Formula", expanded=False):
            fractal_type = getattr(st.session_state, 'current_fractal_type', 'mandelbrot')
            
            formulas = {
                'mandelbrot': "z_{n+1} = z_n² + c",
                'julia': "z_{n+1} = z_n² + c (where c is constant)",
                'burning_ship': "z_{n+1} = (|Re(z_n)| + i|Im(z_n)|)² + c",
                'tricorn': "z_{n+1} = z̄_n² + c (conjugate)",
                'newton': "z_{n+1} = z_n - (z_n³ - 1)/(3z_n²)",
                'phoenix': "z_{n+1} = z_n² + c + 0.5 * z_{n-1}"
            }
            
            # Animated mathematical formula display
            st.markdown('<div class="math-formula">', unsafe_allow_html=True)
            st.latex(formulas.get(fractal_type, "z_{n+1} = f(z_n)"))
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Mathematical explanation with visual elements
            explanations = {
                'mandelbrot': "The Mandelbrot set consists of complex numbers c for which the iteration z_{n+1} = z_n² + c does not diverge when starting from z_0 = 0.",
                'julia': "Julia sets are related to the Mandelbrot set but use a fixed complex parameter c while varying the starting point z_0.",
                'burning_ship': "The Burning Ship fractal uses the absolute values of real and imaginary components before squaring.",
                'tricorn': "The Tricorn uses the complex conjugate of z_n, creating a different symmetry pattern.",
                'newton': "Newton fractals show the basins of attraction for Newton's method applied to finding roots of polynomials.",
                'phoenix': "The Phoenix fractal includes the previous iteration value, creating more complex dynamics."
            }
            
            st.write(explanations.get(fractal_type, "Mathematical fractal based on iterative complex number calculations."))
            
            # Interactive parameter visualization
            if fractal_type == "mandelbrot":
                st.markdown("**Parameter Space Visualization:**")
                # Simple parameter space indicator
                params = st.session_state.fractal_params
                real_pos = (params['center_real'] + 2) / 4 * 100  # Normalize to 0-100%
                imag_pos = (params['center_imag'] + 2) / 4 * 100
                
                position_html = f'''
                <div style="position: relative; width: 100%; height: 100px; background: linear-gradient(45deg, #1a1a2e, #16213e); border-radius: 10px; margin: 10px 0;">
                    <div style="position: absolute; left: {real_pos}%; top: {imag_pos}%; width: 10px; height: 10px; background: #4A90E2; border-radius: 50%; transform: translate(-50%, -50%); box-shadow: 0 0 15px #4A90E2; animation: pulse 2s infinite;"></div>
                    <div style="position: absolute; bottom: 5px; left: 5px; color: #4A90E2; font-size: 12px;">Real: {params['center_real']:.3f}</div>
                    <div style="position: absolute; bottom: 5px; right: 5px; color: #4A90E2; font-size: 12px;">Imag: {params['center_imag']:.3f}</div>
                </div>
                '''
                st.markdown(position_html, unsafe_allow_html=True)
        
        # Enhanced fractal display with container
        st.markdown('<div class="fractal-container">', unsafe_allow_html=True)
        st.image(st.session_state.current_fractal, use_container_width=True, 
                caption="💡 Tip: Use presets to explore interesting regions, bookmark locations you want to return to")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Animated mathematical properties display
        params = st.session_state.fractal_params
        fractal_type = getattr(st.session_state, 'current_fractal_type', 'mandelbrot')
        
        # Create animated info display
        info_html = f'''
        <div style="background: linear-gradient(135deg, rgba(74, 144, 226, 0.1), rgba(123, 104, 238, 0.1)); 
                    padding: 1rem; border-radius: 15px; margin: 1rem 0; 
                    border: 1px solid rgba(74, 144, 226, 0.3);
                    animation: pulse 3s infinite;">
            <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
                <div><strong>🔢 Type:</strong> <span class="loading-wave">{fractal_type.replace('_', ' ').title()}</span></div>
                <div><strong>🔍 Zoom:</strong> <span class="loading-wave">{params['zoom']:.1f}x</span></div>
                <div><strong>📍 Real:</strong> <span class="loading-wave">{params['center_real']:.6f}</span></div>
                <div><strong>📍 Imag:</strong> <span class="loading-wave">{params['center_imag']:.6f}</span></div>
            </div>
        </div>
        '''
        st.markdown(info_html, unsafe_allow_html=True)
    
    else:
        # First-time user guidance
        st.info("👆 Welcome to Aoin's Fractal Studio! Click 'Generate Fractal' to start exploring mathematical patterns.")
        
        # Educational content for new users
        with st.expander("🎓 What are Fractals?", expanded=True):
            st.write("""
            **Fractals** are mathematical objects that display self-similar patterns at every scale. They are generated using iterative formulas applied to complex numbers.
            
            **Key Concepts:**
            - **Complex Plane**: Uses real and imaginary number coordinates
            - **Iteration**: Repeated application of mathematical formulas
            - **Escape Time**: How quickly values diverge from the origin
            - **Self-Similarity**: Patterns repeat at different zoom levels
            
            **Popular Fractals:**
            - **Mandelbrot Set**: The most famous fractal, showing incredible complexity
            - **Julia Sets**: Related patterns with different mathematical properties
            - **Newton Fractals**: Show convergence patterns for root-finding algorithms
            """)
    
    # Quick zoom controls
    st.subheader("⚡ Quick Navigation")
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
    
    with nav_col1:
        if st.button("🔍 Zoom In 2x"):
            st.session_state.zoom_history.append(st.session_state.fractal_params.copy())
            st.session_state.fractal_params['zoom'] *= 2
            st.rerun()
    
    with nav_col2:
        if st.button("🔍 Zoom Out 2x"):
            st.session_state.zoom_history.append(st.session_state.fractal_params.copy())
            st.session_state.fractal_params['zoom'] /= 2
            st.rerun()
    
    with nav_col3:
        if st.button("🏠 Reset View"):
            st.session_state.zoom_history.append(st.session_state.fractal_params.copy())
            st.session_state.fractal_params.update({
                'zoom': 1.0, 'center_real': -0.7269, 'center_imag': 0.1889
            })
            st.rerun()
    
    with nav_col4:
        if st.button("🎯 Center Origin"):
            st.session_state.zoom_history.append(st.session_state.fractal_params.copy())
            st.session_state.fractal_params.update({
                'center_real': 0.0, 'center_imag': 0.0
            })
            st.rerun()
    
    # Generate button (prominent placement)
    if st.button("🎨 Generate Fractal", key="generate_fractal", help="Create new fractal with current settings"):
        with st.spinner("Generating fractal..."):
            fractal = generate_fractal(
                fractal_type=getattr(st.session_state, 'current_fractal_type', 'mandelbrot'),
                width=st.session_state.fractal_params['width'], 
                height=st.session_state.fractal_params['height'], 
                max_iter=st.session_state.fractal_params['iterations'],
                zoom=st.session_state.fractal_params['zoom'], 
                center_real=st.session_state.fractal_params['center_real'], 
                center_imag=st.session_state.fractal_params['center_imag'],
                julia_c=getattr(st.session_state, 'julia_c', -0.7 + 0.27015j)
            )
            
            # Add Sanskrit overlay
            fractal_with_overlay = add_sanskrit_overlay(
                fractal, 
                getattr(st.session_state, 'current_mantra_idx', 0), 
                getattr(st.session_state, 'current_colormap', 'hot')
            )
            st.session_state.current_fractal = fractal_with_overlay
            st.rerun()
    
    # Auto-generate toggle
    auto_generate = st.checkbox("⚡ Auto-generate on parameter change", 
                               value=st.session_state.auto_mode,
                               help="Automatically create new fractal when sliders change")
    st.session_state.auto_mode = auto_generate
    
    # Parameters section (collapsed by default when locked)
    with st.expander("🎛️ Fractal Parameters", expanded=not st.session_state.locked_controls):
        if not st.session_state.locked_controls:
            col1, col2 = st.columns(2)
            
            with col1:
                fractal_type = st.selectbox("Fractal Type", 
                    ["mandelbrot", "julia", "burning_ship", "tricorn", "newton", "phoenix"],
                    format_func=lambda x: x.replace('_', ' ').title(),
                    help="Choose the mathematical formula to visualize")
                zoom = st.slider("Zoom Level", 0.1, 500.0, st.session_state.fractal_params['zoom'], 0.1,
                               help="Higher values zoom deeper into the fractal")
                center_real = st.slider("Center (Real)", -2.0, 2.0, st.session_state.fractal_params['center_real'], 0.001,
                                      help="Horizontal position in the complex plane")
                iterations = st.slider("Iterations", 50, 500, st.session_state.fractal_params['iterations'], 10,
                                     help="Higher values increase detail but take longer to compute")
            
            with col2:
                center_imag = st.slider("Center (Imaginary)", -2.0, 2.0, st.session_state.fractal_params['center_imag'], 0.001,
                                      help="Vertical position in the complex plane")
                resolution = st.selectbox("Resolution", ["400x300", "600x450", "800x600", "1024x768"], index=1,
                                         help="Higher resolution creates larger, more detailed images")
                colormap = st.selectbox("Color Scheme", ["hot", "viridis", "plasma", "magma", "inferno", "cool", "spring", "winter", "autumn"],
                                      help="Color palette for the fractal visualization")
                mantra_idx = st.selectbox("Sanskrit Overlay", range(len(SANSKRIT_MANTRAS)), 
                                        format_func=lambda x: SANSKRIT_MANTRAS[x][1],
                                        help="Traditional mantra to overlay on the image")
                
                # Advanced controls
                with st.expander("🔬 Advanced Options"):
                    escape_radius = st.slider("Escape Radius", 2.0, 10.0, 2.0, 0.1,
                                            help="Mathematical threshold for escape-time algorithm")
                    smooth_coloring = st.checkbox("Smooth Coloring", value=True,
                                                 help="Use continuous coloring for smoother gradients")
                    show_coordinates = st.checkbox("Show Coordinates", value=False,
                                                  help="Display coordinate grid overlay")
                
                # Fractal-specific parameters
                if fractal_type == "julia":
                    st.markdown("**Julia Set Parameters:**")
                    julia_real = st.slider("Julia C (Real)", -2.0, 2.0, -0.7, 0.01)
                    julia_imag = st.slider("Julia C (Imag)", -2.0, 2.0, 0.27015, 0.01)
                    julia_c = julia_real + julia_imag * 1j
                    st.session_state.julia_c = julia_c
                elif fractal_type == "newton":
                    st.info("Newton fractal: Roots of z³ - 1 = 0")
                elif fractal_type == "phoenix":
                    st.info("Phoenix fractal: Uses previous iteration in calculation")
                else:
                    julia_c = -0.7 + 0.27015j
                    st.session_state.julia_c = julia_c
            
            # Parse resolution
            width, height = map(int, resolution.split('x'))
            
            # Update session state
            st.session_state.fractal_params.update({
                'zoom': zoom, 'center_real': center_real, 'center_imag': center_imag,
                'iterations': iterations, 'width': width, 'height': height
            })
            st.session_state.current_fractal_type = fractal_type
            st.session_state.current_colormap = colormap
            st.session_state.current_mantra_idx = mantra_idx
            
            # Auto-generate if enabled
            if auto_generate:
                with st.spinner("Auto-generating..."):
                    fractal = generate_fractal(
                        fractal_type=fractal_type,
                        width=width, height=height, max_iter=iterations,
                        zoom=zoom, center_real=center_real, center_imag=center_imag,
                        julia_c=getattr(st.session_state, 'julia_c', -0.7 + 0.27015j)
                    )
                    fractal_with_overlay = add_sanskrit_overlay(fractal, mantra_idx, colormap)
                    st.session_state.current_fractal = fractal_with_overlay
        else:
            st.info("🔒 Controls are locked to prevent accidental changes. Click unlock to modify parameters.")
    
    # Show current mantra and fractal info
    if st.session_state.current_fractal is not None:
        devanagari, transliteration, meaning = SANSKRIT_MANTRAS[getattr(st.session_state, 'current_mantra_idx', 0)]
        st.markdown(f"**Current Mantra:** {devanagari} ({transliteration}) - *{meaning}*")
    else:
        st.info("👆 Click 'Generate Fractal' to create your first visualization")

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
    if st.button("Generate Audio", key="generate_audio"):
        with st.spinner("Synthesizing audio..."):
            audio_data, sample_rate = generate_audio(
                base_freq=base_freq, harmony_freq=harmony_freq,
                duration=duration, sample_rate=sample_rate
            )
            st.session_state.current_audio = (audio_data, sample_rate)
    
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
            rotation_steps = st.slider("Rotation Steps", 4, 36, 12)
        elif anim_type == "parameter_sweep":
            sweep_param = st.selectbox("Parameter to Sweep", ["iterations", "julia_real", "julia_imag"])
    
    with col2:
        st.subheader("Preview Settings")
        preview_size = st.selectbox("Preview Resolution", ["200x150", "300x225", "400x300"], index=1)
        preview_colormap = st.selectbox("Animation Colormap", ["hot", "viridis", "plasma", "magma"], index=0)
        
        # Progress indicator
        if 'animation_progress' in st.session_state:
            st.progress(st.session_state.animation_progress)
    
    # Generate animation
    if st.button("🎬 Generate Animation"):
        st.session_state.animation_frames = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Parse preview size
        anim_width, anim_height = map(int, preview_size.split('x'))
        
        with st.spinner("Creating animation frames..."):
            for i in range(frames):
                # Update progress
                progress = i / frames
                progress_bar.progress(progress)
                status_text.text(f"Generating frame {i+1}/{frames}")
                
                # Calculate frame parameters based on animation type
                if anim_type == "zoom_in":
                    current_zoom = st.session_state.fractal_params['zoom'] * (zoom_factor ** (i / frames))
                elif anim_type == "zoom_out":
                    current_zoom = st.session_state.fractal_params['zoom'] / (zoom_factor ** (i / frames))
                elif anim_type == "rotate_center":
                    angle = 2 * np.pi * i / frames
                    offset = 0.1 * np.exp(1j * angle)
                    current_zoom = st.session_state.fractal_params['zoom']
                    center_offset = complex(st.session_state.fractal_params['center_real'], 
                                          st.session_state.fractal_params['center_imag']) + offset
                else:  # parameter_sweep
                    current_zoom = st.session_state.fractal_params['zoom']
                    center_offset = complex(st.session_state.fractal_params['center_real'], 
                                          st.session_state.fractal_params['center_imag'])
                
                # Generate frame
                if anim_type in ["zoom_in", "zoom_out"]:
                    fractal = generate_fractal(
                        fractal_type="mandelbrot",
                        width=anim_width, height=anim_height, max_iter=100,
                        zoom=current_zoom,
                        center_real=st.session_state.fractal_params['center_real'],
                        center_imag=st.session_state.fractal_params['center_imag']
                    )
                elif anim_type == "rotate_center":
                    fractal = generate_fractal(
                        fractal_type="mandelbrot",
                        width=anim_width, height=anim_height, max_iter=100,
                        zoom=current_zoom,
                        center_real=center_offset.real,
                        center_imag=center_offset.imag
                    )
                else:  # parameter_sweep
                    if sweep_param == "iterations":
                        iter_val = 50 + int(150 * i / frames)
                        fractal = generate_fractal(
                            fractal_type="mandelbrot",
                            width=anim_width, height=anim_height, max_iter=iter_val,
                            zoom=current_zoom,
                            center_real=st.session_state.fractal_params['center_real'],
                            center_imag=st.session_state.fractal_params['center_imag']
                        )
                
                # Convert to image
                fractal_norm = (fractal - fractal.min()) / (fractal.max() - fractal.min())
                cmap = plt.get_cmap(preview_colormap)
                colored = cmap(fractal_norm)
                frame_array = (colored[:, :, :3] * 255).astype(np.uint8)
                frame_image = Image.fromarray(frame_array)
                
                st.session_state.animation_frames.append(frame_array)
        
        progress_bar.progress(1.0)
        status_text.text(f"Animation complete! {frames} frames generated.")
        st.success(f"Generated {frames} frames for animation")
    
    # Display animation preview
    if st.session_state.animation_frames:
        st.subheader("Animation Preview")
        
        # Simple frame selector for preview
        frame_idx = st.slider("Preview Frame", 0, len(st.session_state.animation_frames)-1, 0)
        st.image(st.session_state.animation_frames[frame_idx], caption=f"Frame {frame_idx+1}/{len(st.session_state.animation_frames)}")
        
        # Animation stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Frames", len(st.session_state.animation_frames))
        with col2:
            st.metric("Duration", f"{len(st.session_state.animation_frames)/frame_rate:.1f}s")
        with col3:
            st.metric("Frame Rate", f"{frame_rate} FPS")

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
    if st.button("Generate Audio", key="generate_audio"):
        with st.spinner("Synthesizing audio..."):
            audio_data, sample_rate = generate_audio(
                base_freq=base_freq, harmony_freq=harmony_freq,
                duration=duration, sample_rate=sample_rate
            )
            st.session_state.current_audio = (audio_data, sample_rate)
    
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

with tab5:
    st.header("Current Parameters")
    
    # Performance stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Session Duration", f"{time.time() - st.session_state.get('start_time', time.time()):.0f}s")
    with col2:
        st.metric("Fractals Generated", len(st.session_state.gallery))
    with col3:
        memory_usage = len(str(st.session_state)) // 1024
        st.metric("Memory Usage", f"{memory_usage}KB")
    
    # Display current settings
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fractal Settings")
        fractal_params = st.session_state.fractal_params
        for key, value in fractal_params.items():
            if isinstance(value, float):
                st.metric(key.replace('_', ' ').title(), f"{value:.3f}")
            else:
                st.metric(key.replace('_', ' ').title(), str(value))
    
    with col2:
        st.subheader("Audio Settings")
        audio_params = st.session_state.audio_params
        for key, value in audio_params.items():
            if isinstance(value, float):
                st.metric(key.replace('_', ' ').title(), f"{value:.1f}")
            else:
                st.metric(key.replace('_', ' ').title(), str(value))
    
    # System information
    st.subheader("System Information")
    system_info = {
        "Fractal Type": getattr(st.session_state, 'current_fractal_type', 'mandelbrot').title(),
        "Color Scheme": getattr(st.session_state, 'current_colormap', 'hot'),
        "Controls": "Locked" if st.session_state.locked_controls else "Unlocked",
        "Auto Mode": "Enabled" if st.session_state.auto_mode else "Disabled",
        "Theme": st.session_state.theme.title()
    }
    
    for key, value in system_info.items():
        st.text(f"{key}: {value}")
    
    # Reset options
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Reset Parameters"):
            st.session_state.fractal_params = {
                'zoom': 1.0, 'center_real': -0.7269, 'center_imag': 0.1889,
                'iterations': 100, 'width': 600, 'height': 450
            }
            st.session_state.audio_params = {
                'base_freq': 136.1, 'harmony_freq': 432.0,
                'duration': 10, 'sample_rate': 22050
            }
            st.success("Parameters reset to defaults")
            st.rerun()
    
    with col2:
        if st.button("Reset All Data"):
            for key in list(st.session_state.keys()):
                if key not in ['start_time']:  # Keep session start time
                    del st.session_state[key]
            st.success("All data reset")
            st.rerun()

with tab6:
    st.header("Export & Download")
    
    # Batch export section
    st.subheader("📦 Batch Export")
    if st.session_state.gallery:
        export_format = st.selectbox("Export Format", ["PNG", "ZIP Archive"])
        
        if st.button("📥 Export All Saved Fractals"):
            if export_format == "ZIP Archive":
                st.info("ZIP archive generation coming soon!")
            else:
                st.success(f"Individual downloads available below")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Visual Export")
        
        if st.session_state.current_fractal is not None:
            # Convert image for download
            img = Image.fromarray(st.session_state.current_fractal)
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)
            
            # Filename with fractal info
            fractal_type = getattr(st.session_state, 'current_fractal_type', 'mandelbrot')
            timestamp = int(time.time())
            filename = f"aoin_{fractal_type}_{timestamp}.png"
            
            st.download_button(
                label="💾 Download Current Fractal",
                data=buf.getvalue(),
                file_name=filename,
                mime="image/png"
            )
            
            # High resolution option
            if st.checkbox("🔍 High Resolution Export (1920x1440)"):
                if st.button("Generate HD Version"):
                    with st.spinner("Generating high resolution fractal..."):
                        hd_fractal = generate_fractal(
                            fractal_type=getattr(st.session_state, 'current_fractal_type', 'mandelbrot'),
                            width=1920, height=1440, 
                            max_iter=st.session_state.fractal_params['iterations'],
                            zoom=st.session_state.fractal_params['zoom'],
                            center_real=st.session_state.fractal_params['center_real'],
                            center_imag=st.session_state.fractal_params['center_imag'],
                            julia_c=getattr(st.session_state, 'julia_c', -0.7 + 0.27015j)
                        )
                        
                        hd_with_overlay = add_sanskrit_overlay(
                            hd_fractal, 
                            getattr(st.session_state, 'current_mantra_idx', 0),
                            getattr(st.session_state, 'current_colormap', 'hot')
                        )
                        
                        hd_img = Image.fromarray(hd_with_overlay)
                        hd_buf = io.BytesIO()
                        hd_img.save(hd_buf, format='PNG')
                        hd_buf.seek(0)
                        
                        st.download_button(
                            label="📸 Download HD Fractal",
                            data=hd_buf.getvalue(),
                            file_name=f"aoin_HD_{fractal_type}_{timestamp}.png",
                            mime="image/png"
                        )
        else:
            st.info("Generate a fractal first")
    
    with col2:
        st.subheader("Audio Export")
        
        if st.session_state.current_audio is not None:
            audio_data, sample_rate = st.session_state.current_audio
            audio_bytes = create_audio_download(audio_data, sample_rate)
            
            st.download_button(
                label="🎵 Download Audio WAV",
                data=audio_bytes,
                file_name=f"aoin_audio_{int(time.time())}.wav",
                mime="audio/wav"
            )
            
            # Audio format options
            st.write("**Audio Info:**")
            st.write(f"- Sample Rate: {sample_rate} Hz")
            st.write(f"- Duration: {len(audio_data)/sample_rate:.1f} seconds")
            st.write(f"- File Size: ~{len(audio_bytes)//1024} KB")
        else:
            st.info("Generate audio first")
    
    # Animation export (if frames exist)
    if st.session_state.animation_frames:
        st.subheader("🎬 Animation Export")
        st.write(f"**Animation Ready:** {len(st.session_state.animation_frames)} frames")
        
        # Simple GIF creation (basic implementation)
        if st.button("📹 Create GIF"):
            st.info("💡 Advanced GIF export coming soon! For now, use individual frame downloads.")
    
    # Data export section
    st.subheader("📊 Data Export")
    
    # Parameters export
    if st.button("⚙️ Export Current Settings"):
        export_data = {
            "timestamp": time.time(),
            "fractal_params": st.session_state.fractal_params,
            "audio_params": st.session_state.audio_params,
            "fractal_type": getattr(st.session_state, 'current_fractal_type', 'mandelbrot'),
            "colormap": getattr(st.session_state, 'current_colormap', 'hot'),
            "app_version": "Aoin Studio v2.0",
            "gallery_count": len(st.session_state.gallery)
        }
        
        json_str = json.dumps(export_data, indent=2)
        st.download_button(
            label="💾 Download Settings JSON",
            data=json_str,
            file_name=f"aoin_settings_{int(time.time())}.json",
            mime="application/json"
        )
    
    # Gallery export
    if st.session_state.gallery:
        if st.button("🖼️ Export Gallery Data"):
            gallery_data = {
                "exported_at": time.time(),
                "total_fractals": len(st.session_state.gallery),
                "fractals": [
                    {
                        "timestamp": item['timestamp'],
                        "params": item['params'],
                        "type": item['type']
                    } for item in st.session_state.gallery
                ]
            }
            
            gallery_json = json.dumps(gallery_data, indent=2)
            st.download_button(
                label="📋 Download Gallery JSON",
                data=gallery_json,
                file_name=f"aoin_gallery_{int(time.time())}.json",
                mime="application/json"
            )

# Footer
st.markdown("---")
st.markdown("**💙 Aoin's Fractal Studio** - Mathematical visualization and audio synthesis")
st.markdown("Built with Streamlit • Mobile optimized interface")
