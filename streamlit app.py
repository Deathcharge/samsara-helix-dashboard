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

# Enhanced CSS with animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    
    .main > div { padding: 0.5rem !important; }
    .stSlider > div > div > div > div { background-color: #4A90E2; }
    
    .stButton > button { 
        width: 100%; height: 3rem; font-size: 1.1rem;
        background: linear-gradient(45deg, #4A90E2, #7B68EE);
        color: white; border: none; border-radius: 15px;
        margin: 0.25rem 0; box-shadow: 0 4px 15px rgba(74, 144, 226, 0.3);
        transition: all 0.3s ease; position: relative; overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px); box-shadow: 0 8px 25px rgba(74, 144, 226, 0.4);
        background: linear-gradient(45deg, #5A9FF2, #8B78FE);
    }
    
    .stButton > button::before {
        content: ''; position: absolute; top: 0; left: -100%;
        width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before { left: 100%; }
    
    .floating-particles {
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        pointer-events: none; z-index: -1; overflow: hidden;
    }
    
    .particle {
        position: absolute; width: 4px; height: 4px;
        background: radial-gradient(circle, #4A90E2, transparent);
        border-radius: 50%; animation: float 15s infinite linear; opacity: 0.6;
    }
    
    @keyframes float {
        0% { transform: translateY(100vh) rotate(0deg); opacity: 0; }
        10% { opacity: 0.6; }
        90% { opacity: 0.6; }
        100% { transform: translateY(-100px) rotate(360deg); opacity: 0; }
    }
    
    .audio-visualizer {
        width: 100%; height: 60px;
        background: linear-gradient(90deg, #1a1a2e, #16213e);
        border-radius: 10px; position: relative; overflow: hidden; margin: 1rem 0;
    }
    
    .audio-bar {
        position: absolute; bottom: 0; width: 4px;
        background: linear-gradient(0deg, #4A90E2, #7B68EE);
        border-radius: 2px; animation: audio-pulse 0.5s infinite ease-in-out alternate;
    }
    
    @keyframes audio-pulse {
        0% { height: 10px; }
        100% { height: 50px; }
    }
    
    .fractal-container {
        position: relative; border-radius: 15px; overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3); transition: all 0.3s ease;
    }
    
    .fractal-container:hover {
        transform: scale(1.02); box-shadow: 0 15px 40px rgba(74, 144, 226, 0.3);
    }
    
    @media (max-width: 768px) {
        .stSlider { margin: 0.25rem 0; }
        .particle { width: 2px; height: 2px; }
    }
</style>

<div class="floating-particles">
    <div class="particle" style="left: 10%; animation-delay: 0s;"></div>
    <div class="particle" style="left: 30%; animation-delay: 4s;"></div>
    <div class="particle" style="left: 50%; animation-delay: 8s;"></div>
    <div class="particle" style="left: 70%; animation-delay: 12s;"></div>
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
    if 'performance_stats' not in st.session_state:
        st.session_state.performance_stats = {"render_times": [], "total_fractals": 0}
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
tab1, tab2, tab3, tab4 = st.tabs(["🎨 Fractal", "🎵 Audio", "📊 Gallery", "📤 Export"])

with tab1:
    st.header("Fractal Generator")
    
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
        if st.button("📍 Bookmark", help="Save current location"):
            bookmark = {
                'name': f"Bookmark {len(st.session_state.bookmark_history) + 1}",
                'params': st.session_state.fractal_params.copy(),
                'type': st.session_state.current_fractal_type,
                'timestamp': time.time()
            }
            st.session_state.bookmark_history.append(bookmark)
            st.success("Location bookmarked!")
    
    with action_col3:
        if st.button("🔙 Previous", help="Go to previous location"):
            if st.session_state.zoom_history:
                prev_params = st.session_state.zoom_history.pop()
                st.session_state.fractal_params.update(prev_params)
                st.rerun()
    
    with action_col4:
        if st.button("🏠 Reset", help="Reset to default view"):
            st.session_state.zoom_history.append(st.session_state.fractal_params.copy())
            st.session_state.fractal_params.update({
                'zoom': 1.0, 'center_real': -0.7269, 'center_imag': 0.1889
            })
            st.rerun()
    
    with action_col5:
        if st.button("🔒 Lock" if not st.session_state.locked_controls else "🔓 Unlock"):
            st.session_state.locked_controls = not st.session_state.locked_controls
    
    # Control lock toggle
    if st.session_state.locked_controls:
        st.warning("🔒 Controls are locked to prevent accidental changes")
    else:
        st.success("🔓 Controls are unlocked")
    
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
                st.session_state.zoom_history.append(st.session_state.fractal_params.copy())
                st.session_state.fractal_params.update(params)
                st.rerun()
    
    # Community presets
    st.subheader("Community Favorites")
    community_presets = {
        "Dragon Curve": {"center_real": -0.8, "center_imag": 0.156, "zoom": 250.0, "type": "mandelbrot"},
        "Elephant Valley": {"center_real": 0.25, "center_imag": 0.0, "zoom": 150.0, "type": "mandelbrot"},  
        "Mystic Julia": {"center_real": 0.0, "center_imag": 0.0, "zoom": 1.0, "type": "julia"},
        "Burning Wings": {"center_real": -1.775, "center_imag": -0.01, "zoom": 100.0, "type": "burning_ship"},
        "Newton Roots": {"center_real": 0.0, "center_imag": 0.0, "zoom": 1.5, "type": "newton"},
        "Phoenix Rising": {"center_real": 0.5667, "center_imag": 0.0, "zoom": 1.0, "type": "phoenix"}
    }
    
    preset_cols2 = st.columns(3)
    for i, (name, params) in enumerate(community_presets.items()):
        col_idx = i % 3
        with preset_cols2[col_idx]:
            if st.button(f"🌟 {name}", key=f"community_{name}"):
                st.session_state.zoom_history.append(st.session_state.fractal_params.copy())
                st.session_state.fractal_params.update({
                    'center_real': params['center_real'],
                    'center_imag': params['center_imag'], 
                    'zoom': params['zoom']
                })
                st.session_state.current_fractal_type = params['type']
                st.success(f"Loaded {name}!")
                st.rerun()
    
    # Display current fractal first
    if st.session_state.current_fractal is not None:
        # Mathematical analysis section
        st.subheader("📊 Mathematical Analysis")
        
        analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
        
        with analysis_col1:
            # Performance metric
            if 'performance_stats' in st.session_state and st.session_state.performance_stats["render_times"]:
                avg_render = np.mean(st.session_state.performance_stats["render_times"][-5:])
                st.metric("Avg Render Time", f"{avg_render:.2f}s")
            else:
                st.metric("Avg Render Time", "N/A")
        
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
            st.metric("Fractal Type", st.session_state.current_fractal_type.replace('_', ' ').title())
        
        st.image(st.session_state.current_fractal, use_container_width=True)
        
        # Mathematical properties display
        params = st.session_state.fractal_params
        fractal_type = st.session_state.current_fractal_type
        st.info(f"**Location:** Real: {params['center_real']:.6f}, Imaginary: {params['center_imag']:.6f} | "
               f"**Zoom:** {params['zoom']:.1f}x | **Type:** {fractal_type.replace('_', ' ').title()}")
        
        # Mathematical formula display
        with st.expander("🔢 Mathematical Formula", expanded=False):
            formulas = {
                'mandelbrot': "z_{n+1} = z_n² + c",
                'julia': "z_{n+1} = z_n² + c (where c is constant)",
                'burning_ship': "z_{n+1} = (|Re(z_n)| + i|Im(z_n)|)² + c",
                'tricorn': "z_{n+1} = z̄_n² + c (conjugate)",
                'newton': "z_{n+1} = z_n - (z_n³ - 1)/(3z_n²)",
                'phoenix': "z_{n+1} = z_n² + c + 0.5 * z_{n-1}"
            }
            
            st.latex(formulas.get(fractal_type, "z_{n+1} = f(z_n)"))
            
            # Mathematical explanation
            explanations = {
                'mandelbrot': "The Mandelbrot set consists of complex numbers c for which the iteration does not diverge when starting from z_0 = 0.",
                'julia': "Julia sets use a fixed complex parameter c while varying the starting point z_0.",
                'burning_ship': "The Burning Ship fractal uses absolute values of real and imaginary components before squaring.",
                'tricorn': "The Tricorn uses the complex conjugate of z_n, creating different symmetry patterns.",
                'newton': "Newton fractals show basins of attraction for Newton's method applied to finding polynomial roots.",
                'phoenix': "The Phoenix fractal includes the previous iteration value, creating more complex dynamics."
            }
            
            st.write(explanations.get(fractal_type, "Mathematical fractal based on iterative complex number calculations."))
    
    # Generate button
    if st.button("🎨 Generate Fractal", key="generate_fractal"):
        with st.spinner("Generating fractal..."):
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
            st.rerun()
    
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
        if st.button("⬆️ Move Up"):
            st.session_state.zoom_history.append(st.session_state.fractal_params.copy())
            st.session_state.fractal_params['center_imag'] += 0.1 / st.session_state.fractal_params['zoom']
            st.rerun()
    
    with nav_col4:
        if st.button("⬇️ Move Down"):
            st.session_state.zoom_history.append(st.session_state.fractal_params.copy())
            st.session_state.fractal_params['center_imag'] -= 0.1 / st.session_state.fractal_params['zoom']
            st.rerun()
    
    # Parameters section
    if not st.session_state.locked_controls:
        with st.expander("🎛️ Fractal Parameters", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                fractal_type = st.selectbox("Fractal Type", 
                    ["mandelbrot", "julia", "burning_ship", "tricorn", "newton", "phoenix"],
                    index=["mandelbrot", "julia", "burning_ship", "tricorn", "newton", "phoenix"].index(st.session_state.current_fractal_type),
                    format_func=lambda x: x.replace('_', ' ').title())
                zoom = st.slider("Zoom Level", 0.1, 500.0, st.session_state.fractal_params['zoom'], 0.1)
                center_real = st.slider("Center (Real)", -2.0, 2.0, st.session_state.fractal_params['center_real'], 0.001)
                iterations = st.slider("Iterations", 50, 500, st.session_state.fractal_params['iterations'], 10)
            
            with col2:
                center_imag = st.slider("Center (Imaginary)", -2.0, 2.0, st.session_state.fractal_params['center_imag'], 0.001)
                resolution = st.selectbox("Resolution", ["400x300", "600x450", "800x600", "1024x768"], index=1)
                colormap = st.selectbox("Color Scheme", ["hot", "viridis", "plasma", "magma", "inferno", "cool", "spring", "winter", "autumn"], 
                                      index=["hot", "viridis", "plasma", "magma", "inferno", "cool", "spring", "winter", "autumn"].index(st.session_state.current_colormap))
                mantra_idx = st.selectbox("Sanskrit Overlay", range(len(SANSKRIT_MANTRAS)), 
                                        index=st.session_state.current_mantra_idx,
                                        format_func=lambda x: SANSKRIT_MANTRAS[x][1])
                
                # Julia set parameter (only show for Julia type)
                if fractal_type == "julia":
                    st.markdown("**Julia Set Parameters:**")
                    julia_real = st.slider("Julia C (Real)", -2.0, 2.0, -0.7, 0.01)
                    julia_imag = st.slider("Julia C (Imag)", -2.0, 2.0, 0.27015, 0.01)
                
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
    else:
        st.info("🔒 Controls are locked to prevent accidental changes. Click unlock to modify parameters.")
    
    # Show current mantra info
    if st.session_state.current_fractal is not None:
        devanagari, transliteration, meaning = SANSKRIT_MANTRAS[st.session_state.current_mantra_idx]
        st.markdown(f"**Current Mantra:** {devanagari} ({transliteration}) - *{meaning}*")

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
            st.success("Saved to gallery!")
        else:
            st.warning("Generate a fractal first")
    
    # System information
    st.subheader("System Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Session Duration", f"{time.time() - st.session_state.start_time:.0f}s")
        st.metric("Fractals Generated", len(st.session_state.gallery))
    
    with col2:
        st.metric("Controls", "Locked" if st.session_state.locked_controls else "Unlocked")
        st.metric("Memory Usage", f"{len(str(st.session_state))//1024}KB")

with tab4:
    st.header("Export & Download")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Visual Export")
        
        if st.session_state.current_fractal is not None:
            # Convert image for download
            img = Image.fromarray(st.session_state.current_fractal)
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)
            
            st.download_button(
                label="💾 Download Fractal PNG",
                data=buf.getvalue(),
                file_name=f"aoin_fractal_{int(time.time())}.png",
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
        else:
            st.info("Generate audio first")
    
    # Parameters export
    st.subheader("Settings Export")
    
    if st.button("Export Current Settings"):
        export_data = {
            "timestamp": time.time(),
            "fractal_params": st.session_state.fractal_params,
            "audio_params": st.session_state.audio_params,
            "gallery_count": len(st.session_state.gallery)
        }
        
        json_str = json.dumps(export_data, indent=2)
        st.download_button(
            label="💾 Download Settings JSON",
            data=json_str,
            file_name=f"aoin_settings_{int(time.time())}.json",
            mime="application/json"
        )

# Footer
st.markdown("---")
st.markdown("**💙 Aoin's Fractal Studio** - Mathematical visualization and audio synthesis")
st.markdown("Built with Streamlit • Mobile optimized interface")
