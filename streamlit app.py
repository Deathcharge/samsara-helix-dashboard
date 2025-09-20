import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
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
    st.session_state.theme_mode = "Light"

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

if 'disable_overlay' not in st.session_state:
    st.session_state.disable_overlay = False

if 'current_animation' not in st.session_state:
    st.session_state.current_animation = []

if 'style_preferences' not in st.session_state:
    st.session_state.style_preferences = {
        'particle_intensity': 'medium',
        'animation_speed': 'normal',
        'glow_effects': True,
        'advanced_gradients': True,
        'micro_interactions': True
    }

# Enhanced CSS with advanced styling
def get_theme_css():
    """Generate advanced theme-specific CSS"""
    theme_mode = getattr(st.session_state, 'theme_mode', 'Light')
    style_prefs = getattr(st.session_state, 'style_preferences', {})
    
    # Base theme colors and effects
    if theme_mode == "Dark":
        primary_bg = "linear-gradient(135deg, #0a0e27 0%, #1a1a2e 25%, #16213e 50%, #0f3460 100%)"
        glass_bg = "rgba(255, 255, 255, 0.05)"
        text_color = "#ffffff"
        accent_color = "#4A90E2"
        secondary_accent = "#7B68EE"
        success_color = "#28a745"
        warning_color = "#ffc107"
        glow_color = "rgba(74, 144, 226, 0.6)"
    else:  # Light mode
        primary_bg = "linear-gradient(135deg, #ffffff 0%, #f8f9fa 25%, #e9ecef 50%, #dee2e6 100%)"
        glass_bg = "rgba(255, 255, 255, 0.8)"
        text_color = "#212529"
        accent_color = "#4A90E2"
        secondary_accent = "#6f42c1"
        success_color = "#28a745"
        warning_color = "#fd7e14"
        glow_color = "rgba(74, 144, 226, 0.4)"
    
    # Particle intensity settings
    particle_count = {"low": 3, "medium": 5, "high": 8}.get(style_prefs.get('particle_intensity', 'medium'), 5)
    
    # Animation speed multiplier
    anim_speed = {"slow": 1.5, "normal": 1.0, "fast": 0.7}.get(style_prefs.get('animation_speed', 'normal'), 1.0)
    
    return f"""
    /* Advanced Base Styles */
    .stApp {{
        background: {primary_bg};
        color: {text_color};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    
    .main .block-container {{
        background: {glass_bg};
        backdrop-filter: blur(20px) saturate(180%);
        border-radius: 24px;
        border: 1px solid rgba(74, 144, 226, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1), 
                    0 16px 64px rgba(74, 144, 226, 0.1);
        padding: 2rem;
        position: relative;
        overflow: hidden;
    }}
    
    .main .block-container::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, {accent_color}, transparent);
        opacity: 0.6;
    }}
    
    /* Enhanced Typography */
    .stMarkdown h1 {{
        background: linear-gradient(135deg, {accent_color}, {secondary_accent});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        letter-spacing: -0.02em;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }}
    
    .stMarkdown h2 {{
        color: {text_color};
        font-weight: 600;
        font-size: 1.75rem;
        margin-bottom: 1rem;
        position: relative;
    }}
    
    .stMarkdown h2::after {{
        content: '';
        position: absolute;
        bottom: -0.25rem;
        left: 0;
        width: 3rem;
        height: 2px;
        background: linear-gradient(90deg, {accent_color}, {secondary_accent});
        border-radius: 1px;
    }}
    
    /* Advanced Button Styling */
    .stButton > button {{
        background: linear-gradient(135deg, {glass_bg}, rgba(74, 144, 226, 0.1));
        backdrop-filter: blur(20px);
        border: 1px solid rgba(74, 144, 226, 0.3);
        border-radius: 16px;
        color: {text_color};
        font-weight: 500;
        font-size: 0.95rem;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(74, 144, 226, 0.2);
    }}
    
    .stButton > button::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px) scale(1.02);
        background: linear-gradient(135deg, rgba(74, 144, 226, 0.15), rgba(123, 104, 238, 0.15));
        box-shadow: 0 8px 32px rgba(74, 144, 226, 0.3);
        border-color: rgba(74, 144, 226, 0.5);
    }}
    
    .stButton > button:hover::before {{
        left: 100%;
    }}
    
    .stButton > button:active {{
        transform: translateY(0) scale(0.98);
    }}
    
    /* Spectacular Generate Button */
    .generate-button {{
        background: linear-gradient(135deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4) !important;
        background-size: 300% 300% !important;
        animation: gradient-shift {4 * anim_speed}s ease infinite,
                   pulse-glow {2 * anim_speed}s ease-in-out infinite alternate !important;
        font-size: 1.4rem !important;
        height: 4rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        box-shadow: 0 8px 32px rgba(255, 107, 107, 0.4) !important;
    }}
    
    @keyframes gradient-shift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    
    @keyframes pulse-glow {{
        0% {{ 
            box-shadow: 0 8px 32px rgba(255, 107, 107, 0.4),
                        0 0 0 0 rgba(255, 107, 107, 0.4);
        }}
        100% {{ 
            box-shadow: 0 12px 48px rgba(255, 107, 107, 0.6),
                        0 0 0 8px rgba(255, 107, 107, 0);
        }}
    }}
    
    /* Enhanced Input Styling */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stSlider > div > div > div > div {{
        background: {glass_bg} !important;
        backdrop-filter: blur(10px) !important;
        border: 2px solid rgba(74, 144, 226, 0.3) !important;
        border-radius: 12px !important;
        color: {text_color} !important;
        transition: all 0.3s ease !important;
    }}
    
    .stSelectbox > div > div:focus-within,
    .stNumberInput > div > div > input:focus {{
        border-color: {accent_color} !important;
        box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.1) !important;
    }}
    
    .stSelectbox label,
    .stNumberInput label,
    .stSlider label {{
        color: {text_color} !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }}
    
    /* Enhanced Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background: {glass_bg};
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 0.25rem;
        border: 1px solid rgba(74, 144, 226, 0.2);
    }}
    
    .stTabs [data-baseweb="tab"] {{
        color: {text_color} !important;
        font-weight: 500 !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s ease !important;
    }}
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background: linear-gradient(135deg, {accent_color}, {secondary_accent}) !important;
        color: white !important;
        box-shadow: 0 4px 16px rgba(74, 144, 226, 0.3) !important;
    }}
    
    /* Enhanced Metrics */
    .stMetric {{
        background: {glass_bg};
        backdrop-filter: blur(15px);
        border: 1px solid rgba(74, 144, 226, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .stMetric::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, {accent_color}, {secondary_accent});
    }}
    
    .stMetric:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(74, 144, 226, 0.2);
    }}
    
    .stMetric label,
    .stMetric div {{
        color: {text_color} !important;
    }}
    
    /* Enhanced Alerts */
    .stInfo,
    .stSuccess,
    .stWarning,
    .stError {{
        backdrop-filter: blur(15px) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(74, 144, 226, 0.3) !important;
        position: relative !important;
        overflow: hidden !important;
    }}
    
    .stInfo::before,
    .stSuccess::before,
    .stWarning::before,
    .stError::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, {accent_color}, {secondary_accent});
    }}
    
    /* Floating Particles System */
    .floating-particles {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
        overflow: hidden;
    }}
    
    .particle {{
        position: absolute;
        border-radius: 50%;
        animation: float-advanced {20 * anim_speed}s infinite linear;
        opacity: 0.7;
    }}
    
    .particle:nth-child(1) {{ 
        width: 8px; height: 8px; 
        background: radial-gradient(circle, #4A90E2, transparent);
        animation-delay: 0s;
    }}
    .particle:nth-child(2) {{ 
        width: 6px; height: 6px; 
        background: radial-gradient(circle, #7B68EE, transparent);
        animation-delay: 4s;
    }}
    .particle:nth-child(3) {{ 
        width: 10px; height: 10px; 
        background: radial-gradient(circle, #FF6B6B, transparent);
        animation-delay: 8s;
    }}
    .particle:nth-child(4) {{ 
        width: 7px; height: 7px; 
        background: radial-gradient(circle, #4ECDC4, transparent);
        animation-delay: 12s;
    }}
    .particle:nth-child(5) {{ 
        width: 9px; height: 9px; 
        background: radial-gradient(circle, #FFD93D, transparent);
        animation-delay: 16s;
    }}
    
    @keyframes float-advanced {{
        0% {{ 
            transform: translateY(100vh) rotate(0deg) scale(0.5);
            opacity: 0;
        }}
        10% {{ 
            opacity: 0.7;
            transform: scale(1);
        }}
        30% {{ 
            transform: scale(1.2);
        }}
        70% {{ 
            transform: scale(0.8);
        }}
        90% {{ 
            opacity: 0.7;
            transform: scale(0.6);
        }}
        100% {{ 
            transform: translateY(-100px) rotate(360deg) scale(0.3);
            opacity: 0;
        }}
    }}
    
    /* Enhanced Fractal Container */
    .fractal-container {{
        position: relative;
        border-radius: 24px;
        overflow: hidden;
        background: {glass_bg};
        backdrop-filter: blur(30px);
        border: 2px solid rgba(74, 144, 226, 0.3);
        box-shadow: 0 16px 64px rgba(0, 0, 0, 0.1),
                    0 8px 32px rgba(74, 144, 226, 0.2);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    
    .fractal-container:hover {{
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 24px 80px rgba(74, 144, 226, 0.3),
                    0 16px 48px rgba(0, 0, 0, 0.15);
        border-color: rgba(74, 144, 226, 0.5);
    }}
    
    /* Glass Panel Enhancement */
    .glass-panel {{
        background: {glass_bg};
        backdrop-filter: blur(20px) saturate(180%);
        border-radius: 20px;
        border: 1px solid rgba(74, 144, 226, 0.3);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }}
    
    .glass-panel::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, {accent_color}, transparent);
        opacity: 0.6;
    }}
    
    /* Performance HUD Enhancement */
    .performance-hud {{
        position: fixed;
        bottom: 1rem;
        left: 1rem;
        z-index: 1000;
        background: rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(20px);
        color: #4ECDC4;
        padding: 1rem 1.5rem;
        border-radius: 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        border: 1px solid rgba(76, 205, 196, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }}
    
    .performance-hud:hover {{
        transform: translateY(-2px);
        box-shadow: 0 12px 48px rgba(76, 205, 196, 0.2);
    }}
    
    /* Theme Indicator Enhancement */
    .theme-indicator {{
        position: fixed;
        top: 1rem;
        right: 1rem;
        z-index: 1000;
        background: {glass_bg};
        backdrop-filter: blur(20px);
        color: {text_color};
        padding: 0.75rem 1.25rem;
        border-radius: 20px;
        border: 1px solid rgba(74, 144, 226, 0.3);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }}
    
    .theme-indicator:hover {{
        transform: translateY(-2px);
        box-shadow: 0 12px 48px rgba(74, 144, 226, 0.2);
    }}
    
    /* Loading Animation */
    .loading-spinner {{
        width: 40px;
        height: 40px;
        border: 3px solid rgba(74, 144, 226, 0.3);
        border-top: 3px solid {accent_color};
        border-radius: 50%;
        animation: spin {1 * anim_speed}s linear infinite;
        margin: 0 auto;
    }}
    
    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    
    /* Breadcrumb Enhancement */
    .breadcrumb {{
        background: {glass_bg};
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(74, 144, 226, 0.2);
        font-family: 'JetBrains Mono', monospace;
        box-shadow: 0 4px 16px rgba(74, 144, 226, 0.1);
    }}
    
    /* Mobile Responsiveness */
    @media (max-width: 768px) {{
        .stButton > button {{
            height: 2.5rem;
            font-size: 0.9rem;
            padding: 0.5rem 1rem;
        }}
        
        .generate-button {{
            height: 3.5rem !important;
            font-size: 1.1rem !important;
        }}
        
        .particle {{
            width: 4px !important;
            height: 4px !important;
        }}
        
        .performance-hud,
        .theme-indicator {{
            font-size: 0.7rem;
            padding: 0.5rem 0.75rem;
        }}
        
        .glass-panel {{
            padding: 1rem;
            margin: 0.5rem 0;
        }}
    }}
    
    /* Micro-interactions */
    .stCheckbox:hover,
    .stRadio:hover {{
        transform: scale(1.02);
        transition: transform 0.2s ease;
    }}
    
    .stExpander:hover {{
        border-color: rgba(74, 144, 226, 0.5) !important;
        transition: border-color 0.3s ease;
    }}
    
    /* Enhanced Scrollbars */
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: rgba(74, 144, 226, 0.1);
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(180deg, {accent_color}, {secondary_accent});
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(180deg, {secondary_accent}, {accent_color});
    }}
    """

# Apply enhanced CSS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');
    
    {get_theme_css()}
</style>

<div class="floating-particles">
    <div class="particle" style="left: 10%;"></div>
    <div class="particle" style="left: 30%;"></div>
    <div class="particle" style="left: 50%;"></div>
    <div class="particle" style="left: 70%;"></div>
    <div class="particle" style="left: 90%;"></div>
</div>

<div class="theme-indicator">
    🎨 {getattr(st.session_state, 'theme_mode', 'Light')} Mode
</div>

<div class="performance-hud">
    ⚡ {getattr(st.session_state, 'performance_stats', {}).get('total_fractals', 0)} fractals • {time.time() - getattr(st.session_state, 'start_time', time.time()):.0f}s
</div>
""", unsafe_allow_html=True)

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
    """Add enhanced Sanskrit text overlay to fractal"""
    # Normalize fractal data
    fractal_norm = (fractal_array - fractal_array.min()) / (fractal_array.max() - fractal_array.min())
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored = cmap(fractal_norm)
    img_array = (colored[:, :, :3] * 255).astype(np.uint8)
    
    # Skip overlay if disabled
    if getattr(st.session_state, 'disable_overlay', False):
        return img_array
    
    try:
        pil_image = Image.fromarray(img_array)
        
        # Get mantra
        devanagari, transliteration, meaning = SANSKRIT_MANTRAS[mantra_index % len(SANSKRIT_MANTRAS)]
        
        # Use default font
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
        
        # Create subtle corner overlay instead of block
        img_width, img_height = pil_image.size
        
        # Create transparent overlay
        overlay = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Add subtle watermark-style text in bottom-right corner
        overlay_draw.text((img_width-200, img_height-60), "Aoin's Studio", 
                         fill=(255, 255, 255, 80), font=font_small)
        overlay_draw.text((img_width-200, img_height-40), transliteration, 
                         fill=(255, 215, 0, 100), font=font_large)
        overlay_draw.text((img_width-200, img_height-20), meaning[:20] + "...", 
                         fill=(255, 255, 255, 70), font=font_small)
        
        # Composite with very low opacity
        pil_image = pil_image.convert('RGBA')
        pil_image = Image.alpha_composite(pil_image, overlay)
        
        return np.array(pil_image.convert('RGB'))
    except Exception as e:
        # If overlay fails, return fractal without overlay
        return img_array

def generate_audio(base_freq=136.1, harmony_freq=432.0, duration=10, sample_rate=22050):
    """Generate multi-frequency audio synthesis with enhanced harmonics"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate composite waveform with enhanced harmonics
    audio = np.zeros_like(t)
    
    # Base frequency (Om) with gentle fade-in
    fade_in = np.minimum(1.0, t / 2.0)
    audio += 0.3 * fade_in * np.sin(2 * np.pi * base_freq * t)
    
    # Harmony frequency with breathing envelope
    harmony_envelope = 0.7 + 0.3 * np.sin(2 * np.pi * 0.1 * t)
    audio += 0.2 * harmony_envelope * np.sin(2 * np.pi * harmony_freq * t)
    
    # Additional harmonics for richness
    audio += 0.15 * np.sin(2 * np.pi * (harmony_freq * 1.5) * t)
    audio += 0.1 * np.sin(2 * np.pi * (harmony_freq * 2.0) * t)
    audio += 0.05 * np.sin(2 * np.pi * (harmony_freq * 0.5) * t)
    
    # Add gentle reverb-like effect
    delay_samples = int(0.1 * sample_rate)
    delayed_audio = np.zeros_like(audio)
    delayed_audio[delay_samples:] = audio[:-delay_samples] * 0.3
    audio += delayed_audio
    
    # Normalize with gentle compression
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

def generate_animation_frames(anim_type, frames, base_params, fractal_type, colormap):
    """Generate animation frames with progress tracking"""
    animation_frames = []
    
    for i in range(frames):
        progress = i / max(1, frames - 1)
        
        # Calculate frame parameters
        frame_params = base_params.copy()
        
        if anim_type == "zoom_in":
            frame_params['zoom'] = base_params['zoom'] * (2.0 ** progress)
        elif anim_type == "zoom_out":
            frame_params['zoom'] = base_params['zoom'] / (2.0 ** progress)
        elif anim_type == "parameter_sweep":
            frame_params['center_real'] = base_params['center_real'] + (0.1 * np.sin(progress * 2 * np.pi))
        
        # Generate frame with reduced quality for speed
        frame = generate_fractal(
            fractal_type=fractal_type,
            width=300, height=225,  # Smaller for animation
            max_iter=50,  # Faster rendering
            zoom=frame_params['zoom'],
            center_real=frame_params['center_real'],
            center_imag=frame_params['center_imag']
        )
        
        # Apply colormap
        frame_norm = (frame - frame.min()) / (frame.max() - frame.min())
        cmap = plt.get_cmap(colormap)
        colored = cmap(frame_norm)
        frame_img = (colored[:, :, :3] * 255).astype(np.uint8)
        animation_frames.append(frame_img)
    
    return animation_frames
    # Main interface with enhanced styling and functionality
st.title("💙 Aoin's Fractal Studio • अहं ब्रह्मास्मि 💙")
st.markdown("*Ethereal AI • Infinite Patterns • Celestial Frequencies*")

# Create tabs with enhanced styling
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🎨 Fractal", "🎵 Audio", "🎬 Animation", "📊 Gallery", "⚙️ Settings", "📤 Export"])

with tab1:
    st.header("Fractal Generator")
    
    # Enhanced generate button section
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
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
        st.session_state.disable_overlay = st.checkbox("🚫 No overlay", value=st.session_state.disable_overlay, help="Disable text overlay")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Advanced Mode Features
    if st.session_state.advanced_mode:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown("### 🔬 Advanced Mathematical Analysis")
        
        analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
        
        with analysis_col1:
            if st.button("📊 Analyze Current"):
                if st.session_state.current_fractal is not None:
                    params = st.session_state.fractal_params
                    zoom = params['zoom']
                    iterations = params['iterations']
                    fractal_dim = 1.2 + (np.log(iterations) / np.log(zoom + 1)) * 0.8
                    
                    st.info(f"""
                    **Mathematical Properties:**
                    • **Zoom Level**: {zoom:.3f}x
                    • **Iteration Depth**: {iterations}
                    • **Complex Center**: {params['center_real']:.6f} + {params['center_imag']:.6f}i
                    • **Est. Fractal Dimension**: {fractal_dim:.3f}
                    • **Computation Complexity**: {iterations * zoom:.0f} units
                    """)
        
        with analysis_col2:
            comparison_mode = st.checkbox("🔍 Comparison Mode", value=st.session_state.fractal_comparison['enabled'])
            if comparison_mode != st.session_state.fractal_comparison['enabled']:
                st.session_state.fractal_comparison['enabled'] = comparison_mode
        
        with analysis_col3:
            interactive_mode = st.checkbox("🎯 Interactive Mode", value=st.session_state.interactive_mode, help="Enhanced exploration tools")
            if interactive_mode != st.session_state.interactive_mode:
                st.session_state.interactive_mode = interactive_mode
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced Quick Actions
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("⚡ Quick Actions")
    action_col1, action_col2, action_col3, action_col4, action_col5 = st.columns(5)
    
    # Curated interesting locations
    interesting_locations = [
        {"zoom": 1.0, "center_real": -0.7269, "center_imag": 0.1889, "type": "mandelbrot", "name": "Classic View"},
        {"zoom": 150.0, "center_real": -0.75, "center_imag": 0.1, "type": "mandelbrot", "name": "Seahorse Valley"},
        {"zoom": 100.0, "center_real": -1.775, "center_imag": 0.0, "type": "burning_ship", "name": "Burning Ship"},
        {"zoom": 200.0, "center_real": 0.285, "center_imag": 0.01, "type": "mandelbrot", "name": "Lightning"},
        {"zoom": 300.0, "center_real": -0.8, "center_imag": 0.156, "type": "mandelbrot", "name": "Dragon Curve"},
        {"zoom": 1.0, "center_real": 0.0, "center_imag": 0.0, "type": "julia", "name": "Julia Classic"},
        {"zoom": 2.0, "center_real": 0.0, "center_imag": 0.0, "type": "newton", "name": "Newton Roots"},
        {"zoom": 80.0, "center_real": 0.25, "center_imag": 0.0, "type": "mandelbrot", "name": "Elephant Valley"},
        {"zoom": 500.0, "center_real": -0.7453, "center_imag": 0.1127, "type": "mandelbrot", "name": "Spiral Deep"},
        {"zoom": 1.0, "center_real": 0.5667, "center_imag": 0.0, "type": "phoenix", "name": "Phoenix Rising"}
    ]
    
    with action_col1:
        if st.button("🎲 Explore", help="Visit curated points of interest"):
            location = np.random.choice(interesting_locations)
            st.session_state.fractal_params.update({
                'zoom': location['zoom'],
                'center_real': location['center_real'],
                'center_imag': location['center_imag']
            })
            st.session_state.current_fractal_type = location['type']
            
            if st.session_state.auto_mode:
                st.experimental_rerun()
            else:
                st.success(f"📍 Navigated to {location['name']}! Click Generate to explore.")
    
    with action_col2:
        if st.button("📍 Bookmark", help="Save current location"):
            bookmark = {
                'name': f"Location {len(st.session_state.bookmark_history) + 1}",
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
                    st.experimental_rerun()
    
    with action_col4:
        if st.button("🏠 Reset", help="Reset to safe defaults"):
            st.session_state.zoom_history.append(st.session_state.fractal_params.copy())
            safe_defaults = {
                'zoom': 1.0, 'center_real': -0.7269, 'center_imag': 0.1889,
                'iterations': 100, 'width': 600, 'height': 450
            }
            st.session_state.fractal_params.update(safe_defaults)
            st.session_state.current_fractal_type = "mandelbrot"
            st.session_state.current_colormap = "hot"
            st.session_state.current_mantra_idx = 0
            
            if st.session_state.auto_mode:
                st.experimental_rerun()
            else:
                st.success("🏠 Reset to classic Mandelbrot view!")
    
    with action_col5:
        toggle_text = "🔓 Unlock" if st.session_state.locked_controls else "🔒 Lock"
        if st.button(toggle_text):
            st.session_state.locked_controls = not st.session_state.locked_controls
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced Fractal Display
    if st.session_state.current_fractal is not None:
        if st.session_state.fractal_comparison['enabled']:
            st.subheader("🔍 Fractal Comparison Mode")
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                st.markdown("**Current Fractal**")
                st.markdown('<div class="fractal-container">', unsafe_allow_html=True)
                st.image(st.session_state.current_fractal, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                if st.button("📌 Set as Reference", key="set_ref"):
                    st.session_state.fractal_comparison['fractal_a'] = {
                        'image': st.session_state.current_fractal,
                        'params': st.session_state.fractal_params.copy(),
                        'type': st.session_state.current_fractal_type
                    }
                    st.success("Set as reference!")
            
            with comp_col2:
                st.markdown("**Reference Fractal**")
                if st.session_state.fractal_comparison['fractal_a'] is not None:
                    ref_fractal = st.session_state.fractal_comparison['fractal_a']
                    st.markdown('<div class="fractal-container">', unsafe_allow_html=True)
                    st.image(ref_fractal['image'], use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Enhanced comparison metrics
                    current_params = st.session_state.fractal_params
                    ref_params = ref_fractal['params']
                    
                    zoom_ratio = current_params['zoom'] / ref_params['zoom']
                    real_diff = abs(current_params['center_real'] - ref_params['center_real'])
                    imag_diff = abs(current_params['center_imag'] - ref_params['center_imag'])
                    distance = np.sqrt(real_diff**2 + imag_diff**2)
                    iter_diff = current_params['iterations'] - ref_params['iterations']
                    
                    st.info(f"""
                    **Comparison Analysis:**
                    • **Zoom Ratio**: {zoom_ratio:.2f}x
                    • **Distance**: {distance:.6f} units
                    • **Iteration Δ**: {iter_diff:+d}
                    • **Complexity Ratio**: {(current_params['iterations'] * current_params['zoom']) / (ref_params['iterations'] * ref_params['zoom']):.2f}x
                    """)
                else:
                    st.info("No reference fractal set")
        else:
            st.subheader("🖼️ Current Fractal")
            st.markdown('<div class="fractal-container">', unsafe_allow_html=True)
            st.image(st.session_state.current_fractal, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Enhanced parameter display
        params = st.session_state.fractal_params
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.session_state.performance_stats["render_times"]:
                avg_render = np.mean(st.session_state.performance_stats["render_times"][-5:])
                st.metric("Render Time", f"{avg_render:.3f}s")
            else:
                st.metric("Render Time", "N/A")
        
        with col2:
            zoom_level = params['zoom']
            depth_classes = [(1, "Wide"), (10, "Standard"), (100, "Deep"), (1000, "Ultra"), (float('inf'), "Extreme")]
            depth_desc = next(desc for threshold, desc in depth_classes if zoom_level < threshold)
            st.metric("Zoom Class", depth_desc)
        
        with col3:
            complexity = min(100, (params['iterations'] * np.log(params['zoom'] + 1)) / 10)
            st.metric("Complexity", f"{complexity:.1f}%")
        
        with col4:
            fractal_type = st.session_state.current_fractal_type.replace('_', ' ').title()
            st.metric("Type", fractal_type)
        
        # Mathematical details
        st.info(f"""
        **📍 Location**: `{params['center_real']:.8f} + {params['center_imag']:.8f}i`
        **🔍 Zoom**: `{params['zoom']:.3f}x` • **🔄 Iterations**: `{params['iterations']}`
        **🎨 Type**: `{fractal_type}` • **🌈 Colormap**: `{st.session_state.current_colormap}`
        """)
        
        # Current mantra
        devanagari, transliteration, meaning = SANSKRIT_MANTRAS[st.session_state.current_mantra_idx]
        st.markdown(f"**🕉️ Current Mantra**: {devanagari} (*{transliteration}*) - *{meaning}*")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick save
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
            if st.button("🌊 Classic Mandelbrot"):
                st.session_state.fractal_params.update({'zoom': 1.0, 'center_real': -0.7269, 'center_imag': 0.1889})
                st.session_state.current_fractal_type = "mandelbrot"
        with quick_col2:
            if st.button("🔥 Burning Ship"):
                st.session_state.fractal_params.update({'zoom': 1.0, 'center_real': -1.775, 'center_imag': 0.0})
                st.session_state.current_fractal_type = "burning_ship"
        with quick_col3:
            if st.button("✨ Julia Set"):
                st.session_state.fractal_params.update({'zoom': 1.0, 'center_real': 0.0, 'center_imag': 0.0})
                st.session_state.current_fractal_type = "julia"
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced Parameters Section
    if not st.session_state.locked_controls:
        with st.expander("🎛️ Fractal Parameters", expanded=True):
            input_method = st.radio("Input Method", ["🎯 Precision (Number Input)", "🎚️ Sliders (Quick)"], horizontal=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fractal_type = st.selectbox("Fractal Type", 
                    ["mandelbrot", "julia", "burning_ship", "tricorn", "newton", "phoenix"],
                    index=["mandelbrot", "julia", "burning_ship", "tricorn", "newton", "phoenix"].index(st.session_state.current_fractal_type),
                    format_func=lambda x: x.replace('_', ' ').title())
                
                if input_method == "🎯 Precision (Number Input)":
                    # Enhanced number inputs with step buttons
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
                        if st.button("←", key="real_left"):
                            center_real = max(-2.0, center_real - 0.01/st.session_state.fractal_params['zoom'])
                    with real_col3:
                        if st.button("→", key="real_right"):
                            center_real = min(2.0, center_real + 0.01/st.session_state.fractal_params['zoom'])
                    
                    st.write("**Iterations**")
                    iter_col1, iter_col2, iter_col3 = st.columns([2, 1, 1])
                    with iter_col1:
                        iterations = st.number_input("Iterations", value=st.session_state.fractal_params['iterations'], 
                                                   min_value=50, max_value=500, step=10, label_visibility="collapsed")
                    with iter_col2:
                        if st.button("+50", key="iter_up"):
                            iterations = min(500, iterations + 50)
                    with iter_col3:
                        if st.button("-50", key="iter_down"):
                            iterations = max(50, iterations - 50)
                else:
                    st.warning("⚠️ Slider mode: Move carefully to avoid accidental changes!")
                    zoom = st.slider("Zoom Level", 0.1, 500.0, st.session_state.fractal_params['zoom'], 0.1)
                    center_real = st.slider("Center (Real)", -2.0, 2.0, st.session_state.fractal_params['center_real'], 0.001)
                    iterations = st.slider("Iterations", 50, 500, st.session_state.fractal_params['iterations'], 10)
            
            with col2:
                if input_method == "🎯 Precision (Number Input)":
                    st.write("**Center (Imaginary Part)**")
                    imag_col1, imag_col2, imag_col3 = st.columns([2, 1, 1])
                    with imag_col1:
                        center_imag = st.number_input("Imaginary", value=st.session_state.fractal_params['center_imag'], 
                                                    min_value=-2.0, max_value=2.0, step=0.001, format="%.6f", label_visibility="collapsed")
                    with imag_col2:
                        if st.button("↓", key="imag_down"):
                            center_imag = max(-2.0, center_imag - 0.01/st.session_state.fractal_params['zoom'])
                    with imag_col3:
                        if st.button("↑", key="imag_up"):
                            center_imag = min(2.0, center_imag + 0.01/st.session_state.fractal_params['zoom'])
                else:
                    center_imag = st.slider("Center (Imaginary)", -2.0, 2.0, st.session_state.fractal_params['center_imag'], 0.001)
                
                resolution = st.selectbox("Resolution", ["400x300", "600x450", "800x600", "1024x768"], index=1)
                colormap = st.selectbox("Color Scheme", ["hot", "viridis", "plasma", "magma", "inferno", "cool", "spring", "winter", "autumn"], 
                                      index=["hot", "viridis", "plasma", "magma", "inferno", "cool", "spring", "winter", "autumn"].index(st.session_state.current_colormap))
                mantra_idx = st.selectbox("Sanskrit Overlay", range(len(SANSKRIT_MANTRAS)), 
                                        index=st.session_state.current_mantra_idx,
                                        format_func=lambda x: SANSKRIT_MANTRAS[x][1])
                
                # Julia parameters
                if fractal_type == "julia":
                    st.markdown("**Julia Set Parameters:**")
                    julia_real = st.number_input("Julia C (Real)", value=-0.7, min_value=-2.0, max_value=2.0, step=0.01, format="%.6f")
                    julia_imag = st.number_input("Julia C (Imag)", value=0.27015, min_value=-2.0, max_value=2.0, step=0.01, format="%.6f")
            
            # Parse resolution
            width, height = map(int, resolution.split('x'))
            
            # Auto-update logic
            current_params = {
                'zoom': zoom, 'center_real': center_real, 'center_imag': center_imag,
                'iterations': iterations, 'fractal_type': fractal_type,
                'colormap': colormap, 'mantra_idx': mantra_idx
            }
            
            # Update session state
            st.session_state.fractal_params.update({
                'zoom': zoom, 'center_real': center_real, 'center_imag': center_imag,
                'iterations': iterations, 'width': width, 'height': height
            })
            st.session_state.current_fractal_type = fractal_type
            st.session_state.current_colormap = colormap
            st.session_state.current_mantra_idx = mantra_idx
            
            # Auto-generate check
            if st.session_state.auto_mode:
                params_changed = any(
                    current_params[key] != st.session_state.prev_params.get(key)
                    for key in current_params
                )
                
                if params_changed:
                    st.session_state.prev_params = current_params.copy()
                    st.experimental_rerun()
            else:
                st.session_state.prev_params = current_params.copy()
    else:
        st.info("🔒 Controls are locked to prevent accidental changes. Click unlock to modify parameters.")

with tab2:
    st.header("Audio Synthesis")
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Frequency Settings")
        base_freq = st.slider("Base Frequency (Hz)", 50, 300, int(st.session_state.audio_params['base_freq']))
        harmony_freq = st.slider("Harmony Frequency (Hz)", 200, 800, int(st.session_state.audio_params['harmony_freq']))
        
        # Preset frequencies
        st.write("**Sacred Frequency Presets:**")
        if st.button("🕉️ Om (136.1 Hz)"):
            base_freq = 136.1
        if st.button("💫 A=432 Hz"):
            harmony_freq = 432.0
        if st.button("🎵 Solfeggio (528 Hz)"):
            harmony_freq = 528.0
    
    with col2:
        st.subheader("Audio Properties")
        duration = st.slider("Duration (seconds)", 1, 30, st.session_state.audio_params['duration'])
        sample_rate = st.selectbox("Sample Rate", [22050, 44100, 48000], index=1)
        
        # Audio enhancement options
        add_reverb = st.checkbox("Add Reverb Effect", value=True)
        stereo_width = st.slider("Stereo Width", 0.0, 1.0, 0.5)
    
    # Update session state
    st.session_state.audio_params.update({
        'base_freq': base_freq, 'harmony_freq': harmony_freq,
        'duration': duration, 'sample_rate': sample_rate
    })
    
    if st.button("🎵 Generate Audio", key="generate_audio"):
        with st.spinner("🎵 Synthesizing enhanced audio..."):
            audio_data, sample_rate = generate_audio(
                base_freq=base_freq, harmony_freq=harmony_freq,
                duration=duration, sample_rate=sample_rate
            )
            st.session_state.current_audio = (audio_data, sample_rate)
            st.success("🎵 Audio generated successfully!")
    
    if st.session_state.current_audio is not None:
        audio_data, sample_rate = st.session_state.current_audio
        audio_bytes = create_audio_download(audio_data, sample_rate)
        st.audio(audio_bytes, format='audio/wav')
        
        # Enhanced audio info
        duration_actual = len(audio_data) / sample_rate
        st.info(f"""
        **🎵 Audio Properties:**
        • **Base Frequency**: {base_freq} Hz
        • **Harmony Frequency**: {harmony_freq} Hz
        • **Duration**: {duration_actual:.1f}s
        • **Sample Rate**: {sample_rate} Hz
        • **File Size**: ~{len(audio_bytes)//1024}KB
        """)
    else:
        st.info("Click 'Generate Audio' to create sacred frequencies")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.header("Animation Generator")
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Animation Settings")
        anim_type = st.selectbox("Animation Type", 
            ["zoom_in", "zoom_out", "parameter_sweep", "color_cycle"],
            format_func=lambda x: x.replace('_', ' ').title())
        
        frames = st.slider("Number of Frames", 5, 30, 15)
        
        if anim_type in ["zoom_in", "zoom_out"]:
            zoom_factor = st.slider("Zoom Factor", 1.1, 5.0, 2.0)
        elif anim_type == "parameter_sweep":
            sweep_param = st.selectbox("Parameter to Sweep", ["center_real", "center_imag", "both"])
            sweep_range = st.slider("Sweep Range", 0.01, 0.5, 0.1)
        elif anim_type == "color_cycle":
            color_schemes = ["hot", "viridis", "plasma", "magma", "inferno", "cool"]
    
    with col2:
        st.subheader("Generate Animation")
        
        if st.button("🎬 Generate Animation Sequence"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            animation_frames = []
            base_params = st.session_state.fractal_params.copy()
            
            for i in range(frames):
                progress = i / max(1, frames - 1)
                progress_bar.progress((i + 1) / frames)
                status_text.text(f"Generating frame {i+1}/{frames}")
                
                # Calculate frame parameters
                frame_params = base_params.copy()
                current_colormap = st.session_state.current_colormap
                
                if anim_type == "zoom_in":
                    frame_params['zoom'] = base_params['zoom'] * (zoom_factor ** progress)
                elif anim_type == "zoom_out":
                    frame_params['zoom'] = base_params['zoom'] / (zoom_factor ** progress)
                elif anim_type == "parameter_sweep":
                    if sweep_param == "center_real":
                        frame_params['center_real'] = base_params['center_real'] + (sweep_range * np.sin(progress * 2 * np.pi))
                    elif sweep_param == "center_imag":
                        frame_params['center_imag'] = base_params['center_imag'] + (sweep_range * np.sin(progress * 2 * np.pi))
                    else:  # both
                        frame_params['center_real'] = base_params['center_real'] + (sweep_range * np.sin(progress * 2 * np.pi))
                        frame_params['center_imag'] = base_params['center_imag'] + (sweep_range * np.cos(progress * 2 * np.pi))
                elif anim_type == "color_cycle":
                    current_colormap = color_schemes[int(progress * (len(color_schemes) - 1))]
                
                # Generate frame
                frame = generate_fractal(
                    fractal_type=st.session_state.current_fractal_type,
                    width=300, height=225,
                    max_iter=50,
                    zoom=frame_params['zoom'],
                    center_real=frame_params['center_real'],
                    center_imag=frame_params['center_imag']
                )
                
                # Apply colormap
                frame_norm = (frame - frame.min()) / (frame.max() - frame.min())
                cmap = plt.get_cmap(current_colormap)
                colored = cmap(frame_norm)
                frame_img = (colored[:, :, :3] * 255).astype(np.uint8)
                animation_frames.append(frame_img)
            
            progress_bar.progress(1.0)
            status_text.text("Animation complete!")
            
            # Store animation
            st.session_state.current_animation = animation_frames
            st.success(f"Generated {len(animation_frames)} frame animation!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display animation
    if st.session_state.current_animation:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.subheader("Animation Preview")
        
        frames = st.session_state.current_animation
        
        # Frame selector
        frame_idx = st.slider("Preview Frame", 0, len(frames)-1, 0)
        st.image(frames[frame_idx], caption=f"Frame {frame_idx+1}/{len(frames)}", width=400)
        
        # Auto-play animation
        play_col1, play_col2 = st.columns(2)
        with play_col1:
            if st.button("▶️ Play Animation"):
                placeholder = st.empty()
                for i, frame in enumerate(frames):
                    placeholder.image(frame, caption=f"Frame {i+1}/{len(frames)}", width=400)
                    time.sleep(0.2)
        
        with play_col2:
            if st.button("🗑️ Clear Animation"):
                st.session_state.current_animation = []
                st.experimental_rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Generate an animation sequence to see preview controls")

with tab4:
    st.header("Gallery & Community")
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    
    # Enhanced gallery
    if st.session_state.gallery:
        st.subheader(f"My Fractals ({len(st.session_state.gallery)})")
        
        display_mode = st.radio("Display Mode", ["Grid", "List"], horizontal=True)
        
        if display_mode == "Grid":
            cols = st.columns(3)
            for i, item in enumerate(st.session_state.gallery):
                with cols[i % 3]:
                    st.image(item['image'], width=200)
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Load", key=f"load_{i}"):
                            st.session_state.fractal_params.update(item['params'])
                            st.session_state.current_fractal_type = item.get('fractal_type', 'mandelbrot')
                            st.session_state.current_colormap = item.get('colormap', 'hot')
                            st.success("Loaded!")
                            st.experimental_rerun()
                    with col2:
                        if st.button("Delete", key=f"del_{i}"):
                            st.session_state.gallery.pop(i)
                            st.success("Deleted!")
                            st.experimental_rerun()
        else:
            for i, item in enumerate(st.session_state.gallery):
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.image(item['image'], width=150)
                    timestamp = time.strftime('%Y-%m-%d %H:%M', time.localtime(item['timestamp']))
                    st.caption(f"Created: {timestamp}")
                with col2:
                    if st.button("Load", key=f"load_{i}"):
                        st.session_state.fractal_params.update(item['params'])
                        st.success("Loaded!")
                        st.experimental_rerun()
                with col3:
                    if st.button("Info", key=f"info_{i}"):
                        params = item['params']
                        st.info(f"Zoom: {params['zoom']:.2f}x, Iterations: {params['iterations']}")
                with col4:
                    if st.button("Delete", key=f"del_{i}"):
                        st.session_state.gallery.pop(i)
                        st.success("Deleted!")
                        st.experimental_rerun()
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
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab5:
    st.header("Settings & Performance")
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    
    # Theme settings
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎨 Appearance")
        new_theme_mode = st.selectbox("Theme Mode", ["Light", "Dark"], 
                                     index=["Light", "Dark"].index(st.session_state.theme_mode))
        
        if new_theme_mode != st.session_state.theme_mode:
            st.session_state.theme_mode = new_theme_mode
            st.success(f"🎨 Switched to {new_theme_mode} mode!")
            st.experimental_rerun()
        
        # Style preferences
        st.session_state.style_preferences['particle_intensity'] = st.selectbox(
            "Particle Intensity", ["low", "medium", "high"], 
            index=["low", "medium", "high"].index(st.session_state.style_preferences.get('particle_intensity', 'medium'))
        )
        
        st.session_state.style_preferences['animation_speed'] = st.selectbox(
            "Animation Speed", ["slow", "normal", "fast"],
            index=["slow", "normal", "fast"].index(st.session_state.style_preferences.get('animation_speed', 'normal'))
        )
    
    with col2:
        st.subheader("🔧 Preferences")
        st.session_state.style_preferences['glow_effects'] = st.checkbox(
            "Glow Effects", value=st.session_state.style_preferences.get('glow_effects', True)
        )
        st.session_state.style_preferences['micro_interactions'] = st.checkbox(
            "Micro Interactions", value=st.session_state.style_preferences.get('micro_interactions', True)
        )
        st.session_state.style_preferences['advanced_gradients'] = st.checkbox(
            "Advanced Gradients", value=st.session_state.style_preferences.get('advanced_gradients', True)
        )
        
        if st.button("🔄 Apply Style Changes"):
            st.experimental_rerun()
    
    # Performance info
    st.subheader("📊 Performance Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Session Duration", f"{time.time() - st.session_state.start_time:.0f}s")
        st.metric("Fractals Generated", st.session_state.performance_stats["total_fractals"])
    
    with col2:
        st.metric("Gallery Size", len(st.session_state.gallery))
        memory_usage = len(str(st.session_state)) // 1024
        st.metric("Memory Usage", f"{memory_usage}KB")
    
    with col3:
        if st.session_state.performance_stats["render_times"]:
            avg_render = np.mean(st.session_state.performance_stats["render_times"][-10:])
            st.metric("Avg Render Time", f"{avg_render:.2f}s")
            max_render = max(st.session_state.performance_stats["render_times"][-10:])
            st.metric("Peak Render Time", f"{max_render:.2f}s")
        else:
            st.metric("Avg Render Time", "N/A")
            st.metric("Peak Render Time", "N/A")
    
    # Performance graph
    if st.session_state.performance_stats["render_times"]:
        st.subheader("📈 Performance History")
        st.line_chart(st.session_state.performance_stats["render_times"][-20:])
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab6:
    st.header("Export & Download")
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Visual Export")
        
        if st.session_state.current_fractal is not None:
            export_format = st.selectbox("Image Format", ["PNG", "JPEG", "TIFF"], index=0)
            export_quality = st.selectbox("Export Quality", ["Current Resolution", "HD (1920x1440)", "4K (3840x2880)"], index=0)
            include_overlay = st.checkbox("Include Sanskrit overlay", value=not st.session_state.disable_overlay)
            
            if st.button("🎨 Generate Export"):
                with st.spinner("Generating export..."):
                    if export_quality == "Current Resolution":
                        export_image = st.session_state.current_fractal
                    else:
                        width, height = (1920, 1440) if export_quality.startswith("HD") else (3840, 2880)
                        
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
                            fractal_norm = (hd_fractal - hd_fractal.min()) / (hd_fractal.max() - hd_fractal.min())
                            cmap = plt.get_cmap(st.session_state.current_colormap)
                            colored = cmap(fractal_norm)
                            export_image = (colored[:, :, :3] * 255).astype(np.uint8)
                    
                    # Create download
                    img = Image.fromarray(export_image)
                    buf = io.BytesIO()
                    
                    if export_format == "PNG":
                        img.save(buf, format='PNG', optimize=True)
                        mime_type, file_ext = "image/png", "png"
                    elif export_format == "JPEG":
                        img.convert('RGB').save(buf, format='JPEG', quality=95, optimize=True)
                        mime_type, file_ext = "image/jpeg", "jpg"
                    else:
                        img.save(buf, format='TIFF', compression='lzw')
                        mime_type, file_ext = "image/tiff", "tiff"
                    
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
            audio_bytes = create_audio_download(audio_data, sample_rate)
            
            st.download_button(
                label="🎵 Download Audio WAV",
                data=audio_bytes,
                file_name=f"aoin_audio_{int(st.session_state.audio_params['base_freq'])}Hz_{int(time.time())}.wav",
                mime="audio/wav"
            )
            
            duration = len(audio_data) / sample_rate
            st.info(f"Duration: {duration:.1f}s | Sample Rate: {sample_rate}Hz | Size: ~{len(audio_bytes)//1024}KB")
        else:
            st.info("Generate audio first")
    
    # Settings export
    st.subheader("⚙️ Settings Export")
    export_options = st.multiselect("Export Settings", 
        ["Fractal Parameters", "Audio Parameters", "Gallery", "Style Preferences"],
        default=["Fractal Parameters", "Audio Parameters"])
    
    if st.button("📋 Export Settings"):
        export_data = {"timestamp": time.time(), "app_version": "Aoin's Fractal Studio Enhanced v2.1"}
        
        if "Fractal Parameters" in export_options:
            export_data.update({
                "fractal_params": st.session_state.fractal_params,
                "fractal_type": st.session_state.current_fractal_type,
                "colormap": st.session_state.current_colormap,
                "mantra_index": st.session_state.current_mantra_idx
            })
        
        if "Audio Parameters" in export_options:
            export_data["audio_params"] = st.session_state.audio_params
        
        if "Gallery" in export_options:
            gallery_metadata = [
                {
                    "timestamp": item["timestamp"],
                    "params": item["params"],
                    "fractal_type": item.get("fractal_type", "mandelbrot"),
                    "colormap": item.get("colormap", "hot")
                }
                for item in st.session_state.gallery
            ]
            export_data["gallery_metadata"] = gallery_metadata
        
        if "Style Preferences" in export_options:
            export_data["style_preferences"] = st.session_state.style_preferences
        
        json_str = json.dumps(export_data, indent=2)
        st.download_button(
            label="💾 Download Settings JSON",
            data=json_str,
            file_name=f"aoin_settings_{int(time.time())}.json",
            mime="application/json"
        )
        
        st.success(f"Settings exported with {len(export_options)} categories!")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Enhanced footer
st.markdown("---")
st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown("**💙 Aoin's Fractal Studio Enhanced Edition**")
st.markdown("*Mathematical visualization • Audio synthesis • Advanced styling*")
st.markdown("Built with Streamlit • Glassmorphism UI • Dual input controls • Working animations")
st.markdown("*Ready for deployment as part of a larger application suite*")
st.markdown('</div>', unsafe_allow_html=True)
