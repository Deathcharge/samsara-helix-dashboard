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

# Custom CSS for Aoin branding and mobile optimization
st.markdown("""
<style>
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
    }
    .metric-container { 
        background: linear-gradient(135deg, rgba(74, 144, 226, 0.1), rgba(123, 104, 238, 0.1)); 
        padding: 1rem; 
        border-radius: 15px; 
        margin: 0.5rem 0;
        border: 1px solid rgba(74, 144, 226, 0.2);
    }
    .block-container { 
        padding-top: 1rem; 
        max-width: 100%; 
    }
    
    .aoin-header {
        background: linear-gradient(90deg, #4A90E2, #7B68EE);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(74, 144, 226, 0.3);
    }
    
    .aoin-subtitle {
        color: #4A90E2;
        font-style: italic;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    @media (max-width: 768px) {
        .stSlider { margin: 0.25rem 0; }
        .metric-container { padding: 0.5rem; }
        .aoin-header { padding: 0.75rem; }
    }
</style>
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

# Sanskrit mantras for overlay
SANSKRIT_MANTRAS = [
    ("अहं ब्रह्मास्मि", "Aham Brahmasmi", "I am Brahman"),
    ("तत्त्वमसि", "Tat Tvam Asi", "Thou art That"),
    ("नेति नेति", "Neti Neti", "Not this, Not this"),
    ("सर्वं खल्विदं ब्रह्म", "Sarvam khalvidam brahma", "All this is Brahman")
]

def generate_fractal(fractal_type="mandelbrot", width=600, height=450, max_iter=100, zoom=1.0, center_real=-0.7269, center_imag=0.1889, julia_c=(-0.7+0.27015j)):
    """Generate different types of fractals"""
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
    
    # Set non-escaped points
    escape_time[escape_time == 0] = max_iter
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
    """Generate multi-frequency audio synthesis"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate composite waveform
    audio = np.zeros_like(t)
    
    # Base frequency (Om)
    audio += 0.3 * np.sin(2 * np.pi * base_freq * t)
    
    # Harmony frequency
    harmony_envelope = 0.7 + 0.3 * np.sin(2 * np.pi * 0.1 * t)  # Slow modulation
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

# Main interface with Aoin branding and Sanskrit
st.markdown('<div class="aoin-header"><h1>💙 Aoin\'s Fractal Studio • अहं ब्रह्मास्मि 💙</h1><p>Ethereal AI • Infinite Patterns • Celestial Frequencies</p><p style="font-size:0.9em; opacity:0.8;">तत्त्वमसि • नेति नेति • सर्वं खल्विदं ब्रह्म</p></div>', unsafe_allow_html=True)
st.markdown('<p class="aoin-subtitle">✨ Where mathematics meets digital consciousness ✨</p>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎨 Fractal", "🎵 Audio", "📊 Parameters", "📤 Export"])

with tab1:
    st.header("Fractal Generator")
    
    # Parameter controls
    col1, col2 = st.columns(2)
    
    with col1:
        fractal_type = st.selectbox("Fractal Type", 
            ["mandelbrot", "julia", "burning_ship", "tricorn"],
            format_func=lambda x: x.replace('_', ' ').title())
        zoom = st.slider("Zoom Level", 0.1, 50.0, st.session_state.fractal_params['zoom'], 0.1)
        center_real = st.slider("Center (Real)", -2.0, 2.0, st.session_state.fractal_params['center_real'], 0.001)
        iterations = st.slider("Iterations", 50, 300, st.session_state.fractal_params['iterations'], 10)
    
    with col2:
        center_imag = st.slider("Center (Imaginary)", -2.0, 2.0, st.session_state.fractal_params['center_imag'], 0.001)
        resolution = st.selectbox("Resolution", ["400x300", "600x450", "800x600"], index=1)
        colormap = st.selectbox("Color Scheme", ["hot", "viridis", "plasma", "magma", "inferno"])
        mantra_idx = st.selectbox("Sanskrit Overlay", range(len(SANSKRIT_MANTRAS)), 
                                format_func=lambda x: SANSKRIT_MANTRAS[x][1])
        
        # Julia set parameter (only show for Julia type)
        if fractal_type == "julia":
            julia_real = st.slider("Julia C (Real)", -2.0, 2.0, -0.7, 0.01)
            julia_imag = st.slider("Julia C (Imag)", -2.0, 2.0, 0.27015, 0.01)
            julia_c = julia_real + julia_imag * 1j
        else:
            julia_c = -0.7 + 0.27015j
    
    # Parse resolution
    width, height = map(int, resolution.split('x'))
    
    # Update session state
    st.session_state.fractal_params.update({
        'zoom': zoom, 'center_real': center_real, 'center_imag': center_imag,
        'iterations': iterations, 'width': width, 'height': height
    })
    
    # Generate fractal
    if st.button("Generate Fractal", key="generate_fractal") or st.checkbox("Auto-generate"):
        with st.spinner("Generating fractal..."):
            fractal = generate_fractal(
                fractal_type=fractal_type,
                width=width, height=height, max_iter=iterations,
                zoom=zoom, center_real=center_real, center_imag=center_imag,
                julia_c=julia_c
            )
            
            # Add Sanskrit overlay
            fractal_with_overlay = add_sanskrit_overlay(fractal, mantra_idx, colormap)
            st.session_state.current_fractal = fractal_with_overlay
    
    # Display fractal
    if st.session_state.current_fractal is not None:
        st.image(st.session_state.current_fractal, use_column_width=True)
        
        # Show current mantra and fractal info
        devanagari, transliteration, meaning = SANSKRIT_MANTRAS[mantra_idx]
        st.markdown(f"**Current Mantra:** {devanagari} ({transliteration}) - *{meaning}*")
        st.markdown(f"**Fractal Type:** {fractal_type.replace('_', ' ').title()}")
    else:
        st.info("Click 'Generate Fractal' to create visualization")

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
    st.header("Current Parameters")
    
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
    
    # Reset button
    if st.button("Reset to Defaults"):
        st.session_state.fractal_params = {
            'zoom': 1.0, 'center_real': -0.7269, 'center_imag': 0.1889,
            'iterations': 100, 'width': 600, 'height': 450
        }
        st.session_state.audio_params = {
            'base_freq': 136.1, 'harmony_freq': 432.0,
            'duration': 10, 'sample_rate': 22050
        }
        st.success("Parameters reset to defaults")
        st.experimental_rerun()

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
                label="Download Fractal PNG",
                data=buf.getvalue(),
                file_name=f"mandelbrot_fractal_{int(time.time())}.png",
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
                label="Download Audio WAV",
                data=audio_bytes,
                file_name=f"synthesized_audio_{int(time.time())}.wav",
                mime="audio/wav"
            )
        else:
            st.info("Generate audio first")
    
    # Parameters export
    st.subheader("Parameters Export")
    
    if st.button("Export Current Parameters"):
        export_data = {
            "timestamp": time.time(),
            "fractal_params": st.session_state.fractal_params,
            "audio_params": st.session_state.audio_params
        }
        
        json_str = json.dumps(export_data, indent=2)
        st.download_button(
            label="Download Parameters JSON",
            data=json_str,
            file_name=f"parameters_{int(time.time())}.json",
            mime="application/json"
        )

# Footer
st.markdown("---")
st.markdown("**💙 Aoin's Fractal Studio** - Mathematical visualization and audio synthesis")
st.markdown("Built with Streamlit • Mobile optimized interface")
