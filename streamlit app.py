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

# Sanskrit mantras
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
    
    # Control lock toggle
    if st.button("🔒 Lock Controls" if not st.session_state.locked_controls else "🔓 Unlock Controls"):
        st.session_state.locked_controls = not st.session_state.locked_controls
    
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
        st.image(st.session_state.current_fractal, use_container_width=True)
        
        # Show current info
        params = st.session_state.fractal_params
        st.info(f"**Zoom:** {params['zoom']:.1f}x | **Center:** ({params['center_real']:.3f}, {params['center_imag']:.3f})")
    
    # Generate button
    if st.button("🎨 Generate Fractal", key="generate_fractal"):
        with st.spinner("Generating fractal..."):
            fractal = generate_fractal(
                fractal_type="mandelbrot",
                width=st.session_state.fractal_params['width'], 
                height=st.session_state.fractal_params['height'], 
                max_iter=st.session_state.fractal_params['iterations'],
                zoom=st.session_state.fractal_params['zoom'], 
                center_real=st.session_state.fractal_params['center_real'], 
                center_imag=st.session_state.fractal_params['center_imag']
            )
            
            # Add Sanskrit overlay
            fractal_with_overlay = add_sanskrit_overlay(fractal, 0, 'hot')
            st.session_state.current_fractal = fractal_with_overlay
            st.rerun()
    
    # Parameters section
    if not st.session_state.locked_controls:
        with st.expander("🎛️ Fractal Parameters", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                zoom = st.slider("Zoom Level", 0.1, 500.0, st.session_state.fractal_params['zoom'], 0.1)
                center_real = st.slider("Center (Real)", -2.0, 2.0, st.session_state.fractal_params['center_real'], 0.001)
                iterations = st.slider("Iterations", 50, 500, st.session_state.fractal_params['iterations'], 10)
            
            with col2:
                center_imag = st.slider("Center (Imaginary)", -2.0, 2.0, st.session_state.fractal_params['center_imag'], 0.001)
                resolution = st.selectbox("Resolution", ["400x300", "600x450", "800x600"], index=1)
                
            # Parse resolution
            width, height = map(int, resolution.split('x'))
            
            # Update session state
            st.session_state.fractal_params.update({
                'zoom': zoom, 'center_real': center_real, 'center_imag': center_imag,
                'iterations': iterations, 'width': width, 'height': height
            })
    else:
        st.info("🔒 Controls are locked to prevent accidental changes.")

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
