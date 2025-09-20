# Samsara Helix Dashboard

An interactive web application for generating Mandelbrot fractals and synthesizing multi-frequency audio patterns. Built with Streamlit for easy deployment and mobile-friendly usage.

## Features

- **Interactive Fractal Generation**: Real-time Mandelbrot set visualization with adjustable zoom, center coordinates, and iteration counts
- **Audio Synthesis**: Multi-frequency sine wave generation with customizable base and harmony frequencies
- **Sanskrit Text Overlays**: Cultural text elements with traditional mantras displayed on fractals
- **Mobile Optimized**: Responsive design that works on smartphones and tablets
- **Export Functionality**: Download generated fractals as PNG images and audio as WAV files
- **Parameter Management**: Save and load custom settings as JSON files

## Live Demo

Visit the deployed application: [Your App URL will appear here after deployment]

## Local Development

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/samsara-helix-dashboard.git
cd samsara-helix-dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run streamlit_app.py
```

4. Open your browser to `http://localhost:8501`

## Usage

### Fractal Generation
1. Navigate to the "Fractal" tab
2. Adjust parameters using the sliders:
   - **Zoom Level**: Controls magnification (0.1x to 50x)
   - **Center Coordinates**: Sets the focus point in the complex plane
   - **Iterations**: Determines calculation precision (50-300)
   - **Resolution**: Choose image size for performance vs quality
   - **Color Scheme**: Select from matplotlib colormaps
3. Click "Generate Fractal" or enable auto-generation
4. Sanskrit overlay text will be added to the generated image

### Audio Synthesis
1. Navigate to the "Audio" tab
2. Set frequency parameters:
   - **Base Frequency**: Primary tone (50-300 Hz)
   - **Harmony Frequency**: Secondary tone (200-800 Hz)
   - **Duration**: Audio length (1-30 seconds)
   - **Sample Rate**: Audio quality (22.05kHz or 44.1kHz)
3. Click "Generate Audio" to create the sound
4. Use the built-in player to listen to the result

### Export Options
1. Go to the "Export" tab
2. Download generated fractals as PNG images
3. Save audio synthesis as WAV files
4. Export current parameters as JSON for later use

## Technical Details

### Fractal Mathematics
The application generates Mandelbrot sets using the iterative formula:
```
z(n+1) = z(n)² + c
```
Where `c` is a complex number representing each pixel coordinate, and the algorithm determines whether the sequence diverges.

### Audio Synthesis
Multi-frequency audio is created by combining sine waves:
- Base frequency (configurable)
- Harmony frequency (configurable) 
- Additional harmonic overtones (1.5x and 2x harmony frequency)
- Amplitude modulation for breathing-like envelope effects

### Performance Considerations
- Default settings are optimized for mobile devices
- Higher resolutions and iteration counts increase computation time
- Audio generation scales linearly with duration and sample rate

## Deployment

### Streamlit Community Cloud
1. Fork this repository to your GitHub account
2. Sign up at [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository and set `streamlit_app.py` as the main file
5. Deploy and share your public URL

### Other Platforms
The application can also be deployed on:
- Heroku
- Railway
- Google Cloud Run
- Any platform supporting Python web applications

## File Structure
```
samsara-helix-dashboard/
├── streamlit_app.py      # Main application code
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Dependencies
- **streamlit**: Web application framework
- **numpy**: Numerical computing for fractal mathematics
- **matplotlib**: Plotting and color mapping
- **Pillow**: Image processing for text overlays
- **scipy**: Scientific computing for audio file generation

## Contributing
This is a personal project, but suggestions and improvements are welcome. Please open an issue or submit a pull request.

## License
This project is open source and available under the MIT License.

## Technical Notes

### Mathematical Accuracy
The fractal generation uses standard Mandelbrot set mathematics with proper complex number arithmetic and escape-time algorithms.

### Audio Engineering
Audio synthesis employs basic signal processing principles to create sine wave combinations. The frequencies and harmonics are mathematically derived.

### Cultural Elements
Sanskrit text overlays are included as cultural and artistic elements. The display of traditional mantras is for aesthetic purposes and cultural appreciation.

## Troubleshooting

### Common Issues
- **Slow generation**: Reduce resolution or iteration count
- **Memory errors**: Lower the zoom level or image size
- **Audio not playing**: Check browser audio permissions
- **Mobile performance**: Use default settings for optimal experience

### Browser Compatibility
- Chrome: Full support
- Firefox: Full support  
- Safari: Full support
- Mobile browsers: Optimized responsive design

## Contact
For questions or issues, please open a GitHub issue in this repository.
