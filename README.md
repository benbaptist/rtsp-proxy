# RTSP Stream Proxy

A reliable RTSP proxy that handles unreliable camera streams by maintaining continuous output even during connection issues.

## Features

- Buffers and maintains continuous RTSP stream output
- Automatically replays last received frame during connection issues
- Displays error message after configurable timeout period
- Uses FFmpeg for efficient stream handling
- TCP transport for better reliability

## Requirements

- Python 3.7+
- FFmpeg installed on your system
- Python packages (install via `pip install -r requirements.txt`):
  - ffmpeg-python
  - numpy
  - opencv-python

## Installation

1. Install FFmpeg on your system if not already installed
   - macOS: `brew install ffmpeg`
   - Ubuntu/Debian: `sudo apt-get install ffmpeg`

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the proxy with:

```bash
python rtsp_proxy.py input_rtsp_url output_rtsp_url [--timeout SECONDS]
```

Example:
```bash
python rtsp_proxy.py rtsp://camera.local:554/stream rtsp://localhost:8554/stream --timeout 15
```

### Parameters

- `input_rtsp_url`: The source RTSP stream URL
- `output_rtsp_url`: The destination RTSP stream URL where the proxy will serve the stream
- `--timeout`: (Optional) Time in seconds before showing "No frames received" message (default: 15.0)

## How it Works

1. Continuously reads frames from the input RTSP stream
2. Buffers the most recent frame
3. If the input stream fails:
   - Continues outputting the last received frame
   - After the timeout period, displays "No frames received" message
4. Automatically reconnects when the input stream becomes available again

Press Ctrl+C to stop the proxy. 