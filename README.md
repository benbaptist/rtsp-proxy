# RTSP Stream Proxy

A reliable RTSP proxy that handles unreliable camera streams by maintaining continuous output even during connection issues.

## Features

- Buffers and maintains continuous RTSP stream output
- Automatically replays last received frame during connection issues
- Displays error message after configurable timeout period
- Uses FFmpeg for efficient stream handling
- TCP transport for better reliability
- Configurable output codec, bitrate, and encoding parameters
- Support for H.264, H.265, and stream copy

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
python rtsp_proxy.py input_rtsp_url output_rtsp_url [options]
```

Example with default settings:
```bash
python rtsp_proxy.py rtsp://camera.local:554/stream rtsp://localhost:8554/stream
```

Example with custom encoding settings:
```bash
python rtsp_proxy.py rtsp://camera.local:554/stream rtsp://localhost:8554/stream \
    --codec h264 --bitrate 4M --preset fast --gop 60
```

### Parameters

Required:
- `input_rtsp_url`: The source RTSP stream URL
- `output_rtsp_url`: The destination RTSP stream URL where the proxy will serve the stream

Optional:
- `--timeout SECONDS`: Time before showing "No frames received" message (default: 15.0)
- `--codec {h264,h265,copy}`: Output codec (default: h264)
  - `h264`: H.264/AVC encoding
  - `h265`: H.265/HEVC encoding
  - `copy`: Stream copy (no re-encoding)
- `--bitrate BITRATE`: Output bitrate (e.g., 2M, 4M, 8M) (default: 2M)
- `--preset {ultrafast,superfast,veryfast,faster,fast,medium,slow,slower,veryslow}`: 
  Encoding preset (default: medium)
- `--gop GOP_SIZE`: GOP (Group of Pictures) size (default: 30)

## How it Works

1. Continuously reads frames from the input RTSP stream
2. Buffers the most recent frame
3. If the input stream fails:
   - Continues outputting the last received frame
   - After the timeout period, displays "No frames received" message
4. Automatically reconnects when the input stream becomes available again
5. Transcodes the stream using the specified codec and parameters (unless using copy mode)

### Encoding Presets

The preset determines the encoding speed vs quality tradeoff:
- `ultrafast`: Fastest encoding, larger file size, lower quality
- `medium`: Balanced encoding speed and quality (default)
- `veryslow`: Highest quality, smallest file size, but slower encoding

For low-latency applications, use faster presets like `ultrafast` or `superfast`.

Press Ctrl+C to stop the proxy. 