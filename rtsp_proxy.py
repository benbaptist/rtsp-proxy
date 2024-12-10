import ffmpeg
import numpy as np
import cv2
import time
import threading
from queue import Queue
import subprocess
import sys
import select
import os
import signal
from typing import Optional, Tuple, Dict
from threading import Event

class RTSPProxy:
    def __init__(self, input_url: str, output_url: str, timeout_seconds: float = 15.0,
                 codec: str = 'h264', bitrate: str = '2M', preset: str = 'medium',
                 gop: int = 30, read_timeout: float = 5.0, fps: float = 30.0,
                 width: int = 1920, height: int = 1080):
        self.input_url = input_url
        self.output_url = output_url
        self.timeout_seconds = timeout_seconds
        self.read_timeout = read_timeout
        self.frame_queue = Queue(maxsize=1)
        self.last_frame_time = 0
        self.running = False
        self.last_frame: Optional[np.ndarray] = None
        
        # Resolution settings
        self.output_width = width
        self.output_height = height
        self.input_width = None   # Will be detected from input stream
        self.input_height = None
        self.scale_filter = None  # Will be set once input resolution is known
        
        self.fps = fps
        self.frame_interval = 1.0 / fps
        
        # Process management
        self.frame_available = Event()
        self.ffmpeg_input_process = None
        self.ffmpeg_output_process = None
        self.last_frame_received = time.time()
        self.last_frame_written = time.time()
        self.next_frame_time = time.time()
        
        # Encoding parameters
        self.codec = codec
        self.bitrate = bitrate
        self.preset = preset
        self.gop = gop

    def kill_process_and_children(self, process):
        """Kill a process and all its children more effectively."""
        if process is None:
            return
            
        try:
            # Send SIGTERM first
            process.terminate()
            try:
                process.wait(timeout=2)
                return
            except subprocess.TimeoutExpired:
                pass
                
            # If still running, force kill
            process.kill()
            process.wait(timeout=1)
            
            # On Unix-like systems, ensure child processes are killed
            if hasattr(os, 'killpg'):
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except:
                    pass
        except:
            pass

    def read_with_timeout(self, pipe, size, timeout):
        """Read from a pipe with timeout and better error handling."""
        if pipe is None:
            return None
            
        try:
            if select.select([pipe], [], [], timeout)[0]:
                data = pipe.read(size)
                if not data or len(data) != size:
                    return None
                return data
        except (select.error, IOError, ValueError):
            return None
        return None

    def get_codec_parameters(self) -> Dict:
        """Get FFmpeg parameters for the selected codec."""
        params = {
            'c:v': self.codec,
            'b:v': self.bitrate,
            'g': str(self.gop),
            'r': str(self.fps),
        }
        
        if self.codec in ['h264', 'libx264']:
            params.update({
                'preset': self.preset,
                'tune': 'zerolatency',
                'profile:v': 'main',
                'pix_fmt': 'yuv420p',
                'force_key_frames': f'expr:gte(t,n_forced*{self.gop}/{self.fps})'
            })
        elif self.codec == 'h265' or self.codec == 'libx265':
            params.update({
                'preset': self.preset,
                'x265-params': f'no-repeat-headers=1:keyint={self.gop}:min-keyint={self.gop}'
            })
        elif self.codec == 'copy':
            # Remove unnecessary parameters for stream copy
            params = {'c:v': 'copy'}
            
        return params

    def create_error_frame(self, message: str = "No frames received") -> np.ndarray:
        """Create a black frame with error message."""
        frame = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 2
        text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
        text_x = (self.output_width - text_size[0]) // 2
        text_y = (self.output_height + text_size[1]) // 2
        cv2.putText(frame, message, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        return frame

    def get_input_resolution(self) -> Tuple[int, int]:
        """Detect input stream resolution using FFprobe."""
        try:
            probe = ffmpeg.probe(self.input_url)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            return int(video_info['width']), int(video_info['height'])
        except Exception as e:
            print(f"Error detecting input resolution: {e}", file=sys.stderr)
            return self.output_width, self.output_height  # Fallback to output resolution

    def setup_scaling(self):
        """Configure scaling filter based on input and output resolutions."""
        # Detect input resolution if not already set
        if self.input_width is None or self.input_height is None:
            self.input_width, self.input_height = self.get_input_resolution()
        
        if (self.input_width, self.input_height) == (self.output_width, self.output_height):
            self.scale_filter = None  # No scaling needed
            return
        
        # Calculate scaling parameters to maintain aspect ratio
        input_aspect = self.input_width / self.input_height
        output_aspect = self.output_width / self.output_height
        
        if input_aspect > output_aspect:
            # Input is wider - fit to width
            new_width = self.output_width
            new_height = int(self.output_width / input_aspect)
            pad_top = (self.output_height - new_height) // 2
            pad_bottom = self.output_height - new_height - pad_top
            self.scale_filter = (
                f'scale={new_width}:{new_height}:force_original_aspect_ratio=decrease,'
                f'pad={self.output_width}:{self.output_height}:0:{pad_top}:black'
            )
        else:
            # Input is taller - fit to height
            new_height = self.output_height
            new_width = int(self.output_height * input_aspect)
            pad_left = (self.output_width - new_width) // 2
            pad_right = self.output_width - new_width - pad_left
            self.scale_filter = (
                f'scale={new_width}:{new_height}:force_original_aspect_ratio=decrease,'
                f'pad={self.output_width}:{self.output_height}:{pad_left}:0:black'
            )

    def read_stream(self):
        """Read frames from input RTSP stream."""
        while self.running:
            try:
                # Setup scaling on first run or after errors
                if self.scale_filter is None:
                    self.setup_scaling()
                
                # Build input stream with scaling if needed
                stream = (
                    ffmpeg
                    .input(self.input_url, rtsp_transport='tcp')
                )
                
                if self.scale_filter:
                    stream = stream.filter('scale', size=f'{self.output_width}x{self.output_height}',
                                        force_original_aspect_ratio='decrease')
                
                stream = (
                    stream
                    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                    .overwrite_output()
                )
                
                self.ffmpeg_input_process = stream.run_async(pipe_stdout=True, pipe_stderr=True)

                consecutive_failures = 0
                frame_size = self.output_width * self.output_height * 3

                while self.running:
                    in_bytes = self.read_with_timeout(self.ffmpeg_input_process.stdout, frame_size, self.read_timeout)
                    
                    if in_bytes is None or len(in_bytes) != frame_size:
                        consecutive_failures += 1
                        if consecutive_failures >= 3:
                            print("Multiple consecutive read failures, restarting input process...", file=sys.stderr)
                            self.kill_process_and_children(self.ffmpeg_input_process)
                            break
                        time.sleep(0.1)
                        continue

                    consecutive_failures = 0
                    frame = (
                        np.frombuffer(in_bytes, np.uint8)
                        .reshape([self.output_height, self.output_width, 3])
                    )
                    
                    current_time = time.time()
                    self.last_frame = frame.copy()
                    self.last_frame_time = current_time
                    self.last_frame_received = current_time
                    
                    while not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except:
                            break
                    self.frame_queue.put(frame)
                    self.frame_available.set()

            except Exception as e:
                print(f"Error in read_stream: {e}", file=sys.stderr)
                self.kill_process_and_children(self.ffmpeg_input_process)
                time.sleep(1)

    def write_stream(self):
        """Write frames to output RTSP stream with constant framerate."""
        while self.running:
            try:
                codec_params = self.get_codec_parameters()
                stream = (
                    ffmpeg
                    .input('pipe:', format='rawvideo', pix_fmt='rgb24',
                           s=f'{self.output_width}x{self.output_height}',
                           framerate=str(self.fps))  # Set input framerate
                    .output(self.output_url, format='rtsp', rtsp_transport='tcp', **codec_params)
                    .overwrite_output()
                )
                
                self.ffmpeg_output_process = stream.run_async(pipe_stdin=True)
                self.next_frame_time = time.time()  # Initialize next frame time
                
                while self.running:
                    current_time = time.time()
                    
                    # Wait until it's time for the next frame
                    sleep_time = self.next_frame_time - current_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    
                    # Update next frame time
                    self.next_frame_time += self.frame_interval
                    # If we're falling behind, reset to maintain sync
                    if current_time > self.next_frame_time + self.frame_interval:
                        self.next_frame_time = current_time + self.frame_interval
                    
                    # Try to get a new frame
                    frame = None
                    if self.frame_available.is_set():
                        try:
                            frame = self.frame_queue.get_nowait()
                            self.frame_available.clear()
                            self.last_frame = frame.copy()
                            self.last_frame_written = current_time
                        except:
                            pass
                    
                    # If no new frame, check if we should use last frame or error frame
                    if frame is None:
                        if current_time - self.last_frame_received > self.timeout_seconds:
                            frame = self.create_error_frame()
                        elif self.last_frame is not None:
                            frame = self.last_frame
                        else:
                            frame = self.create_error_frame()
                    
                    # Write frame
                    try:
                        self.ffmpeg_output_process.stdin.write(frame.tobytes())
                    except:
                        print("Error writing to FFmpeg output", file=sys.stderr)
                        break
                    
                    # Check process health
                    if self.ffmpeg_output_process.poll() is not None:
                        print("FFmpeg output process died, restarting...", file=sys.stderr)
                        break

            except Exception as e:
                print(f"Error in write_stream: {e}", file=sys.stderr)
                self.kill_process_and_children(self.ffmpeg_output_process)
                time.sleep(1)

    def start(self):
        """Start the RTSP proxy with improved process management."""
        self.running = True
        
        self.reader_thread = threading.Thread(target=self.read_stream)
        self.reader_thread.daemon = True
        self.reader_thread.start()

        self.writer_thread = threading.Thread(target=self.write_stream)
        self.writer_thread.daemon = True
        self.writer_thread.start()

    def stop(self):
        """Stop the RTSP proxy and cleanup resources."""
        self.running = False
        self.frame_available.set()  # Unblock any waiting threads
        
        # Kill FFmpeg processes
        self.kill_process_and_children(self.ffmpeg_input_process)
        self.kill_process_and_children(self.ffmpeg_output_process)
        
        # Wait for threads to finish
        if hasattr(self, 'reader_thread'):
            self.reader_thread.join(timeout=2)
        if hasattr(self, 'writer_thread'):
            self.writer_thread.join(timeout=2)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RTSP Proxy with frame buffering')
    parser.add_argument('input_url', help='Input RTSP URL')
    parser.add_argument('output_url', help='Output RTSP URL')
    parser.add_argument('--timeout', type=float, default=15.0,
                      help='Timeout in seconds before showing error message (default: 15.0)')
    parser.add_argument('--read-timeout', type=float, default=5.0,
                      help='Timeout in seconds for reading from FFmpeg (default: 5.0)')
    parser.add_argument('--codec', type=str, default='h264',
                      choices=['libx264', 'libx265', 'copy'],
                      help='Output codec (default: libx264)')
    parser.add_argument('--bitrate', type=str, default='2M',
                      help='Output bitrate (e.g., 2M, 4M, 8M) (default: 2M)')
    parser.add_argument('--preset', type=str, default='medium',
                      choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'],
                      help='Encoding preset (default: medium)')
    parser.add_argument('--gop', type=int, default=30,
                      help='GOP size (default: 30)')
    parser.add_argument('--fps', type=float, default=30.0,
                      help='Output framerate (default: 30.0)')
    parser.add_argument('--width', type=int, default=1920,
                      help='Output width (default: 1920)')
    parser.add_argument('--height', type=int, default=1080,
                      help='Output height (default: 1080)')
    args = parser.parse_args()

    proxy = RTSPProxy(
        args.input_url,
        args.output_url,
        args.timeout,
        codec=args.codec,
        bitrate=args.bitrate,
        preset=args.preset,
        gop=args.gop,
        read_timeout=args.read_timeout,
        fps=args.fps,
        width=args.width,
        height=args.height
    )
    
    try:
        proxy.start()
        print(f"RTSP Proxy started")
        print(f"Input URL: {args.input_url}")
        print(f"Output URL: {args.output_url}")
        print(f"Codec: {args.codec}")
        print(f"Bitrate: {args.bitrate}")
        print(f"Preset: {args.preset}")
        print(f"GOP size: {args.gop}")
        print(f"Output FPS: {args.fps}")
        print(f"Output resolution: {args.width}x{args.height}")
        print("Press Ctrl+C to stop")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping proxy...")
        proxy.stop()
        print("Proxy stopped") 