import ffmpeg
import numpy as np
import cv2
import time
import threading
from queue import Queue
import subprocess
import sys
import select
from typing import Optional, Tuple, Dict

class RTSPProxy:
    def __init__(self, input_url: str, output_url: str, timeout_seconds: float = 15.0,
                 codec: str = 'h264', bitrate: str = '2M', preset: str = 'medium',
                 gop: int = 30, read_timeout: float = 5.0):
        self.input_url = input_url
        self.output_url = output_url
        self.timeout_seconds = timeout_seconds
        self.read_timeout = read_timeout  # Timeout for reading from FFmpeg
        self.frame_queue = Queue(maxsize=1)  # Only keep latest frame
        self.last_frame_time = 0
        self.running = False
        self.last_frame: Optional[np.ndarray] = None
        self.frame_width = 1920  # Default resolution
        self.frame_height = 1080
        
        # Encoding parameters
        self.codec = codec
        self.bitrate = bitrate
        self.preset = preset
        self.gop = gop

    def kill_process_and_children(self, process):
        """Kill a process and all its children."""
        try:
            process.kill()
        except:
            pass
        try:
            process.wait(timeout=1)
        except:
            pass

    def read_with_timeout(self, pipe, size, timeout):
        """Read from a pipe with timeout."""
        if select.select([pipe], [], [], timeout)[0]:
            return pipe.read(size)
        return None

    def get_codec_parameters(self) -> Dict:
        """Get FFmpeg parameters for the selected codec."""
        params = {
            'c:v': self.codec,
            'b:v': self.bitrate,
            'g': str(self.gop),  # GOP size
        }
        
        # Codec-specific parameters
        if self.codec in ['h264', 'libx264']:
            params.update({
                'preset': self.preset,
                'tune': 'zerolatency',
                'profile:v': 'main',
                "pix_fmt": "yuv420p"
            })
        elif self.codec == 'h265' or self.codec == 'libx265':
            params.update({
                'preset': self.preset,
                'x265-params': 'no-repeat-headers=1',
            })
        elif self.codec == 'copy':
            # Remove unnecessary parameters for stream copy
            params = {'c:v': 'copy'}
            
        return params

    def create_error_frame(self, message: str = "No frames received") -> np.ndarray:
        """Create a black frame with error message."""
        frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 2
        text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
        text_x = (self.frame_width - text_size[0]) // 2
        text_y = (self.frame_height + text_size[1]) // 2
        cv2.putText(frame, message, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        return frame

    def read_stream(self):
        """Read frames from input RTSP stream."""
        while self.running:
            process = None
            try:
                process = (
                    ffmpeg
                    .input(self.input_url, rtsp_transport='tcp')
                    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                    .overwrite_output()
                    .run_async(pipe_stdout=True, pipe_stderr=True)
                )

                last_successful_read = time.time()
                frame_size = self.frame_width * self.frame_height * 3

                while self.running:
                    # Read with timeout
                    in_bytes = self.read_with_timeout(process.stdout, frame_size, self.read_timeout)
                    
                    if in_bytes is None or len(in_bytes) != frame_size:
                        current_time = time.time()
                        if current_time - last_successful_read > self.read_timeout:
                            print("FFmpeg input process timed out, restarting...", file=sys.stderr)
                            self.kill_process_and_children(process)
                            break
                        continue

                    frame = (
                        np.frombuffer(in_bytes, np.uint8)
                        .reshape([self.frame_height, self.frame_width, 3])
                    )
                    
                    # Update last frame and timestamp
                    self.last_frame = frame
                    self.last_frame_time = time.time()
                    last_successful_read = self.last_frame_time
                    
                    # Update queue with latest frame
                    if not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except:
                            pass
                    self.frame_queue.put(frame)

            except Exception as e:
                print(f"Error reading stream: {e}", file=sys.stderr)
                if process:
                    self.kill_process_and_children(process)
                time.sleep(1)  # Wait before retrying

    def write_stream(self):
        """Write frames to output RTSP stream."""
        while self.running:
            process = None
            try:
                # Get codec parameters
                codec_params = self.get_codec_parameters()
                
                # Build ffmpeg command
                stream = (
                    ffmpeg
                    .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{self.frame_width}x{self.frame_height}')
                    .output(self.output_url, format='rtsp', rtsp_transport='tcp', **codec_params)
                    .overwrite_output()
                )
                
                # For debugging: print the ffmpeg command
                print("FFmpeg command:", ' '.join(stream.compile()))
                
                process = stream.run_async(pipe_stdin=True)
                last_write_time = time.time()

                while self.running:
                    current_time = time.time()
                    frame = None

                    try:
                        frame = self.frame_queue.get_nowait()
                        last_write_time = current_time
                    except:
                        if self.last_frame is not None:
                            if current_time - self.last_frame_time > self.timeout_seconds:
                                frame = self.create_error_frame()
                            else:
                                frame = self.last_frame
                        else:
                            frame = self.create_error_frame()

                    if frame is not None:
                        try:
                            process.stdin.write(frame.tobytes())
                            last_write_time = current_time
                        except:
                            print("Error writing to FFmpeg, restarting output process...", file=sys.stderr)
                            self.kill_process_and_children(process)
                            break

                    # Check if we haven't written anything for too long
                    if current_time - last_write_time > self.read_timeout:
                        print("FFmpeg output process timed out, restarting...", file=sys.stderr)
                        self.kill_process_and_children(process)
                        break

                    time.sleep(1/30)  # Limit to 30 fps

            except Exception as e:
                print(f"Error writing stream: {e}", file=sys.stderr)
                if process:
                    self.kill_process_and_children(process)
                time.sleep(1)  # Wait before retrying

    def start(self):
        """Start the RTSP proxy."""
        self.running = True
        
        # Start reader thread
        self.reader_thread = threading.Thread(target=self.read_stream)
        self.reader_thread.daemon = True
        self.reader_thread.start()

        # Start writer thread
        self.writer_thread = threading.Thread(target=self.write_stream)
        self.writer_thread.daemon = True
        self.writer_thread.start()

    def stop(self):
        """Stop the RTSP proxy."""
        self.running = False
        self.reader_thread.join()
        self.writer_thread.join()

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
    args = parser.parse_args()

    proxy = RTSPProxy(
        args.input_url,
        args.output_url,
        args.timeout,
        codec=args.codec,
        bitrate=args.bitrate,
        preset=args.preset,
        gop=args.gop,
        read_timeout=args.read_timeout
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
        print("Press Ctrl+C to stop")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping proxy...")
        proxy.stop()
        print("Proxy stopped") 