import ffmpeg
import numpy as np
import cv2
import time
import threading
from queue import Queue
import subprocess
import sys
from typing import Optional, Tuple

class RTSPProxy:
    def __init__(self, input_url: str, output_url: str, timeout_seconds: float = 15.0):
        self.input_url = input_url
        self.output_url = output_url
        self.timeout_seconds = timeout_seconds
        self.frame_queue = Queue(maxsize=1)  # Only keep latest frame
        self.last_frame_time = 0
        self.running = False
        self.last_frame: Optional[np.ndarray] = None
        self.frame_width = 1920  # Default resolution
        self.frame_height = 1080

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
            try:
                process = (
                    ffmpeg
                    .input(self.input_url, rtsp_transport='tcp')
                    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                    .overwrite_output()
                    .run_async(pipe_stdout=True, pipe_stderr=True)
                )

                while self.running:
                    frame_size = self.frame_width * self.frame_height * 3
                    in_bytes = process.stdout.read(frame_size)
                    if not in_bytes:
                        break
                    
                    frame = (
                        np.frombuffer(in_bytes, np.uint8)
                        .reshape([self.frame_height, self.frame_width, 3])
                    )
                    
                    # Update last frame and timestamp
                    self.last_frame = frame
                    self.last_frame_time = time.time()
                    
                    # Update queue with latest frame
                    if not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except:
                            pass
                    self.frame_queue.put(frame)

                process.kill()
            except Exception as e:
                print(f"Error reading stream: {e}", file=sys.stderr)
                time.sleep(1)  # Wait before retrying

    def write_stream(self):
        """Write frames to output RTSP stream."""
        while self.running:
            try:
                process = (
                    ffmpeg
                    .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{self.frame_width}x{self.frame_height}')
                    .output(self.output_url, format='rtsp', rtsp_transport='tcp')
                    .overwrite_output()
                    .run_async(pipe_stdin=True)
                )

                while self.running:
                    current_time = time.time()
                    frame = None

                    # Check if we have a new frame
                    try:
                        frame = self.frame_queue.get_nowait()
                    except:
                        # No new frame available
                        if self.last_frame is not None:
                            if current_time - self.last_frame_time > self.timeout_seconds:
                                frame = self.create_error_frame()
                            else:
                                frame = self.last_frame
                        else:
                            frame = self.create_error_frame()

                    if frame is not None:
                        process.stdin.write(frame.tobytes())

                    time.sleep(1/30)  # Limit to 30 fps

                process.kill()
            except Exception as e:
                print(f"Error writing stream: {e}", file=sys.stderr)
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
    args = parser.parse_args()

    proxy = RTSPProxy(args.input_url, args.output_url, args.timeout)
    
    try:
        proxy.start()
        print(f"RTSP Proxy started")
        print(f"Input URL: {args.input_url}")
        print(f"Output URL: {args.output_url}")
        print("Press Ctrl+C to stop")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping proxy...")
        proxy.stop()
        print("Proxy stopped") 