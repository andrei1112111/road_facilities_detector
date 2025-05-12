#from light import Light
#from pitscndcracks import PitsAndCracks
#from segment import Segment

import os
import json
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PIL import Image
import threading

# Configuration
MAX_WORKERS = 4  # Adjust based on CPU cores
BATCH_SIZE = 1   # Process files in batches

class ImageProcessor:
    def __init__(self):
        self.queue = Queue()
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self.lock = threading.Lock()
        self.processed_count = 0

    def process_image(self, image_path):
        try:
            with Image.open(image_path) as img:
                metadata = {
                    "filename": os.path.basename(image_path),
                    "timestamp": time.time()
                    # !!!Обръаотка тут!!!
                    # детекторы сюда
                }
            
            json_path = f"{image_path}.json"
            with self.lock:
                with open(json_path, 'w') as f:
                    json.dump(metadata, f)
                self.processed_count += 1
            
            if self.processed_count % BATCH_SIZE == 0:
                print(f"Processed {self.processed_count} files...")
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

class BulkHandler(FileSystemEventHandler):
    def __init__(self, processor):
        self.processor = processor
        self.current_batch = []

    def on_created(self, event):
        print(f"Event: {event.event_type}")
        if not event.is_directory and event.src_path.lower().endswith(('.png','.jpg','.jpeg')):
            self.processor.executor.submit(
                self.processor.process_image,
                event.src_path
            )

if __name__ == "__main__":
    watch_dir = os.getenv('WATCH_DIR', '/app/external')
    
    os.makedirs(watch_dir, exist_ok=True)

    processor = ImageProcessor()
    event_handler = BulkHandler(processor)
    
    observer = Observer()
    observer.schedule(event_handler, watch_dir, recursive=True)
    observer.start()
    
    try:
        while True:
            time.sleep(5)
            print(f"Watching: {watch_dir} (Contents: {os.listdir(watch_dir)})")
            print(f"Status: {processor.processed_count} files processed | Queue: {processor.queue.qsize()}")
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    processor.executor.shutdown()