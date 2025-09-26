import time, torch

class PromptStats:
    def __init__(self):
        self.prompt_start_time = None
        self.total_image_time = 0.0
        self.image_start_time = None
        self.peak_memories = []
        self.prompt_total_time = None
        self.avg_time_per_image = None
        self.avg_memory = None

    def start_prompt_timer(self):
        self.prompt_start_time = time.time()

    def stop_prompt_timer(self):
        if self.prompt_start_time is None:
            raise RuntimeError("Prompt timer not started. Call start_prompt_timer() first.")
        elapsed = time.time() - self.prompt_start_time
        self.prompt_total_time = elapsed
        self.prompt_start_time = None

    def start_image_time(self):
        self.image_start_time = time.time()

    def stop_image_time(self):
        if self.image_start_time is None:
            raise RuntimeError("Image timer not started. Call start_image_time() first.")
        torch.cuda.synchronize()  # Wait for GPU to finish computations
        elapsed = time.time() - self.image_start_time
        self.total_image_time += elapsed
        self.image_start_time = None
        return elapsed

    def record_peak_memory(self):
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        self.peak_memories.append(peak_memory_mb)

    def average_image_time(self, num_images):
        self.average_image_time =  self.total_image_time / num_images if num_images > 0 else 0.0

    def average_peak_memory(self):
        if self.peak_memories:
            self.avg_memory = sum(self.peak_memories) / len(self.peak_memories)
        return 0.0
    
    def get_summary(self, num_images):
        self.stop_prompt_timer()
        self.average_image_time(num_images)
        self.average_peak_memory()