from typing import Optional, Callable
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class BatchProgressTracker:
    """
    A progress tracker for batch processing tasks that provides both console progress bar
    and callback-based progress updates.
    """
    def __init__(self, total: int, desc: str = "Processing", 
                 callback: Optional[Callable[[str, float, int, int, int], None]] = None,
                 silent: bool = False):
        """
        Initialize the progress tracker.
        
        Args:
            total: Total number of items to process
            desc: Description for the progress bar
            callback: Optional callback function that receives
                     (status, progress_percentage, completed, failed, in_progress)
            silent: If True, suppress progress bar but still track progress
        """
        self.total = total
        self.completed = 0
        self.failed = 0
        self.in_progress = 0
        self.callback = callback
        self.silent = silent
        
        if not silent:
            self.pbar = tqdm(total=total, desc=desc, unit='items')
        else:
            self.pbar = None

    def update(self, completed_delta: int = 0, failed_delta: int = 0, 
               in_progress_delta: int = 0) -> None:
        """
        Update progress counters and trigger progress reporting.
        
        Args:
            completed_delta: Number of newly completed items
            failed_delta: Number of newly failed items
            in_progress_delta: Change in number of in-progress items
        """
        self.completed += completed_delta
        self.failed += failed_delta
        self.in_progress += in_progress_delta
        
        # Update the progress bar if not in silent mode
        if self.pbar is not None:
            self.pbar.update(completed_delta)
            self.pbar.set_postfix({
                'completed': self.completed,
                'failed': self.failed,
                'in_progress': self.in_progress
            })
        
        # Calculate progress percentage
        progress = (self.completed + self.failed) / self.total * 100
        
        # Log progress at appropriate intervals if in silent mode
        if self.silent and (self.completed + self.failed) % max(1, self.total // 10) == 0:
            logger.info(f"Progress: {progress:.1f}% ({self.completed + self.failed}/{self.total})")
        
        # Call the callback if provided
        if self.callback:
            status = f"Processing: {progress:.1f}% complete"
            self.callback(status, progress, self.completed, self.failed, self.in_progress)
            
    def close(self) -> None:
        """Close the progress bar if it exists."""
        if self.pbar is not None:
            self.pbar.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()