# src/clients/openai/batch_client.py

import os
import json
import asyncio
import tempfile
import time
import aiohttp
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import httpx
import tiktoken

from src.clients.base import BaseClient
from src.models.registry import registry
from src.rate_limiting.limiter import RateLimiter
from src.utils.logger import get_logger

logger = get_logger(__name__)

class OpenAIBatchClient(BaseClient):
    """Client for OpenAI's batch processing API.
    
    The Batch API offers:
    - 50% cost discount compared to synchronous APIs
    - Higher rate limits (separate pool from standard rate limits)
    - Guaranteed completion within 24 hours
    """
    
    # API endpoints
    API_URL_FILES = "https://api.openai.com/v1/files"
    API_URL_BATCH = "https://api.openai.com/v1/batches"
    API_URL = "https://api.openai.com/v1/chat/completions"
    
    # Status values for batch jobs
    TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}
    
    def __init__(self, api_key: str):
        """Initialize the batch client with API key and rate limiters."""
        super().__init__(api_key)
        
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        # Load batch API configuration from registry
        batch_config = registry.get_batch_config("openai")
        
        if batch_config:
            # Use configuration from YAML
            self.MAX_REQUESTS_PER_BATCH = batch_config.max_requests_per_batch
            self.MAX_FILE_SIZE_BYTES = batch_config.max_file_size_bytes
            
            # Create rate limiters for different operations from config
            self.limiters = {}
            if batch_config.rate_limits:
                for operation, limits in batch_config.rate_limits.items():
                    if "calls" in limits and "period" in limits:
                        self.limiters[operation] = RateLimiter(
                            limits["calls"], limits["period"]
                        )
        else:
            # Default values if configuration is not available
            logger.warning("Batch API configuration not found. Using default values.")
            self.MAX_REQUESTS_PER_BATCH = 50000
            self.MAX_FILE_SIZE_BYTES = 200 * 1024 * 1024  # 200 MB
            
            # Default rate limits
            default_limits = {
                "file_upload": {"calls": 30, "period": 60},
                "batch_create": {"calls": 60, "period": 60},
                "batch_retrieve": {"calls": 300, "period": 60},
                "file_download": {"calls": 60, "period": 60}
            }
            
            self.limiters = {
                action: RateLimiter(limits["calls"], limits["period"])
                for action, limits in default_limits.items()
            }
        
        # Default polling intervals for status checks (with exponential backoff)
        self.initial_poll_interval = 5
        self.max_poll_interval = 60
        self.poll_backoff_factor = 1.5
        
        logger.info(f"Initialized OpenAI batch client (max batch: {self.MAX_REQUESTS_PER_BATCH}, " +
                   f"max size: {self.MAX_FILE_SIZE_BYTES/1024/1024:.1f}MB)")
    
    def get_rate_limiter(self, model: str) -> RateLimiter:
        """Get the appropriate rate limiter for a model.
        
        For batch operations, we use operation-specific limiters rather than
        model-specific ones.
        
        Args:
            model: Model name (unused in batch client)
            
        Returns:
            The batch_retrieve rate limiter as a fallback
        """
        # For compatibility with BaseClient interface
        return self.limiters["batch_retrieve"]
    
    async def make_request(
        self, 
        session: aiohttp.ClientSession, 
        model: str, 
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Standard request method for compatibility with BaseClient.
        
        This is included to maintain compatibility with the BaseClient interface.
        For actual batch processing, use process_batch() instead.
        
        Args:
            session: HTTP session
            model: Model name
            data: Request data
            
        Returns:
            API response
        """
        # For one-off requests, use the regular API
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Validate model exists in configuration
        try:
            model_config = registry.get_model(model)
            model_name = model_config.name
        except ValueError as e:
            logger.error(f"Invalid model requested: {model}")
            raise e
        
        # Use batch retrieve limiter as a fallback
        await self.limiters["batch_retrieve"].acquire()
        
        # Update model name from config
        data["model"] = model_name
        
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ValueError(f"Error {response.status}: {error_text}")
            
            return await response.json()
    
    async def upload_file(
        self, 
        file_path: Path,
        session: Optional[aiohttp.ClientSession] = None
    ) -> str:
        """Upload a JSONL file to OpenAI for batch processing.
        
        Args:
            file_path: Path to the JSONL file containing batch requests
            session: Optional HTTP session (creates one if not provided)
            
        Returns:
            File ID from OpenAI
            
        Raises:
            ValueError: If the file exceeds size limits or upload fails
        """
        # Check file size before uploading
        file_size = os.path.getsize(file_path)
        if file_size > self.MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"File size ({file_size} bytes) exceeds maximum allowed "
                f"({self.MAX_FILE_SIZE_BYTES} bytes)"
            )
        
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Determine if we need to create and close the session
        should_close_session = session is None
        session = session or aiohttp.ClientSession()
        
        try:
            logger.info(f"Uploading file {file_path.name} ({file_size} bytes)")
            
            with open(file_path, "rb") as f:
                form_data = aiohttp.FormData()
                form_data.add_field(
                    "file",
                    f,
                    filename=file_path.name,
                    content_type="application/jsonl"
                )
                form_data.add_field("purpose", "batch")
                
                # Respect rate limits for file uploads
                await self.limiters["file_upload"].acquire()
                
                async with session.post(
                    self.API_URL_FILES,
                    headers=headers,
                    data=form_data
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"File upload failed: {error_text}")
                    
                    result = await response.json()
                    logger.info(f"File uploaded successfully, ID: {result['id']}")
                    return result["id"]
        finally:
            if should_close_session:
                await session.close()
    
    async def create_batch(
        self, 
        file_id: str, 
        model: str,
        completion_window: str = "24h",
        metadata: Optional[Dict[str, str]] = None,
        session: Optional[aiohttp.ClientSession] = None
    ) -> str:
        """Create a batch processing job.
        
        Args:
            file_id: ID of the uploaded file
            model: Model name
            completion_window: Time window for completion ("24h")
            metadata: Optional metadata for the batch
            session: Optional HTTP session
            
        Returns:
            Batch ID
            
        Raises:
            ValueError: If batch creation fails
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Validate model exists in configuration
        try:
            model_config = registry.get_model(model)
            model_name = model_config.name
            logger.info(f"Using model {model} (API name: {model_name}) for batch processing")
        except ValueError as e:
            logger.error(f"Invalid model requested: {model}")
            raise e
        
        data = {
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": completion_window,
            "model": model_name
        }
        
        # Add metadata if provided
        if metadata:
            data["metadata"] = metadata
        
        # Determine if we need to create and close the session
        should_close_session = session is None
        session = session or aiohttp.ClientSession()
        
        try:
            logger.info(f"Creating batch job with model {model}")
            
            # Respect rate limits for batch creation
            await self.limiters["batch_create"].acquire()
            
            async with session.post(
                self.API_URL_BATCH,
                headers=headers,
                json=data
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"Batch creation failed: {error_text}")
                
                result = await response.json()
                logger.info(f"Batch created successfully, ID: {result['id']}")
                return result["id"]
        finally:
            if should_close_session:
                await session.close()
    
    async def get_batch_status(
        self, 
        batch_id: str,
        session: Optional[aiohttp.ClientSession] = None
    ) -> Dict[str, Any]:
        """Get the status of a batch job.
        
        Args:
            batch_id: Batch ID
            session: Optional HTTP session
            
        Returns:
            Batch status information
            
        Raises:
            ValueError: If status retrieval fails
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Determine if we need to create and close the session
        should_close_session = session is None
        session = session or aiohttp.ClientSession()
        
        try:
            # Respect rate limits for status checks
            await self.limiters["batch_retrieve"].acquire()
            
            async with session.get(
                f"{self.API_URL_BATCH}/{batch_id}",
                headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"Failed to get batch status: {error_text}")
                
                return await response.json()
        finally:
            if should_close_session:
                await session.close()
    
    async def download_results(
        self, 
        file_id: str,
        output_path: Optional[Path] = None,
        session: Optional[aiohttp.ClientSession] = None
    ) -> Path:
        """Download batch results to a file.
        
        Args:
            file_id: File ID of the results
            output_path: Path to save results (creates temp file if None)
            session: Optional HTTP session
            
        Returns:
            Path to the downloaded results file
            
        Raises:
            ValueError: If download fails
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Create output path if not provided
        if output_path is None:
            fd, temp_name = tempfile.mkstemp(suffix='.jsonl')
            os.close(fd)
            output_path = Path(temp_name)
        
        # Determine if we need to create and close the session
        should_close_session = session is None
        session = session or aiohttp.ClientSession()
        
        try:
            logger.info(f"Downloading results to {output_path}")
            
            # Respect rate limits for file downloads
            await self.limiters["file_download"].acquire()
            
            async with session.get(
                f"{self.API_URL_FILES}/{file_id}/content",
                headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"Failed to download results: {error_text}")
                
                content = await response.read()
                
                with open(output_path, "wb") as f:
                    f.write(content)
                
                logger.info(f"Results downloaded to {output_path}")
                return output_path
        finally:
            if should_close_session:
                await session.close()
    
    async def cancel_batch(
        self,
        batch_id: str,
        session: Optional[aiohttp.ClientSession] = None
    ) -> Dict[str, Any]:
        """Cancel a batch job.
        
        Args:
            batch_id: Batch ID to cancel
            session: Optional HTTP session
            
        Returns:
            Cancellation status information
            
        Raises:
            ValueError: If cancellation fails
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Determine if we need to create and close the session
        should_close_session = session is None
        session = session or aiohttp.ClientSession()
        
        try:
            logger.info(f"Cancelling batch {batch_id}")
            
            # Respect rate limits for batch operations
            await self.limiters["batch_create"].acquire()
            
            async with session.post(
                f"{self.API_URL_BATCH}/{batch_id}/cancel",
                headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"Failed to cancel batch: {error_text}")
                
                result = await response.json()
                logger.info(f"Batch {batch_id} cancellation requested")
                return result
        finally:
            if should_close_session:
                await session.close()
    
    async def wait_for_batch_completion(
        self,
        batch_id: str,
        session: Optional[aiohttp.ClientSession] = None,
        initial_poll_interval: Optional[int] = None,
        max_poll_interval: Optional[int] = None,
        poll_backoff_factor: Optional[float] = None,
        timeout: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Wait for a batch job to complete, with exponential backoff.
        
        Args:
            batch_id: Batch ID to monitor
            session: Optional HTTP session
            initial_poll_interval: Initial seconds between status checks
            max_poll_interval: Maximum seconds between status checks
            poll_backoff_factor: Factor to increase interval by
            timeout: Maximum time to wait in seconds (None for no limit)
            progress_callback: Optional function to call with status updates
            
        Returns:
            Tuple of (output_file_id, final_status_info)
            
        Raises:
            TimeoutError: If timeout is reached
            RuntimeError: If batch fails or is cancelled
        """
        # Use provided values or defaults
        poll_interval = initial_poll_interval or self.initial_poll_interval
        max_interval = max_poll_interval or self.max_poll_interval
        backoff = poll_backoff_factor or self.poll_backoff_factor
        
        start_time = time.time()
        should_close_session = session is None
        session = session or aiohttp.ClientSession()
        
        try:
            while True:
                # Check if we've exceeded timeout
                if timeout and (time.time() - start_time > timeout):
                    raise TimeoutError(f"Batch {batch_id} did not complete within timeout")
                
                # Get status
                status_info = await self.get_batch_status(batch_id, session)
                status = status_info.get("status")
                
                # Calculate progress metrics
                total_jobs = status_info.get("total_jobs", 0)
                completed_jobs = status_info.get("completed_jobs", 0)
                failed_jobs = status_info.get("failed_jobs", 0)
                in_progress_jobs = status_info.get("in_progress_jobs", 0)
                
                # Call progress callback if provided
                if progress_callback and total_jobs > 0:
                    progress_pct = (completed_jobs + failed_jobs) / total_jobs * 100
                    progress_callback(status, progress_pct, completed_jobs, failed_jobs, in_progress_jobs)
                
                # Log progress
                if total_jobs > 0:
                    progress_pct = (completed_jobs + failed_jobs) / total_jobs * 100
                    logger.info(
                        f"Batch {batch_id} status: {status} - "
                        f"Progress: {progress_pct:.1f}% "
                        f"({completed_jobs + failed_jobs}/{total_jobs}) - "
                        f"Completed: {completed_jobs}, Failed: {failed_jobs}, "
                        f"In progress: {in_progress_jobs}"
                    )
                else:
                    logger.info(f"Batch {batch_id} status: {status}")
                
                # Check if batch is in a terminal state
                if status in self.TERMINAL_STATUSES:
                    if status == "completed":
                        logger.info(f"Batch {batch_id} completed successfully")
                        return status_info.get("output_file_id"), status_info
                    elif status == "failed":
                        error_msg = status_info.get("error", {}).get("message", "Unknown error")
                        raise RuntimeError(f"Batch failed: {error_msg}")
                    elif status == "expired":
                        raise RuntimeError("Batch expired: did not complete within 24-hour window")
                    elif status == "cancelled":
                        raise RuntimeError("Batch was cancelled")
                
                # Exponential backoff with capping
                await asyncio.sleep(poll_interval)
                poll_interval = min(poll_interval * backoff, max_interval)
        finally:
            if should_close_session:
                await session.close()
    
    async def process_batch(
        self,
        file_path: Path,
        model: str,
        output_path: Optional[Path] = None,
        metadata: Optional[Dict[str, str]] = None,
        completion_window: str = "24h",
        progress_callback: Optional[callable] = None
    ) -> Path:
        """Process a batch from file upload to result download.
        
        This is the main method that orchestrates the entire batch workflow.
        
        Args:
            file_path: Path to the JSONL file with batch requests
            model: Model name to use
            output_path: Path to save results (optional)
            metadata: Optional metadata for the batch
            completion_window: Time window for completion
            progress_callback: Optional function for progress updates
            
        Returns:
            Path to the downloaded results file
            
        Raises:
            Various exceptions if any step fails
        """
        async with aiohttp.ClientSession() as session:
            # 1. Upload file
            file_id = await self.upload_file(file_path, session)
            
            # 2. Create batch job
            batch_id = await self.create_batch(
                file_id=file_id,
                model=model,
                completion_window=completion_window,
                metadata=metadata,
                session=session
            )
            
            try:
                # 3. Wait for completion
                output_file_id, _ = await self.wait_for_batch_completion(
                    batch_id=batch_id,
                    session=session,
                    progress_callback=progress_callback
                )
                
                # 4. Download results
                result_path = await self.download_results(
                    file_id=output_file_id,
                    output_path=output_path,
                    session=session
                )
                
                return result_path
            except Exception as e:
                # If something goes wrong, try to cancel the batch
                logger.error(f"Error during batch processing: {e}")
                try:
                    await self.cancel_batch(batch_id, session)
                except Exception as cancel_error:
                    logger.error(f"Failed to cancel batch: {cancel_error}")
                raise
    
    async def split_and_process_large_batch(
        self,
        requests: List[Dict[str, Any]],
        model: str,
        output_dir: Optional[Path] = None,
        metadata: Optional[Dict[str, str]] = None,
        progress_callback: Optional[callable] = None
    ) -> List[Path]:
        """Split large batches and process them.
        
        If the number of requests exceeds MAX_REQUESTS_PER_BATCH, this method
        will split the batch into multiple smaller batches and process them.
        
        Args:
            requests: List of API request objects
            model: Model name to use
            output_dir: Directory to save results (creates temp dir if None)
            metadata: Optional metadata for the batches
            progress_callback: Optional function for progress updates
            
        Returns:
            List of paths to result files
        """
        # Create temp directory if needed
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp())
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Split requests into batches of appropriate size
        batch_size = self.MAX_REQUESTS_PER_BATCH
        batches = [requests[i:i+batch_size] for i in range(0, len(requests), batch_size)]
        
        logger.info(f"Split {len(requests)} requests into {len(batches)} batches")
        
        result_files = []
        
        for i, batch_requests in enumerate(batches):
            batch_metadata = metadata.copy() if metadata else {}
            batch_metadata["batch_number"] = str(i + 1)
            batch_metadata["total_batches"] = str(len(batches))
            
            # Create a temporary JSONL file for this batch
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                batch_file = Path(f.name)
                for request in batch_requests:
                    f.write(json.dumps(request) + '\n')
            
            try:
                # Process this batch
                batch_output = output_dir / f"batch_results_{i + 1}.jsonl"
                
                # Custom progress callback that adds batch context
                def batch_progress_callback(*args):
                    if progress_callback:
                        progress_callback(batch_num=i+1, total_batches=len(batches), *args)
                
                result_file = await self.process_batch(
                    file_path=batch_file,
                    model=model,
                    output_path=batch_output,
                    metadata=batch_metadata,
                    progress_callback=batch_progress_callback
                )
                
                result_files.append(result_file)
            finally:
                # Clean up temporary file
                if batch_file.exists():
                    os.unlink(batch_file)
        
        return result_files
    
    @staticmethod
    def merge_result_files(result_files: List[Path], output_path: Path) -> Path:
        """Merge multiple result files into a single file.
        
        Args:
            result_files: List of result file paths
            output_path: Path for the merged file
            
        Returns:
            Path to the merged file
        """
        with open(output_path, 'w') as outfile:
            for file_path in result_files:
                with open(file_path, 'r') as infile:
                    outfile.write(infile.read())
        
        return output_path
        
    async def generate(self, prompt: str, model: str, **kwargs) -> Dict[str, Any]:
        """Implementation of the abstract method from BaseClient.
        
        For single prompts, we use the regular API since batch is not efficient.
        """
        logger.warning("Using batch client for single request - consider using OpenAIClient instead")
        try:
            # Get model config and validate
            model_config = registry.get_model(model)
            actual_model_name = model_config.name
            
            # Check rate limits with token count
            await self.check_rate_limit(model, prompt)
            
            logger.debug(f"Generating response for prompt with {self.estimate_tokens(prompt)} tokens")
            
            # Merge parameters
            request_params = {}
            if model_config.parameters:
                request_params.update(model_config.parameters)
            if kwargs:
                request_params.update(kwargs)
            
            messages = [{"role": "user", "content": prompt}]
            
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(
                        self.API_URL,
                        headers=self.headers,
                        json={
                            "model": actual_model_name,
                            "messages": messages,
                            **request_params
                        },
                        timeout=60.0
                    )
                    
                    if response.status_code != 200:
                        logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                        raise ValueError(f"OpenAI API error: {response.status_code} - {response.text}")
                    
                    result = response.json()
                    return {
                        "response": result["choices"][0]["message"]["content"],
                        "model": model,
                        "usage": result.get("usage", {})
                    }
                except Exception as e:
                    logger.error(f"Error in batch API request: {str(e)}")
                    raise
                    
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    async def generate_batch(self, prompts: List[str], model: str, **kwargs) -> List[Dict[str, Any]]:
        """Implementation of the abstract method from BaseClient.
        
        For batch processing, we create a temporary JSONL file with the requests.
        """
        try:
            if len(prompts) > self.MAX_REQUESTS_PER_BATCH:
                raise ValueError(f"Batch size {len(prompts)} exceeds maximum of {self.MAX_REQUESTS_PER_BATCH}")
            
            # Get model config
            model_config = registry.get_model(model)
            actual_model_name = model_config.name
            
            # Calculate total tokens for the batch
            total_tokens = sum(self.estimate_tokens(prompt) for prompt in prompts)
            await self.check_rate_limit(model, total_tokens)
            
            logger.info(f"Processing batch of {len(prompts)} prompts ({total_tokens} total tokens)")
            
            # Merge parameters
            request_params = {}
            if model_config.parameters:
                request_params.update(model_config.parameters)
            if kwargs:
                request_params.update(kwargs)
            
            # Prepare messages for each prompt
            messages_list = [[{"role": "user", "content": prompt}] for prompt in prompts]
            
            # Make batched API request
            async with httpx.AsyncClient() as client:
                try:
                    tasks = []
                    for messages in messages_list:
                        tasks.append(
                            client.post(
                                self.API_URL,
                                headers=self.headers,
                                json={
                                    "model": actual_model_name,
                                    "messages": messages,
                                    **request_params
                                },
                                timeout=60.0
                            )
                        )
                    
                    # Execute requests concurrently
                    responses = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    results = []
                    total_tokens_generated = 0
                    
                    for i, response in enumerate(responses):
                        if isinstance(response, Exception):
                            logger.error(f"Error in batch request {i}: {str(response)}")
                            results.append({
                                "error": str(response),
                                "model": model
                            })
                            continue
                            
                        if response.status_code != 200:
                            error_msg = f"OpenAI API error: {response.status_code} - {response.text}"
                            logger.error(error_msg)
                            results.append({
                                "error": error_msg,
                                "model": model
                            })
                            continue
                            
                        result = response.json()
                        response_text = result["choices"][0]["message"]["content"]
                        tokens_generated = self.estimate_tokens(response_text)
                        total_tokens_generated += tokens_generated
                        
                        results.append({
                            "response": response_text,
                            "model": model,
                            "usage": result.get("usage", {})
                        })
                    
                    logger.info(f"Batch processing complete: {len(results)} responses generated " +
                              f"({total_tokens_generated} tokens)")
                    return results
                    
                except Exception as e:
                    logger.error(f"Error in batch processing: {str(e)}")
                    raise
                    
        except Exception as e:
            logger.error(f"Failed to process batch: {str(e)}")
            raise
