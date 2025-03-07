# src/processors/batch_processor.py
def split_batch(self, requests, max_batch_size=50000):
    """Split a large batch into smaller chunks."""
    return [requests[i:i+max_batch_size] for i in range(0, len(requests), max_batch_size)]

async def process_large_batch(self, requests, model_config):
    """Process a large batch by splitting if necessary."""
    if len(requests) > self.MAX_BATCH_SIZE:
        batches = self.split_batch(requests)
        results = []
        for batch in batches:
            batch_results = await self.process_batch(batch, model_config)
            results.extend(batch_results)
        return results
    else:
        return await self.process_batch(requests, model_config)
