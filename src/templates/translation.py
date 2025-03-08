# Dictionary of templates for different intentions
templates = {
    "default": """
    Please answer the following question:
    {question}
    """,
    
    "summarize": """
    Please summarize the following text:
    {text}
    """,
    
    "sentiment": """
    Analyze the sentiment of the following text:
    {text}
    
    Please classify it as positive, negative, or neutral and explain why.
    """,
    
    "translate": """
    Please translate the following text:
    
    Original text ({source_lang}):
    {text}
    
    Translate to {target_lang} while maintaining the original meaning, tone, and style.
    Please ensure proper grammar and natural expression in the target language.
    
    Guidelines:
    - Maintain any technical terminology accurately
    - Preserve formatting and structure
    - Keep named entities unchanged unless there's a widely accepted translation
    - Handle idiomatic expressions appropriately for the target culture
    """
}