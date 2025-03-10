import asyncio
from llmprompt_nexus import NexusManager

async def main():
    llm = NexusManager({
        "perplexity": "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"
    })

    custom_template = {
        "template": """
        Analyze the following {language} code:
        
        {code}
        
        Provide:
        - Code quality score (0-10)
        - Best practices followed
        - Suggested improvements
        """,
        "description": "Template for code review",  # Optional
        "system_message": "You are an expert code reviewer.",  # Optional
    }

    # Use the custom template
    result = await llm.generate_batch(
        input_data={
            "language": "Python",
            "code": "def hello(): print('world')"
        },
        model_id="sonar-pro",
        template_config=custom_template  # Pass template directly
    )
    
    print(result)
    
    
if __name__ == "__main__":
    asyncio.run(main())