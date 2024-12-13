import os
import asyncio
from pathlib import Path
import aiohttp
import pandas as pd
from tqdm import tqdm

# Set up your Perplexity API key
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY") or "API KEY HERE ROBI Wink Wink"

# Model configurations
MODELS = {
    "sonar-small": {
        "name": "llama-3.1-sonar-small-128k-online",
        "api": "perplexity",
        "api_url": "https://api.perplexity.ai/chat/completions"
    }
}

# Define the prompt templates
templates = [
    {
        "name": "global", "template": 
"""
A partir de ahora vas a clasificar la intención comunicativa global de los mensajes que te voy a enviar.
La intención del mensaje debe ser una de estas 13 categorías: ''informativa'', ''opinion personal'', ''elogio'', ''critica'', ''deseo'', ''peticion'', ''pregunta'', ''obligacion'', ''sugerencia'', ''sarcasmo / broma'', ''promesa'', ''amenaza'' o ''emotiva''.

Quiero que tu respuesta sea única y solamente: entre corchetes ( [ ] ) la intención comunicativa global seleccionada.
Mensaje:

"""
},
    
    {"name": "global-explained", "template": 
"""
Vas a clasificar la intención comunicativa global de los mensajes que te voy a enviar.
La intención del mensaje debe ser una de estas 13 categorías:
- informativa: el mensaje aporta información sobre el tema que se expone.
- opinión personal: el emisor incluye su punto de vista neutro.
- sugerencia: el emisor invita o recomienda que el destinatario realice algo.
- obligación: el emisor obliga al destinatario a realizar una acción. Tiene una fuerza intencional mayor que la categoría "sugerencia".
- petición: el emisor tiene la intención de solicitar alguna cosa a otra persona.
- pregunta: Cuando el emisor formula una pregunta para la que busca una explicación explícita. No sirven las preguntas retóricas.
- amenaza: se da a entender que una acción ocurrirá en el futuro en caso de que se cumpla -o no- una condición que se expresa en el mensaje.
- promesa: el emisor se compromete a realizar una acción en el futuro o a confirmar la veracidad de algo.
- elogio: el emisor valora positivamente aquello que se describe en el mensaje.
- crítica: el usuario valora negativamente aquello que se describe en el mensaje. 
- emotiva: el emisor expresa algún estado psicológico propio -como los sentimientos, las emociones o los agradecimientos- sobre lo que se describe en el mensaje.
- deseo: el emisor refleja su deseo de que ocurra algo que se indica en el mensaje.
- sarcasmo / broma: el emisor pretende expresar lo contrario de lo que se dice o darle un sentido figurado al mensaje a partir de figuras retóricas o tonos humorísticos.

Quiero que tu respuesta sea única y solamente: entre corchetes ( [ ] ) la intención comunicativa seleccionada.
Mensaje:

"""
},
]

async def get_perplexity_response(session, template, text, model_config):
    """Make a call to the AI API to get a response."""
    prompt = f"{template['template']}\n{text}"
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    data = {
        "model": model_config["name"],
        "messages": [
            {"role": "system", "content": "Eres un experto lingüista español."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": "Optional",
        "return_images": False,
        "return_related_questions": False,
        "stream": False,
    }
    
    try:
        async with session.post(model_config["api_url"], headers=headers, json=data) as response:
            if response.status == 200:
                result = await response.json()
                response_text = result['choices'][0]['message']['content']
                # Extract text between brackets
                return response_text.split("[")[1].split("]")[0].strip()
            else:
                print(f"Error: {response.status}")
                return ""
    except Exception as e:
        print(f"Error processing response: {e}")
        return ""

async def process_file(input_file, templates, model_config):
    """Process a single TSV file with the given templates and model."""
    df = pd.read_csv(input_file, sep='\t')
    
    async with aiohttp.ClientSession() as session:
        for template in templates:
            column_name = f"GlobalIntention_{template['name']}_{model_config['name']}"
            df[column_name] = ""
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {template['name']}"):
                if (model_config["api"] == "perplexity"):
                    response = await get_perplexity_response(session, template, row['Text'], model_config)
                else:
                    raise Exception(f"Unsuported API {model_config["api"]}")
                    
                df.at[idx, column_name] = response
    
    # Create output filename with model name
    output_file = input_file.parent / f"{input_file.stem}_{model_config['name']}.tsv"
    df.to_csv(output_file, sep='\t', index=False)
    print(f"Saved results to {output_file}")
    
async def process_directory(input_dir):
    """Process all TSV files in the input directory."""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory {input_dir} does not exist")
    
    tsv_files = list(input_path.glob("*.tsv"))
    if not tsv_files:
        raise ValueError(f"No TSV files found in {input_dir}")
    
    for model_name, model_config in MODELS.items():
        print(f"\nProcessing with model: {model_name}")
        for file_path in tsv_files:
            print(f"\nProcessing file: {file_path}")
            await process_file(file_path, templates, model_config)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_directory>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    asyncio.run(process_directory(input_dir))