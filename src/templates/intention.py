templates = [
    {
        "name": "global", "template": 
    """
    A partir de ahora vas a clasificar la intención comunicativa global de los mensajes que te voy a enviar.
    La intención del mensaje debe ser una de estas 13 categorías: ''informativa'', ''opinion personal'', ''elogio'', ''critica'', ''deseo'', ''peticion'', ''pregunta'', ''obligacion'', ''sugerencia'', ''sarcasmo / broma'', ''promesa'', ''amenaza'' o ''emotiva''.

    Quiero que tu respuesta sea única y solamente: entre corchetes ( [ ] ) la intención comunicativa global seleccionada.
    Mensaje:

    """
},  {
        "name": "global-explained", "template": 
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
}]