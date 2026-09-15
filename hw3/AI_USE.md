# AI_USE.md

## 1. What I used AI for vs. wrote myself

-Used AI to understand the coding syntax for the prompt engineering cells
-Used to structure the self attention code blocks
-Designed coding structure and code by myself. Identified questions for getting the best results in prompt engineering and drew inferences for self attention (causal and masked) heatmaps.

## 2. One specific wrong thing the AI produced
-The AI asked me to use ChatGPT by default. 
-Also while importing libraries for langchain, one of the libraries "langchain.prompts" had become legacy which wither AI models was not able to resolve.
-When I switched to Groq per the AI's suggestion, it then
gave me `ChatGroq(model="llama-3.1-8b-instant")`, which produced this real error:

    NotFoundError: Error code: 404 - {'error': {'message': 'The model
    `llama-3.1-8b-instant` does not exist or you do not have access to it.',
    'type': 'invalid_request_error', 'code': 'model_not_found'}}

The model name the AI suggested from memory was stale/deprecated on Groq's current
API.
## 3. How I found out it was wrong?
The error was direct, immediate exception at the llm.invoke(...) calls and I had to confirm the same from documentation rather than AI. 

## 4. What I changed and why my fix works?

-I found out the correct libraries and valid AI models that could have been used without errors, instead of the legacy errors that were generated when the AI suggested models/libraries were run/imported.
