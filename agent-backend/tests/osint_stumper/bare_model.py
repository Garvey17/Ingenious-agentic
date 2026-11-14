"""Call the configured model directly, without tools or retrieval."""

import asyncio

from app.config import settings


async def ask_bare_model(question: str) -> str:
    """Ask the configured provider the question with no tools or retrieved context."""
    if settings.llm_provider == "openai":
        from openai import AsyncOpenAI

        async with AsyncOpenAI(api_key=settings.openai_api_key or "") as client:
            response = await client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": question}],
                temperature=settings.openai_temperature,
                max_tokens=settings.openai_max_tokens,
            )
            return response.choices[0].message.content or ""

    if settings.llm_provider == "gemini":
        import google.generativeai as genai

        genai.configure(api_key=settings.google_api_key or "")
        model = genai.GenerativeModel(
            model_name=settings.gemini_model,
            generation_config=genai.types.GenerationConfig(
                temperature=settings.gemini_temperature,
                max_output_tokens=settings.gemini_max_tokens,
            ),
        )
        response = await asyncio.to_thread(model.generate_content, question)
        return response.text or ""

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
