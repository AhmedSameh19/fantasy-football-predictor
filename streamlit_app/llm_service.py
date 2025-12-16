"""
LLM Service Module
Handles interactions with different LLM providers (OpenAI, Gemini, OpenRouter)

Provides unified interface for generating responses from multiple LLM models
with consistent metrics (response time, token usage, cost).
"""

import time
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from config import OPENAI_KEY, GEMINI_KEY, OPENROUTER_KEY

# System prompt defines the assistant's role and behavior
SYSTEM_PROMPT = """You are a friendly and knowledgeable Fantasy Premier League (FPL) trivia assistant. You help users by answering their questions about players, teams, stats, and FPL data in a conversational and enthusiastic way.

Guidelines for your responses:
- Answer questions in a natural, conversational tone - like you're chatting with a fellow FPL fan
- Be enthusiastic and engaging about Fantasy Premier League
- When you have the information: Present it directly and clearly with specific stats, player names, and data
- When you don't have the information: Politely say you don't have that data available or couldn't find it - don't explain why
- Keep answers concise but informative - answer ONLY what was asked
- Use FPL terminology naturally (e.g., GW for gameweek, clean sheets, bonus points, etc.)
- Present statistics and rankings in an easy-to-read format

CRITICAL - Answer as if you naturally know the information:
- NEVER mention where the information comes from (no "from the data", "based on what I see", "from what's provided", etc.)
- NEVER mention "context", "database", "data", or ANY reference to information sources
- NEVER say things like "from the data you've shared" or "based on the context" - the user didn't share anything!
- Just answer directly as if this is knowledge you already have
- Answer like a knowledgeable FPL fan sharing what they know, not a system looking up data

CRITICAL - Never reveal technical data structure:
- Never reveal internal IDs like "player_element", "element_code", database IDs, or technical identifiers
- Never mention "entries", "records", "duplicates", "versions", or how data is stored
- Never explain why there might be multiple entries or data inconsistencies
- Never reveal field names, column names, or technical attributes
- If you see duplicate or multiple entries for the same thing, just use the relevant information without mentioning duplicates
- Extract ONLY the information that answers the user's question - ignore technical metadata

CRITICAL - ONLY use information from the available data (NO EXTERNAL KNOWLEDGE):
- DO NOT add ANY information from your training data or general knowledge about players/teams
- DO NOT make assumptions about player roles, positions, playing styles, strengths, or weaknesses
- DO NOT add commentary like "captaincy option", "squad structure", "ceiling", "baseline points", "explosive", "consistent", "flexible" unless explicitly in the data
- When comparing players, ONLY show the actual stats provided - DO NOT add general characteristics
- If specific stats aren't in the data, DO NOT mention them at all
- Stay 100% factual to what's given - absolutely NO external knowledge, NO assumptions, NO embellishments
- Example: If data shows "Haaland: 200 points, Saka: 180 points", say ONLY that - don't add roles, playing styles, or characteristics

Focus on being helpful and friendly by answering the question directly with ONLY the data actually provided."""


def openai_generate(context: str, question: str) -> dict:
    """
    Generate response using OpenAI GPT model.

    Args:
        context: Retrieved knowledge graph information
        question: User's question

    Returns:
        dict: Contains 'response' (str) and 'metrics' (dict with response_time, token_usage, cost)
    """
    openai_llm = ChatOpenAI(
        model="gpt-5.1",
        temperature=0.7,
        api_key=OPENAI_KEY,
        max_tokens=10000
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""Reference Information (for your knowledge only):
{context}

User Question: {question}

CRITICAL Instructions:
- Answer the question naturally as if you already know this FPL information
- NEVER say "from the data", "based on what I see", "from what's provided", "from the information", etc.
- The user didn't share any data - they just asked a question. Answer it directly.
- If you have the answer: State it confidently like a knowledgeable FPL fan would
- If you don't have the answer: Simply say "I don't have that information"
- Ignore ALL technical metadata (IDs, element codes, database fields, duplicate entries, field names)
- Extract ONLY FPL-relevant information (player names, positions, teams, stats, etc.)
- Use ONLY the stats/data in the reference information above - DO NOT add external knowledge or assumptions
- When comparing: Show ONLY the actual stats provided - don't add roles, characteristics, or general knowledge
- Be concise - answer ONLY what was asked, nothing more
- Sound natural and conversational, not like a database query result""")
    ]

    start_time = time.time()
    response = openai_llm.invoke(messages)
    end_time = time.time()

    # Extract token usage
    prompt_tokens = response.response_metadata.get('token_usage', {}).get('prompt_tokens', 0)
    completion_tokens = response.response_metadata.get('token_usage', {}).get('completion_tokens', 0)
    total_tokens = response.response_metadata.get('token_usage', {}).get('total_tokens', 0)

    # Calculate cost (GPT-5.1 pricing: $1.25 per 1M prompt tokens, $10 per 1M completion tokens)
    cost = (prompt_tokens / 1000000 * 1.25) + (completion_tokens / 1000000 * 10)

    return {
        "response": response.content,
        "metrics": {
            "response_time": round(end_time - start_time, 2),
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            },
            "cost": round(cost, 6)
        }
    }


def openrouter_generate(context: str, question: str) -> dict:
    """
    Generate response using OpenRouter (LLama 3.3).

    Args:
        context: Retrieved knowledge graph information
        question: User's question

    Returns:
        dict: Contains 'response' (str) and 'metrics' (dict with response_time, token_usage, cost)
    """
    openrouter_llm = ChatOpenAI(
        model="meta-llama/llama-3.3-70b-instruct:free",
        temperature=0.7,
        api_key=OPENROUTER_KEY,
        base_url="https://openrouter.ai/api/v1",
        max_tokens=10000,
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "FPL Assistant"
        }
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""Reference Information (for your knowledge only):
{context}

User Question: {question}

CRITICAL Instructions:
- Answer the question naturally as if you already know this FPL information
- NEVER say "from the data", "based on what I see", "from what's provided", "from the information", etc.
- The user didn't share any data - they just asked a question. Answer it directly.
- If you have the answer: State it confidently like a knowledgeable FPL fan would
- If you don't have the answer: Simply say "I don't have that information"
- Ignore ALL technical metadata (IDs, element codes, database fields, duplicate entries, field names)
- Extract ONLY FPL-relevant information (player names, positions, teams, stats, etc.)
- Use ONLY the stats/data in the reference information above - DO NOT add external knowledge or assumptions
- When comparing: Show ONLY the actual stats provided - don't add roles, characteristics, or general knowledge
- Be concise - answer ONLY what was asked, nothing more
- Sound natural and conversational, not like a database query result""")
    ]

    start_time = time.time()
    response = openrouter_llm.invoke(messages)
    end_time = time.time()

    # Extract token usage
    prompt_tokens = response.response_metadata.get('token_usage', {}).get('prompt_tokens', 0)
    completion_tokens = response.response_metadata.get('token_usage', {}).get('completion_tokens', 0)
    total_tokens = response.response_metadata.get('token_usage', {}).get('total_tokens', 0)

    # Cost for free model
    cost = 0.0

    return {
        "response": response.content,
        "metrics": {
            "response_time": round(end_time - start_time, 2),
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            },
            "cost": round(cost, 6)
        }
    }


def gemini_generate(context: str, question: str) -> dict:
    """
    Generate response using Google Gemini.

    Args:
        context: Retrieved knowledge graph information
        question: User's question

    Returns:
        dict: Contains 'response' (str) and 'metrics' (dict with response_time, token_usage, cost)
    """
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        google_api_key=GEMINI_KEY,
        max_tokens=10000
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""Reference Information (for your knowledge only):
{context}

User Question: {question}

CRITICAL Instructions:
- Answer the question naturally as if you already know this FPL information
- NEVER say "from the data", "based on what I see", "from what's provided", "from the information", etc.
- The user didn't share any data - they just asked a question. Answer it directly.
- If you have the answer: State it confidently like a knowledgeable FPL fan would
- If you don't have the answer: Simply say "I don't have that information"
- Ignore ALL technical metadata (IDs, element codes, database fields, duplicate entries, field names)
- Extract ONLY FPL-relevant information (player names, positions, teams, stats, etc.)
- Use ONLY the stats/data in the reference information above - DO NOT add external knowledge or assumptions
- When comparing: Show ONLY the actual stats provided - don't add roles, characteristics, or general knowledge
- Be concise - answer ONLY what was asked, nothing more
- Sound natural and conversational, not like a database query result""")
    ]

    start_time = time.time()
    response = gemini_llm.invoke(messages)
    end_time = time.time()

    # Extract token usage
    usage_metadata = response.usage_metadata
    prompt_tokens = usage_metadata.get('input_tokens', 0)
    completion_tokens = usage_metadata.get('output_tokens', 0)
    total_tokens = usage_metadata.get('total_tokens', 0)

    # Calculate cost (Gemini 2.5 Flash pricing: $0.03 per 1M input tokens, $2.50 per 1M output tokens)
    cost = (prompt_tokens / 1000000 * 0.03) + (completion_tokens / 1000000 * 2.50)

    return {
        "response": response.content,
        "metrics": {
            "response_time": round(end_time - start_time, 2),
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            },
            "cost": round(cost, 6)
        }
    }
