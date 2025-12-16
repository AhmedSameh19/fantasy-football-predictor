"""
Intent Classifier Module
Uses OpenAI LLM to classify user queries into predefined intent categories

Intent classification is the first step in the input processing pipeline.
It determines which Cypher query template to use for retrieving relevant data.
"""

import json
from langchain_openai import ChatOpenAI
from config import OPENAI_KEY
from intent_queries import INTENTION_CATEGORIES, INTENTION_MAP, INTENT_DESCRIPTIONS
from langchain_core.messages import HumanMessage


def call_llm_for_classification(user_input: str) -> str:
    """
    Call OpenAI API for intent classification with structured JSON output.

    Uses GPT-5-mini with JSON mode to ensure structured, consistent responses.

    Args:
        user_input: The user's natural language query

    Returns:
        str: Intent category name (validated against INTENTION_CATEGORIES)
    """
    openai_llm = ChatOpenAI(
        model="gpt-5-mini-2025-08-07",
        temperature=0.1,
        api_key=OPENAI_KEY,
        max_tokens=10000,
        model_kwargs={"response_format": {"type": "json_object"}}  # Enforce JSON output
    )

    # Build prompt with category definitions and examples
    prompt = f"""You are an intent classifier for a Fantasy Premier League (FPL) knowledge graph system.

Your task:
Given a user query, classify it into EXACTLY one of the following categories:
{', '.join(INTENTION_CATEGORIES)}

Category definitions:
{chr(10).join(f"- {cat}: {desc}" for cat, desc in INTENT_DESCRIPTIONS.items())}

Examples:
User Input: "Show me Haaland's stats last season."
Output: {{"category": "player_season_stats"}}

User Input: "How did Salah do in GW 5?"
Output: {{"category": "player_gw_stats"}}

User Input: "Who plays for Arsenal this season?"
Output: {{"category": "team_players"}}

User Input: "Compare Son and Rashford this season."
Output: {{"category": "compare_players"}}

User Input: "Who is Trent Alexander-Arnold?"
Output: {{"category": "player_basic_info"}}

Now classify the user's input below.
Return a JSON object with a single "category" field containing ONLY one category name from the list above.
If the query doesn't match any category, return {{"category": "Unknown"}}.

User Input: "{user_input}"
"""

    messages = [HumanMessage(content=prompt)]
    response = openai_llm.invoke(messages)
    print(f"Intent Classification Response: {response.content}")

    # Parse JSON response
    try:
        result = json.loads(response.content)
        category = result.get("category", "Unknown")

        # Validate category
        if category not in INTENTION_CATEGORIES:
            print(f"Warning: LLM returned unexpected category: {category}")
            return "Unknown"

        return category
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON response: {response.content}")
        return "Unknown"


def classify_intent(user_input: str):
    """
    Classify user input into one of the predefined intent categories using LLM with structured output.

    This function uses OpenAI's JSON mode with few-shot prompting to ensure accurate,
    consistent classification into predefined categories.

    Args:
        user_input: The user's natural language query

    Returns:
        tuple: (category_name, corresponding_cypher_query_template)
               Returns ("Unknown", None) if classification fails
    """
    # Get LLM prediction with structured JSON output
    category = call_llm_for_classification(user_input)

    # Return category and its corresponding Cypher query template
    return category, INTENTION_MAP.get(category)
