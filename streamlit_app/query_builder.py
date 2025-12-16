"""
Query Builder Module
Populates Cypher query templates with extracted entities

Takes a Cypher query template and fills in parameter values from grounded entities.
Validates that all required parameters are available before executing the query.
"""

import re


def populate_query(query, entities):
    """
    Populate a Cypher query template with extracted and grounded entities.

    This function:
    1. Extracts parameter placeholders from the query ($player, $season, etc.)
    2. Maps parameters to entity types
    3. Fills parameters with grounded entity values
    4. Validates that all required (non-OPTIONAL) parameters are present
    5. Assigns default values to optional parameters that remain unassigned:
       - season: Defaults to '22/23' (latest season)
       - other optional params: Defaults to None (NULL in Neo4j)

    Args:
        query: Cypher query template with parameter placeholders (e.g., $player, $season)
        entities: Dictionary of grounded entities from extract_entities function

    Returns:
        dict or None: Dictionary of parameters to pass to Neo4j, or None if required parameters are missing
    """
    if query is None:
        return None

    # Extract all parameter names from the query using regex
    # Matches $parameter_name patterns
    param_pattern = r'\$(\w+)'
    required_params = set(re.findall(param_pattern, query))

    # Initialize parameters dictionary
    parameters = {}

    # Mapping of parameter names to entity types
    param_to_entity_map = {
        'player': 'players',
        'player1': 'players',
        'player2': 'players',
        'team': 'teams',
        'opponent': 'teams',
        'position': 'positions',
        'season': 'seasons',
        'gw': 'gameweeks',
        'fixture': 'fixtures',
        'limit': 'statistics'  # Special case for LIMIT clauses
    }

    # Populate parameters based on extracted entities
    for param in required_params:
        entity_type = param_to_entity_map.get(param)

        if entity_type and entity_type in entities:
            entity_list = entities[entity_type]

            if entity_list:
                if param == 'player1' and len(entity_list) >= 1:
                    # For player comparisons, use first player
                    parameters[param] = entity_list[0]['grounded']
                elif param == 'player2' and len(entity_list) >= 2:
                    # For player comparisons, use second player
                    parameters[param] = entity_list[1]['grounded']
                elif param in ['player', 'team', 'opponent', 'position', 'season']:
                    # Use the grounded value from the first match
                    parameters[param] = entity_list[0]['grounded']
                elif param == 'gw':
                    # Gameweek should be an integer
                    parameters[param] = entity_list[0]['grounded']
                elif param == 'fixture':
                    # Fixture number should be an integer
                    parameters[param] = entity_list[0]['grounded']
                elif param == 'limit':
                    # Extract number from statistics if present
                    # Default to 10 if not specified
                    limit_value = 10
                    if entities.get('statistics'):
                        for stat in entities['statistics']:
                            # Try to extract number from phrases like "top 5", "best 10"
                            num_match = re.search(r'\d+', stat)
                            if num_match:
                                limit_value = int(num_match.group())
                                break
                    parameters[param] = limit_value

    # Check for missing required parameters (non-OPTIONAL matches)
    missing_params = []
    for param in required_params:
        if param not in parameters:
            # Check if the parameter is in an OPTIONAL MATCH clause
            # If not, it's required
            optional_pattern = rf'OPTIONAL\s+MATCH.*\${param}\b'
            if not re.search(optional_pattern, query, re.IGNORECASE | re.DOTALL):
                missing_params.append(param)

    if missing_params:
        print(f"Warning: Missing required parameters: {missing_params}")
        print(f"Available entities: {list(entities.keys())}")
        # BUG FIX: Return None instead of (None, None)
        return None

    # Handle optional parameters that are still unassigned
    # For optional parameters not yet filled, assign default values
    for param in required_params:
        if param not in parameters:
            # This parameter is optional (in OPTIONAL MATCH) and wasn't assigned
            # Assign default values to avoid unassigned variables in the query
            if param == 'season':
                # For season, use the latest season 22/23
                parameters[param] = '2022-23'
                print(f"Info: Optional parameter '{param}' assigned default value: 2022-23")
            else:
                # For other optional parameters, use None (which Neo4j treats as NULL)
                parameters[param] = None
                print(f"Info: Optional parameter '{param}' assigned default value: None")

    return parameters
