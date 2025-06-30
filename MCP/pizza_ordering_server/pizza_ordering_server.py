import os
import csv
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP()

# Path to the pizza dataset
PIZZA_DATASET_PATH = Path(os.environ.get("DATA_DIR"), "pizza", "pizzas_dataset.csv")

def _load_pizza_data():
    """Loads and parses pizza data from the CSV file."""
    if not os.path.exists(PIZZA_DATASET_PATH):
        return None, None

    best_orders = {}
    all_orders = []
    
    try:
        with open(PIZZA_DATASET_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    pizzeria = row.get('Lugar')
                    pizzas = int(row.get('Pizzas', 0))
                    people = int(row.get('Personas', 0))
                    leftover_str = row.get('Lo que sobró', '0').replace(',', '.')
                    leftovers = float(leftover_str)
                    date = row.get('Fecha', 'N/A')

                    if not pizzeria or people == 0:
                        continue

                    order_data = {
                        'pizzeria': pizzeria,
                        'pizzas': pizzas,
                        'people': people,
                        'leftovers': leftovers,
                        'date': date,
                    }
                    all_orders.append(order_data)

                    if pizzeria not in best_orders or leftovers < best_orders[pizzeria]['leftovers']:
                        best_orders[pizzeria] = order_data
                
                except (ValueError, TypeError):
                    continue
        return best_orders, all_orders
    except Exception:
        return None, None

@mcp.resource("pizzas://history")
def get_pizza_history() -> str:
    """
    Loads and analyzes the pizza order history, returning the full history
    and the best-performing order (minimum leftovers) for each pizzeria.
    """
    best_orders, all_orders = _load_pizza_data()

    if all_orders is None:
        return "# Pizza Order History\n\nCould not find the pizza dataset."

    if not all_orders:
        return "# Pizza Order History\n\nNo valid order data found to analyze."

    # Build the full history table
    full_history_content = "# Full Pizza Order History\n\n"
    full_history_content += "| Pizzeria | Date | Pizzas Ordered | People Fed | Leftover Pizzas |\n"
    full_history_content += "|----------|------|----------------|------------|-----------------|\n"
    for order in all_orders:
        full_history_content += f"| {order['pizzeria']} | {order['date']} | {order['pizzas']} | {order['people']} | {order['leftovers']:.2f} |\n"
    
    full_history_content += "\n\n"
    
    # Build the best performing orders table
    best_orders_content = "# Pizza Order History: Best Performing Orders\n\n"
    best_orders_content += "This table shows the most efficient order (minimum leftover pizza) for each pizzeria based on past data.\n\n"
    best_orders_content += "| Pizzeria | Pizzas Ordered | People Fed | Leftover Pizzas |\n"
    best_orders_content += "|----------|----------------|------------|-----------------|\n"

    for name, data in best_orders.items():
        best_orders_content += f"| {name} | {data['pizzas']} | {data['people']} | {data['leftovers']:.2f} |\n"
    
    best_orders_content += "\nThis data can be used to extrapolate how many pizzas to order in the future.\n"
    
    return full_history_content + best_orders_content

@mcp.prompt()
def generate_ordering_prompt(pizzeria: str, num_people: int) -> str:
    """
    Generates a self-contained prompt for the AI to decide how many pizzas to
    order, including the necessary historical data.

    Args:
        pizzeria: The name of the pizzeria to order from.
        num_people: The number of people to order for.
    """
    best_orders, _ = _load_pizza_data()

    if not best_orders:
        return "Sorry, I could not load any pizza order history."
        
    if pizzeria not in best_orders:
        return f"Sorry, I do not have any historical order data for '{pizzeria}'."

    best_order = best_orders[pizzeria]

    return f"""
Here is the single best-performing past order for '{pizzeria}':
- Pizzas Ordered: {best_order['pizzas']}
- People Fed: {best_order['people']}

Based *only* on this historical data, determine the optimal number of pizzas to order for {num_people} people to minimize waste.

Follow these steps for your calculation:
1. From the historical data provided above, calculate a 'pizzas per person' ratio by dividing 'Pizzas Ordered' by 'People Fed'.
2. Use that ratio to estimate the number of pizzas for {num_people} people.
3. Round the result up to the nearest whole number to ensure there is enough food.

Explain your reasoning clearly, showing the calculation.

Example Reasoning:
"The best order for 'Pizza Palace' was 5 pizzas for 10 people, which is 0.5 pizzas per person. For {num_people} people, this suggests ordering {num_people} * 0.5 = X.X pizzas. Therefore, I recommend ordering Y pizzas (rounding up)."

Now, please provide your recommendation for {num_people} people ordering from '{pizzeria}'.
"""

if __name__ == "__main__":
    mcp.run(transport="stdio")
