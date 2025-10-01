import pandas as pd
import openai
import os

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # or directly set as a string

# Load the CSV
df = pd.read_csv("Macro_sheet_april_2024.csv")

# Get the current dishes
existing_dishes = df['Dish'].dropna().str.lower().unique()

# Define new food items to add
new_foods = ["Tofu Stir Fry", "Avocado Toast", "Miso Soup", "Falafel Wrap", "Quinoa Salad"]
new_items = [dish for dish in new_foods if dish.lower() not in existing_dishes]

# Function to get food info using GPT
def fetch_food_info(dish_name):
    prompt = f"""
    Provide the following for the dish "{dish_name}":
    - Protein (g)
    - Fat (g)
    - Carbohydrates (g)
    - Cuisine (e.g., Indian, Mediterranean)
    - Common allergens
    - Tags: choose from Breakfast, Snacks, Drinks, Lunch, Dinner, NA (multiple allowed)
    - Public image URL

    Format:
    Protein (g): ...
    Fat (g): ...
    Carbohydrates (g): ...
    cuisines: ...
    allergy: ...
    Tags: ...
    image_url: ...
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# Parse GPT result
def parse_gpt_output(output):
    entry = {}
    for line in output.strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            entry[key.strip()] = value.strip()
    return entry

# Generate new rows
new_rows = []
for dish in new_items:
    gpt_output = fetch_food_info(dish)
    parsed = parse_gpt_output(gpt_output)
    new_row = {
        'Dish': dish,
        'Protein (g)': parsed.get("Protein (g)", ""),
        'Fat (g)': parsed.get("Fat (g)", ""),
        'Carbohydrates (g)': parsed.get("Carbohydrates (g)", ""),
        'cuisines': parsed.get("cuisines", ""),
        'allergy': parsed.get("allergy", ""),
        'Tags': parsed.get("Tags", ""),
        'image_url': parsed.get("image_url", ""),
        'source': "openai-gpt"
    }
    new_rows.append(new_row)

# Append to the dataframe and save
df_extended = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
df_extended.to_csv("Macro_sheet_filled_openai.csv", index=False)

print("✅ Final file saved as Macro_sheet_filled_openai.csv")
