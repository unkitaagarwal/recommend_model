from flask import Flask, request, jsonify
import pandas as pd
import json
import ast
from typing import List, Dict, Set, Any
from dataclasses import dataclass

app = Flask(__name__)

@dataclass
class NutritionTargets:
    daily_calories: int
    protein: int
    fat: int
    carbs: int

class MealRecommender:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.used_meals: Set[str] = set()

    @staticmethod
    def calculate_calories(meal: Dict[str, Any]) -> float:
        return (meal['Protein (g)'] * 4 +
                meal['Carbohydrates (g)'] * 4 +
                meal['Fat (g)'] * 9)

    def filter_meals(self,
                    preferred_cuisines: List[str],
                    allergies: List[str],
                    required_tags: List[str] = None,
                    count: int = 3) -> List[Dict[str, Any]]:
        # Start with meals that aren't used yet
        available_meals = self.df[~self.df['Dish'].isin(self.used_meals)]

        # Apply cuisine filter if specified
        if preferred_cuisines:
            cuisine_filter = available_meals['cuisines'].apply(
                lambda x: any(cuisine in x for cuisine in preferred_cuisines)
            )
            filtered_meals = available_meals[cuisine_filter]

            # Fallback to all cuisines if not enough meals found
            if len(filtered_meals) < count:
                filtered_meals = available_meals

        else:
            filtered_meals = available_meals

        # Apply allergy filter
        if allergies:
            filtered_meals = filtered_meals[
                filtered_meals['allergy'].apply(
                    lambda x: not any(allergy in x for allergy in allergies)
                )
            ]

        # Apply tag filter if specified
        if required_tags:
            tag_filter = filtered_meals['tags'].apply(
                lambda x: any(tag.lower() in [t.lower() for t in x] for tag in required_tags)
            )
            filtered_meals = filtered_meals[tag_filter]

        # Convert to records and add calories
        meals = filtered_meals.head(count).to_dict('records')
        for meal in meals:
            meal['Total Calories'] = self.calculate_calories(meal)
            self.used_meals.add(meal['Dish'])

        return meals

    def get_section_meals(self,
                         section_index: int,
                         preferred_cuisines: List[str],
                         allergies: List[str]) -> List[Dict[str, Any]]:
        section_meals = []

        if section_index == 0:
            # Section 0: 3 breakfast + 3 drinks
            breakfast_meals = self.filter_meals(
                preferred_cuisines, allergies, ['Breakfast'], 3
            )
            drink_meals = self.filter_meals(
                preferred_cuisines, allergies, ['Drinks'], 3
            )
            section_meals.extend(breakfast_meals + drink_meals)

        elif section_index in [1, 2]:
            # Sections 1-2: Any meals
            section_meals.extend(self.filter_meals(
                preferred_cuisines, allergies, count=3
            ))

        else:
            # Sections 3+: Snacks with fallback to drinks
            snack_meals = self.filter_meals(
                preferred_cuisines, allergies, ['Snacks'], 3
            )
            if len(snack_meals) < 3:
                remaining_count = 3 - len(snack_meals)
                drink_meals = self.filter_meals(
                    preferred_cuisines, allergies, ['Drinks'], remaining_count
                )
                section_meals.extend(snack_meals + drink_meals)
            else:
                section_meals.extend(snack_meals)

        return section_meals

def load_data() -> pd.DataFrame:
    try:
        df = pd.read_csv('test01.csv')
        df = df.fillna('')
        df['allergy'] = df['allergy'].fillna("[]")
        df['image_url'] = df['image_url'].fillna("https://example.com/placeholder.jpg")
        df = df.drop_duplicates(subset='Dish')

        # Convert string representations to Python objects
        for col in ['cuisines', 'allergy', 'Tags']:
            df[col] = df[col].apply(
                lambda x: ast.literal_eval(x.replace("'", '"')) if isinstance(x, str) else x
            )

        df['tags'] = df['Tags']
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return pd.DataFrame()

@app.route('/recommend', methods=['POST'])
def handle_recommendations():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Extract request parameters
        num_meals = data.get('num_meals', 3)
        nutrition_targets = NutritionTargets(
            daily_calories=data.get('daily_target', 2000),
            protein=data.get('protein_target', 0),
            fat=data.get('fat_target', 0),
            carbs=data.get('carb_target', 0)
        )
        preferred_cuisines = data.get('cuisines', [])
        allergies = data.get('allergies', [])

        # Initialize recommender with data
        df = load_data()
        recommender = MealRecommender(df)

        # Generate meal sections
        meal_sections = []
        for section_index in range(num_meals):
            section_meals = recommender.get_section_meals(
                section_index, preferred_cuisines, allergies
            )
            meal_sections.append(section_meals)

        return jsonify({
            "meal_sections": meal_sections,
            "total_sections": len(meal_sections),
            "recommendations_per_section": len(meal_sections[0]) if meal_sections else 0,
            "daily_target": nutrition_targets.daily_calories,
            "protein_target": nutrition_targets.protein,
            "fat_target": nutrition_targets.fat,
            "carb_target": nutrition_targets.carbs
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "meal_sections": [[] for _ in range(num_meals)]
        })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
