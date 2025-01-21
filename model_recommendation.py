from flask import Flask, request, jsonify
import pandas as pd
import json
import os
from typing import List, Dict, Set, Any
from dataclasses import dataclass

app = Flask(__name__)

@dataclass
class NutritionTargets:
    daily_calories: int
    protein: int
    fat: int
    carbs: int
    calories_per_section: int

class MealRecommender:
    def __init__(self, df: pd.DataFrame, nutrition_targets: NutritionTargets, num_sections: int):
        self.df = df
        self.used_meals: Set[str] = set()
        self.nutrition_targets = nutrition_targets
        self.num_sections = num_sections
        self.target_per_vertical = self.nutrition_targets.daily_calories / self.num_sections
        self.vertical_indices = {0: [], 1: [], 2: []}  # Track meals by vertical index

    @staticmethod
    def calculate_calories(meal: Dict[str, Any]) -> float:
        return round((meal['Protein (g)'] * 4 +
                meal['Carbohydrates (g)'] * 4 +
                meal['Fat (g)'] * 9), 1)

    def get_vertical_total(self, vertical_index: int) -> float:
        """Get total calories for a vertical index"""
        return sum(meal['Total Calories'] for meal in self.vertical_indices[vertical_index])

    def get_target_per_position(self, vertical_index: int) -> float:
        """Calculate target calories for next position in vertical index"""
        meals_in_vertical = len(self.vertical_indices[vertical_index])
        remaining_positions = self.num_sections - meals_in_vertical
        if remaining_positions <= 0:
            return 0

        current_total = self.get_vertical_total(vertical_index)
        remaining_target = self.target_per_vertical - current_total
        return remaining_target / remaining_positions

    def select_best_meal(self, filtered_meals: pd.DataFrame, target_calories: float) -> Dict[str, Any]:
        """Select meal closest to target calories"""
        if filtered_meals.empty:
            return None

        # Calculate calories for all meals
        filtered_meals.loc[:, 'calories'] = filtered_meals.apply(
            lambda row: self.calculate_calories(row), axis=1
        )
        filtered_meals.loc[:, 'calories_diff'] = abs(filtered_meals['calories'] - target_calories)

        best_meal = filtered_meals.nsmallest(1, 'calories_diff').iloc[0].to_dict()
        best_meal['Total Calories'] = best_meal['calories']
        return best_meal

    def filter_meals(self,
                    preferred_cuisines: List[str],
                    allergies: List[str],
                    required_tags: List[str] = None,
                    vertical_index: int = 0) -> Dict[str, Any]:

        target_calories = self.get_target_per_position(vertical_index)
        print(f"\nProcessing vertical index {vertical_index}")
        print(f"Current total: {self.get_vertical_total(vertical_index)}")
        print(f"Target for this position: {target_calories}")

        available_meals = self.df[~self.df['Dish'].isin(self.used_meals)]

        if preferred_cuisines:
            cuisine_filter = available_meals['cuisines'].apply(
                lambda x: any(cuisine in x for cuisine in preferred_cuisines)
            )
            filtered_meals = available_meals[cuisine_filter].copy()
            if len(filtered_meals) < 1:
                filtered_meals = available_meals.copy()
        else:
            filtered_meals = available_meals.copy()

        if allergies:
            filtered_meals = filtered_meals[
                filtered_meals['allergy'].apply(
                    lambda x: not any(allergy.lower() in [a.lower() for a in x] for allergy in allergies)
                )
            ]

        if required_tags:
            tag_filter = filtered_meals['tags'].apply(
                lambda x: any(tag.lower() in [t.lower() for t in x] for tag in required_tags)
            )
            filtered_meals = filtered_meals[tag_filter]

        selected_meal = self.select_best_meal(filtered_meals, target_calories)

        if selected_meal:
            self.used_meals.add(selected_meal['Dish'])
            self.vertical_indices[vertical_index].append(selected_meal)
            print(f"Selected {selected_meal['Dish']} with {selected_meal['Total Calories']} calories")
            print(f"New total for vertical index {vertical_index}: {self.get_vertical_total(vertical_index)}")

        return selected_meal

    def get_section_meals(self,
                         section_index: int,
                         preferred_cuisines: List[str],
                         allergies: List[str],
                         items_per_section: int) -> List[Dict[str, Any]]:
        section_meals = []

        for vertical_index in range(items_per_section):
            tags = None
            if section_index == 0:
                tags = ['Breakfast'] if vertical_index == 0 else None
            elif section_index == self.num_sections - 1:
                tags = ['Snacks'] if vertical_index == 0 else None

            meal = self.filter_meals(
                preferred_cuisines=preferred_cuisines,
                allergies=allergies,
                required_tags=tags,
                vertical_index=vertical_index
            )
            if meal:
                section_meals.append(meal)

        return section_meals

def load_data() -> pd.DataFrame:
    try:
        df = pd.read_csv('Macro list  - Sheet(2).csv')
        df = df.fillna('')

        for col in ['cuisines', 'allergy', 'Tags']:
            df[col] = df[col].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x else []
            )

        df['image'] = df['image_url'].fillna("https://example.com/placeholder.jpg")
        df = df.drop_duplicates(subset='Dish')
        df['tags'] = df['Tags']

        print(f"Loaded {len(df)} unique dishes")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        print(f"Current working directory: {os.getcwd()}")
        return pd.DataFrame()

@app.route('/recommend', methods=['POST'])
def handle_recommendations():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        num_meals = data.get('num_meals', 3)
        items_per_section = 3
        daily_target = data.get('daily_target', 2000)

        nutrition_targets = NutritionTargets(
            daily_calories=daily_target,
            protein=data.get('protein_target', 0),
            fat=data.get('fat_target', 0),
            carbs=data.get('carb_target', 0),
            calories_per_section=daily_target / num_meals
        )

        preferred_cuisines = data.get('cuisines', [])
        allergies = data.get('allergies', data.get('allergy', []))
        if isinstance(allergies, str):
            allergies = [allergies]

        print(f"Processing request - Target per vertical index: {daily_target/num_meals}")

        df = load_data()
        recommender = MealRecommender(df, nutrition_targets, num_meals)

        meal_sections = []
        for section_index in range(num_meals):
            section_meals = recommender.get_section_meals(
                section_index, preferred_cuisines, allergies, items_per_section
            )
            meal_sections.append(section_meals)

        index_calories = {}
        for i in range(items_per_section):
            total_calories = sum(
                section[i]['Total Calories']
                for section in meal_sections
                if i < len(section)
            )
            index_calories[f"index_{i}_total"] = round(total_calories, 1)

        response_data = {
            "meal_sections": meal_sections,
            "total_sections": len(meal_sections),
            "recommendations_per_section": items_per_section,
            "daily_target": nutrition_targets.daily_calories,
            "index_calories": index_calories,
            "protein_target": nutrition_targets.protein,
            "fat_target": nutrition_targets.fat,
            "carbs": nutrition_targets.carbs,
            "applied_filters": {
                "cuisines": preferred_cuisines,
                "allergies": allergies
            }
        }

        return jsonify(response_data)

    except Exception as e:
        print(f"Error in recommendation: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "meal_sections": [[] for _ in range(num_meals)]
        })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
