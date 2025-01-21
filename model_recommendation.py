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
        self.target_per_meal = self.nutrition_targets.daily_calories / self.num_sections
        self.vertical_indices = {0: [], 1: [], 2: []}

    @staticmethod
    def calculate_calories(meal: Dict[str, Any]) -> float:
        return round((meal['Protein (g)'] * 4 +
                meal['Carbohydrates (g)'] * 4 +
                meal['Fat (g)'] * 9), 1)

    def select_meal_for_target(self,
                             available_meals: pd.DataFrame,
                             target_calories: float) -> Dict[str, Any]:
        """Select meal closest to target calories"""
        if available_meals.empty:
            return None

        available_meals.loc[:, 'calories'] = available_meals.apply(
            lambda row: self.calculate_calories(row), axis=1
        )

        # Allow 20% deviation from target
        max_allowed = target_calories * 1.2
        min_allowed = target_calories * 0.8
        filtered_meals = available_meals[
            (available_meals['calories'] >= min_allowed) &
            (available_meals['calories'] <= max_allowed)
        ]

        if filtered_meals.empty:
            filtered_meals = available_meals

        filtered_meals.loc[:, 'calories_diff'] = abs(
            filtered_meals['calories'] - target_calories
        )

        best_meal = filtered_meals.nsmallest(1, 'calories_diff').iloc[0].to_dict()
        best_meal['Total Calories'] = best_meal['calories']
        return best_meal

    def filter_meals(self,
                    preferred_cuisines: List[str],
                    allergies: List[str],
                    required_tags: List[str] = None,
                    target_calories: float = None,
                    vertical_index: int = 0) -> List[Dict[str, Any]]:

        print(f"\nFiltering meals for vertical index {vertical_index}")
        print(f"Target calories: {target_calories}")

        available_meals = self.df[~self.df['Dish'].isin(self.used_meals)].copy()

        if preferred_cuisines:
            cuisine_filter = available_meals['cuisines'].apply(
                lambda x: any(cuisine in x for cuisine in preferred_cuisines)
            )
            filtered_meals = available_meals[cuisine_filter]
            if len(filtered_meals) < 1:
                filtered_meals = available_meals
        else:
            filtered_meals = available_meals

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

        selected_meal = self.select_meal_for_target(
            filtered_meals,
            target_calories
        )

        if selected_meal:
            self.used_meals.add(selected_meal['Dish'])
            self.vertical_indices[vertical_index].append(selected_meal)
            print(f"Selected {selected_meal['Dish']} with {selected_meal['Total Calories']} calories")
            return [selected_meal]

        return []

    def get_section_meals(self,
                     section_index: int,
                     preferred_cuisines: List[str],
                     allergies: List[str],
                     items_per_section: int) -> List[Dict[str, Any]]:
        section_meals = []

        if section_index == 0:
            # First get 3 breakfast items
            for _ in range(items_per_section):
                breakfast = self.filter_meals(
                    preferred_cuisines,
                    allergies,
                    ['Breakfast'],
                    target_calories=self.target_per_meal * 0.6  # ~240 calories
                )
                if breakfast:
                    section_meals.extend(breakfast)

            # Then get 3 drink items
            for _ in range(items_per_section):
                drink = self.filter_meals(
                    preferred_cuisines,
                    allergies,
                    ['Drinks'],
                    target_calories=self.target_per_meal * 0.4  # ~160 calories
                )
                if drink:
                    section_meals.extend(drink)

        else:
            # For all other sections, get exactly 3 items
            for _ in range(items_per_section):
                if section_index == self.num_sections - 1:
                    # Last section: try snacks first
                    meal = self.filter_meals(
                        preferred_cuisines,
                        allergies,
                        ['Snacks'],
                        target_calories=self.target_per_meal  # ~400 calories
                    )
                    if not meal:
                        # Fallback to drinks if no snacks available
                        meal = self.filter_meals(
                            preferred_cuisines,
                            allergies,
                            ['Drinks'],
                            target_calories=self.target_per_meal
                        )
                else:
                    # Middle sections: regular meals
                    meal = self.filter_meals(
                        preferred_cuisines,
                        allergies,
                        target_calories=self.target_per_meal  # ~400 calories
                    )
                if meal:
                    section_meals.extend(meal)

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

        print(f"Processing request - Daily target: {daily_target}")
        print(f"Target per meal: {daily_target/num_meals}")

        df = load_data()
        recommender = MealRecommender(df, nutrition_targets, num_meals)

        meal_sections = []
        for section_index in range(num_meals):
            section_meals = recommender.get_section_meals(
                section_index, preferred_cuisines, allergies, items_per_section
            )
            meal_sections.append(section_meals)

        # Calculate vertical sums
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
