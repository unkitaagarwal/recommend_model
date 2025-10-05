from flask import Flask, request, jsonify
import pandas as pd
import json
from typing import List, Dict, Set, Any
from dataclasses import dataclass

app = Flask(__name__)
application = app

@app.get("/")
def root_health_check():
    return "OK", 200

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
        self.protein_per_meal = self.nutrition_targets.protein / self.num_sections
        self.fat_per_meal = self.nutrition_targets.fat / self.num_sections
        self.carbs_per_meal = self.nutrition_targets.carbs / self.num_sections

    def calculate_calories(self, row):
        return (4 * row['Protein (g)']) + (9 * row['Fat (g)']) + (4 * row['Carbohydrates (g)'])

    def filter_meals(self,
                     preferred_cuisines: List[str],
                     allergies: List[str],
                     required_tags: List[str] = None,
                     target_calories: float = None,
                     priority: str = None) -> List[Dict[str, Any]]:
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

        if priority == 'high_protein':
            filtered_meals = filtered_meals[filtered_meals['Protein (g)'] >= 15]

        if filtered_meals.empty:
            return []

        filtered_meals['calories'] = filtered_meals.apply(lambda row: self.calculate_calories(row), axis=1)
        filtered_meals['calories_diff'] = abs(filtered_meals['calories'] - target_calories)
        filtered_meals['protein_diff'] = abs(filtered_meals['Protein (g)'] - self.protein_per_meal)
        filtered_meals['fat_diff'] = abs(filtered_meals['Fat (g)'] - self.fat_per_meal)
        filtered_meals['carbs_diff'] = abs(filtered_meals['Carbohydrates (g)'] - self.carbs_per_meal)

        if priority == 'high_protein':
            filtered_meals['total_diff'] = (
                filtered_meals['calories_diff'] +
                (0.5 * filtered_meals['protein_diff']) -
                (0.5 * filtered_meals['Protein (g)']) +
                (1.5 * filtered_meals['fat_diff']) +
                (1.5 * filtered_meals['carbs_diff'])
            )
        else:
            filtered_meals['total_diff'] = (
                filtered_meals['calories_diff'] +
                filtered_meals['protein_diff'] +
                filtered_meals['fat_diff'] +
                filtered_meals['carbs_diff']
            )

        best_meal = filtered_meals.nsmallest(1, 'total_diff').iloc[0].to_dict()
        best_meal['Total Calories'] = best_meal['calories']
        self.used_meals.add(best_meal['Dish'])
        return [best_meal]

    def generate_meal_sections(self, num_sections: int, target_calories: float,
                               preferred_cuisines: List[str], allergies: List[str]) -> List[List[Dict[str, Any]]]:
        meal_sections = []

        for section_index in range(num_sections):
            section = []

            if section_index == 0:
                for _ in range(3):
                    breakfast = self.filter_meals(preferred_cuisines, allergies, required_tags=['Breakfast'], target_calories=target_calories, priority='high_protein')
                    if not breakfast:
                        breakfast = self.filter_meals(preferred_cuisines, allergies, required_tags=['Snacks'], target_calories=target_calories, priority='high_protein')
                    section += breakfast

                for _ in range(3):
                    drink = self.filter_meals(preferred_cuisines, allergies, required_tags=['Drinks'], target_calories=target_calories, priority='high_protein')
                    if not drink:
                        drink = self.filter_meals(preferred_cuisines, allergies, required_tags=['Snacks'], target_calories=target_calories, priority='high_protein')
                    section += drink

            elif section_index == 1:
                for _ in range(3):
                    section += self.filter_meals(preferred_cuisines, allergies, required_tags=['Lunch'], target_calories=target_calories)
            elif section_index == 2:
                for _ in range(3):
                    section += self.filter_meals(preferred_cuisines, allergies, required_tags=['Dinner'], target_calories=target_calories)
            else:
                for _ in range(3):
                    section += self.filter_meals(preferred_cuisines, allergies, required_tags=['Snacks', 'Drinks'], target_calories=target_calories)

            # Fallback logic
            if not section:
                if section_index == 0:
                    fallback_tags = ['Breakfast', 'Drinks', 'Snacks']

                    for _ in range(3):
                        partial_fallback = self.filter_meals(
                            preferred_cuisines, allergies,
                            required_tags=fallback_tags,
                            target_calories=target_calories,
                            priority=None  # allow low-protein options like Kaya Toast
                        )
                        section += partial_fallback

                    while len(section) < 3:
                        broader_fallback = self.filter_meals(
                            [], allergies,
                            required_tags=fallback_tags,
                            target_calories=target_calories,
                            priority='high_protein'
                        )
                        if not broader_fallback:
                            break
                        section += broader_fallback

                elif section_index == 1:
                    fallback_tags = ['Lunch']
                    for _ in range(3):
                        section += self.filter_meals([], allergies, required_tags=fallback_tags, target_calories=target_calories)
                elif section_index == 2:
                    fallback_tags = ['Dinner']
                    for _ in range(3):
                        section += self.filter_meals([], allergies, required_tags=fallback_tags, target_calories=target_calories)
                else:
                    fallback_tags = ['Snacks', 'Drinks']
                    for _ in range(3):
                        section += self.filter_meals([], allergies, required_tags=fallback_tags, target_calories=target_calories)

            meal_sections.append(section)

        return meal_sections

@app.route("/recommend", methods=["POST"])
def recommend_meals():
    data = request.get_json()
    num_meals = data["num_meals"]
    nutrition_targets = NutritionTargets(
        daily_calories=data["daily_target"],
        protein=data["protein_target"],
        fat=data["fat_target"],
        carbs=data["carb_target"],
        calories_per_section=data["daily_target"] / data["num_meals"]
    )

    df = pd.read_csv("Foodlist.csv")
    df['tags'] = df['Tags'].apply(json.loads)
    df['cuisines'] = df['cuisines'].apply(json.loads)
    df['allergy'] = df['allergy'].apply(json.loads)

    recommender = MealRecommender(df, nutrition_targets, num_meals)
    meal_sections = recommender.generate_meal_sections(
        num_sections=num_meals,
        target_calories=nutrition_targets.calories_per_section,
        preferred_cuisines=data.get("cuisines", []),
        allergies=data.get("allergies", [])
    )

    # NEW PER-MEAL MACRO VALIDATION
    protein_per_meal = nutrition_targets.protein / len(meal_sections)
    fat_per_meal = nutrition_targets.fat / len(meal_sections)
    carbs_per_meal = nutrition_targets.carbs / len(meal_sections)
    tolerance = 0.4  # 40%

    def is_within_target(value, target):
        return (1 - tolerance) * target <= value <= (1 + tolerance) * target

    protein_ok = all(is_within_target(m['Protein (g)'], protein_per_meal) for section in meal_sections for m in section)
    fat_ok = all(is_within_target(m['Fat (g)'], fat_per_meal) for section in meal_sections for m in section)
    carbs_ok = all(is_within_target(m['Carbohydrates (g)'], carbs_per_meal) for section in meal_sections for m in section)

    macro_summary = {
        "protein_target": nutrition_targets.protein,
        "fat_target": nutrition_targets.fat,
        "carbs_target": nutrition_targets.carbs,
        "protein_per_meal": protein_per_meal,
        "fat_per_meal": fat_per_meal,
        "carbs_per_meal": carbs_per_meal,
        "protein_ok": protein_ok,
        "fat_ok": fat_ok,
        "carbs_ok": carbs_ok
    }

    return jsonify({
        "applied_filters": {
            "cuisines": data.get("cuisines", []),
            "allergies": data.get("allergies", [])
        },
        "macro_summary": macro_summary,
        "meal_sections": meal_sections
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
