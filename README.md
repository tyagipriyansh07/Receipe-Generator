
# Custom Recipe Generator Based on Dietary Restrictions

This project involves building a recipe generator that creates detailed recipes based on dietary tags and available ingredients using a fine-tuned Mistral model. The model was trained and deployed using a custom dataset, with the final deployment hosted on Streamlit.

## Table of Contents

###### Project Overview
###### Dataset Preparation
###### Data Preprocessing
###### Model Training
###### Model Deployment
###### Usage
###### Results
###### Contributing

### 1.Project overview

The objective of this project is to generate personalized recipes based on user dietary restrictions and available ingredients. The project employs a generative AI model to provide recipe suggestions while also generating relevant nutritional information. The following requirements were addressed:

-> Generative Model: A generative AI model (Mistral) is used to create recipe suggestions.

-> User Input: The model takes dietary restrictions as input (e.g., vegan, low-carb).

-> Nutritional Information: Nutritional values (calories, protein, fat, sodium) are generated for each recipe.

Although the project is not deployed due to GPU restrictions, the Jupyter notebooks and dataset are available for anyone interested in experimenting with the model or reproducing the results.


###  2.Data Preparation

The dataset used for training the model was formed by merging two recipe datasets available online. The steps involved in preparing the dataset are as follows:

i. Data Collection: We combined datasets which included Epicurious dataset and RecipeNLG dataset. The combined dataset included recipe names, ingredients, and nutritional information such as calories, protien, sodium and fat.

ii. Tag Creation: A new tag column was created to categorize recipes based on dietary preferences, such as vegan , 'low fat', non-veg , 'high sodium' , 'high protein' , and 'low calorie'. Tags were assigned based on some thresholds analysis and recipe descriptions.

iii. Dataset Cleaning and Formatting: Missing values in nutritional information (calories, protein, fat, sodium) were removed.
Ingredients lists were tokenized and cleaned to remove any irrelevant data (e.g., special characters).
The final dataset contained the following columns: title , calories , protein , fat, sodium , ingredients , and tag .

The dataset is available in this repository as a CSV file for further use or experimentation.


### 3. Model Training

The model used for this project is Mistral 12B, a large language model chosen for its strong capabilities in handling complex natural language tasks. The steps for training the model included:

-> Model Selection: Mistral 12B was selected based on performance  and resource requirements.

-> Fine-Tuning: The model was fine-tuned using the custom recipe dataset. The training process involved adjusting hyperparameters such as learning rate, batch size, and training epochs to optimize the model.

-> Quantization: The model was initially prepared for deployment using 4-bit quantization to reduce memory usage. However, since the project is no longer being deployed, this quantized version may be omitted when running the model in a CPU environment.

## Model Deployment
## Usage

Running Locally
If you want to use the model and dataset, follow these steps:

Clone the Repository:

git clone https://github.com/your-repo/recipe-generator.git
cd recipe-generator

Install Dependencies: Install the necessary Python packages:

pip install -r requirements.txt

Run the Jupyter Notebooks: The repository includes Jupyter notebooks that demonstrate how to preprocess the data, train the model, and generate recipes. To run these, start Jupyter:


Open the notebooks Model Training.ipynb and NEBULLA.ipynb to explore the code and execute the steps.

Use the Model: You can load the pre-trained Mistral model from the notebook, input dietary restrictions and ingredients, and generate recipes with nutritional information.
## Results

The model successfully generates detailed recipes based on user inputs, including dietary restrictions and available ingredients. Each recipe includes a name, ingredients list, and nutritional information (calories, protein, fat, sodium).
## Contributing

Contributions are welcome! Feel free to fork the project and submit a pull request, or open an issue for discussion.

