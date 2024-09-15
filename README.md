
# Custom Recipe Generator Based on Dietary Restrictions

This project involves building a recipe generator that creates detailed recipes based on dietary tags and available ingredients using a fine-tuned Mistral model. The model was trained and deployed using a custom dataset, with the final deployment hosted on Streamlit.

## Table of Contents

###### Project Overview
###### Dataset Preparation & Preprocessing
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


###  2.Data Preparation & Preprocessing

The dataset used for training the model was formed by merging two recipe datasets available online. The steps involved in preparing the dataset are as follows:

i. Data Collection: We combined datasets which included Epicurious dataset and RecipeNLG dataset. The combined dataset included recipe names, ingredients, and nutritional information such as calories, protien, sodium and fat.

ii. Tag Creation: A new tag column was created to categorize recipes based on dietary preferences, such as vegan , 'low fat', non-veg , 'high sodium' , 'high protein' , and 'low calorie'. Tags were assigned based on some thresholds analysis and recipe descriptions.

iii. Dataset Cleaning and Formatting: Missing values in nutritional information (calories, protein, fat, sodium) were removed.
Ingredients lists were tokenized and cleaned to remove any irrelevant data (e.g., special characters).
The final dataset contained the following columns: title , calories , protein , fat, sodium , ingredients , and tag .

The dataset is available in this repository as a CSV file for further use or experimentation.


### 3. Model Training

The model training involves fine-tuning the pre-trained Mistral 7B model with our custom recipe dataset. Here’s a breakdown of the steps taken:

-> Model Selection:
The model training involved fine-tuning the pre-trained Mistral 7B model (provided by Unsloth) using our custom recipe dataset. Mistral was selected for its efficiency in handling natural language tasks like generating recipes based on dietary tags and ingredients.

-> Data Preparation:
The dataset was preprocessed to align with the model's input format, which included dietary tags and ingredients. Prompts were constructed to guide the model in generating the recipe and nutritional information.

-> Training Procedure:
We used Hugging Face's SFTTrainer for fine-tuning, adjusting model weights to better fit our dataset.
Hyperparameters:
        - Batch Size: 4 (to fit within GPU memory limits).
        - Gradient Accumulation: 2 steps to simulate a larger batch size.
        - Learning Rate: 2e-4.
        - Steps: 150, chosen to balance performance and training time.
        - Precision: Mixed precision (fp16 or bf16) for reduced memory usage.
        - Quantization: 4-bit precision was used to lower memory requirements, making deployment more feasible.

The trained model was saved for later use in the inference phase.

-> Memory Management:
Techniques like adjusting batch sizes and quantization were employed to handle GPU constraints, requiring at least 15 GB of GPU memory to avoid out-of-memory errors.

### 4. Model Deployment

The project was designed for deployment on Streamlit. However, due to GPU limitations, direct deployment has not been completed. You can deploy the model by running the Streamlit.ipynb file step by step if you have access to a GPU with sufficient memory.

### 5. Usage

To use this project, follow these steps:

Step 1-> Download the Dataset: Download the CSV dataset file from the repository.

Step 2->Run the Model Training File: Execute the model training file to fine-tune the Mistral model on the custom dataset.

Step 3->Run the Inference File: Use the inference file to see the results of the trained model and generate recipes based on dietary restrictions and ingredients.

Step 4->Deploy on Streamlit (Optional): If you have enough GPU power, you can deploy the model using the provided Streamlit file.

Running Locally
If you want to use the model and dataset, follow these steps:

Clone the Repository:

git clone https://github.com/your-repo/recipe-generator.git
cd recipe-generator

Install Dependencies: Install the necessary Python packages:

pip install -r requirements.txt

Run the Jupyter Notebooks: The repository includes Jupyter notebooks that demonstrate how to preprocess the data, train the model, and generate recipes. To run these, start Jupyter:


Open the notebooks Model Training.ipynb and Inference.ipynb to explore the code and execute the steps.

Use the Model: You can load the pre-trained Mistral model from the notebook, input dietary restrictions and ingredients, and generate recipes with nutritional information.

### 6. Results

The fine-tuned model successfully generates detailed recipes based on the input dietary restrictions and available ingredients. Each recipe includes:

A recipe name
A step-by-step direction list
Nutritional information (calories, protein, fat, sodium)

Below our two examples attached from where you can understand:

### Example 1:

![Example 1](/EX1.jpg)

### Example 2:

![Example 2](examples/ex2.jpg)





## Contributing
Contributions to this project are welcome! If you want to improve the model, enhance the dataset, or add new features, feel free to fork the project and submit a pull request. You can also open an issue to discuss any ideas or concerns.

