# Build a Multi-Modal GenAI Application: Challenge Lab Solution

This guide provides solutions for completing the Build a Multi-Modal GenAI Application Challenge Lab. Follow the instructions below to create Python functions for both tasks.

## Task Overview

- **Task 1:** Develop a function to generate an image of a bouquet using the `imagen-3.0-generate-002` model.
- **Task 2:** Develop a function to analyze the generated image and create birthday wishes using the `gemini-2.0-flash-001` model.

## Task 1: Generate Bouquet Image

Create a Python file named `create_bouquet_image.py` and define the function as follows:

```python
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

# Function to generate a bouquet image using the supplied prompt
def generate_bouquet_image(prompt: str):    

    # Initialize Vertex AI with the lab details
    vertexai.init(project="YOUR_PROJECT_ID", location="YOUR_LAB_LOCATION")

    # Load the model "imagen-3.0-generate-002"
    model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")

    # Generate an image using the prompt
    images = model.generate_images(
        prompt=prompt,
        number_of_images=1,
        seed=1,
        add_watermark=False
    )

    # The image is saved locally with the name bouquet.jpeg
    with open("bouquet.jpeg", "wb") as f:
        f.write(images[0]._image_bytes)

    print("Saved bouquet.jpeg from the prompt")

# Call the function with the required prompt
generate_bouquet_image("Create an image containing a bouquet of 2 sunflowers and 3 roses")
```

# Run the file using:

```bash
/usr/bin/python3 /create_bouquet_image.py
```

## Task 2: Analyze Bouquet Image

Create another Python file named `generate_birthday_wishes.py` and define the function as follows:

```python
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Image

from google import genai
from google.genai.types import HttpOptions

def analyze_bouquet_image(project_id: str, location: str) -> None:
    # Initialize Vertex AI
    vertexai.init(project=project_id, location=location)

    # Load Gemini multimodal model
    multimodal_model = GenerativeModel("gemini-2.0-flash-001")

    # Load the image and create the input parts, with streaming enabled.
    # Note: the stream=True flag is passed as a keyword argument after the input list.
    stream_response = multimodal_model.generate_content(
        [
            Part.from_image(Image.load_from_file("bouquet.jpeg")),
            "Generate birthday wishes based on this image?"
        ],
        stream=True
    )

    # Print each part as it is generated in streaming mode.
    for part in stream_response:
        print(part.text, end="")

# Replace with your actual values
project_id = "YOUR_PROJECT_ID"
location = "YOUR_LAB_LOCATION"

# Call the function to print the streaming response.
analyze_bouquet_image(project_id, location)
```

# Run the file using:

```bash
/usr/bin/python3 generate_birthday_wishes.py
```

Copy and paste the appropriate code blocks into your lab environment, creating the specified Python files, and run them using the given commands. By following these steps, you'll be able to complete the lab tasks effectively within the Skillboost environment.
