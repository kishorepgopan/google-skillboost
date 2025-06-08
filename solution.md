Solution 1 # Build a Multi-Modal GenAI Application Challenge Lab

```
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

def generate_bouquet_image(prompt: str):
    """Generates an image using the imagen-3.0-generate-002 model and saves it locally."""

    # Initialize Vertex AI with the lab details
    vertexai.init(project="YOUR_PROJECT_ID", location="YOUR_LAB_LOCATION")

    # Load the model "imagen-3.0-generate-002"
    model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")

    # Generate image using the prompt
    images = model.generate_images(
        prompt=prompt,
        number_of_images=1,
        seed=1,
        add_watermark=False
    )

    # The image is saved in local with the name bouquet.jpeg
    with open("bouquet.jpeg", "wb") as f:
        f.write(images[0]._image_bytes)

    print("Saved bouquet.jpeg from the prompt")

# Call the function with the required prompt
generate_bouquet_image("Create an image containing a bouquet of 2 sunflowers and 3 roses")
```

Solution 2 # Build a Multi-Modal GenAI Application Challenge Lab

```
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
