# Databricks notebook source
from typing import Sequence

# flash_attn==2.5.8
# numpy==1.24.4
# Pillow==10.3.0
# Requests==2.31.0
# torch==2.3.0
# torchvision==0.18.0
# transformers==4.43.0
# accelerate==0.30.0

pip_requirements: Sequence[str] = [
   # "flash_attn==2.5.8",
    "Pillow==10.3.0",
    "torch==2.3.0",
    "transformers==4.43.0",
    "cloudpickle==2.2.1",
    "accelerate==0.30.0",
    "torchvision==0.18.0",
    "mlflow",
    "python-dotenv",
]
pip_requirements = " ".join(pip_requirements)

%pip install --upgrade {pip_requirements}
%restart_python

# COMMAND ----------

from typing import Sequence

from importlib.metadata import version


pip_requirements: Sequence[str] = [
  f"transformers=={version('transformers')}",
  f"torch=={version('torch')}",
  f"torchvision=={version('torchvision')}",
  f"accelerate=={version('accelerate')}",
  f"flash_attn=={version('flash_attn')}",
  f"pillow=={version('Pillow')}",
  f"Requests=={version('Requests')}",
  f"mlflow=={version('mlflow')}",
]

print("\n".join(pip_requirements))


# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

from typing import Any

from mlflow.models import ModelConfig

model_config_file: str = "model_config.yaml"
config: ModelConfig = ModelConfig(development_config=model_config_file)

models_config: dict[str, Any] = config.get("models")
registered_vision_model: str = models_config.get("registered_vision_model")
vision_model_endpoint: str = models_config.get("vision_model_endpoint")

print(f"registered_vision_model: {registered_vision_model}")
print(f"vision_model_endpoint: {vision_model_endpoint}")

# COMMAND ----------

from PIL import Image
import requests
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

model_id = "microsoft/Phi-3.5-vision-instruct"

# Note: set _attn_implementation='eager' if you don't have flash_attn installed
model = AutoModelForCausalLM.from_pretrained(
  model_id,
  device_map="cuda",
  trust_remote_code=True,
  torch_dtype="auto",
  _attn_implementation='flash_attention_2'
)

# for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
processor = AutoProcessor.from_pretrained(model_id,
  trust_remote_code=True,
  num_crops=4
)

images = []
placeholder = ""

# Note: if OOM, you might consider reduce number of frames in this example.
for i in range(1, 20):
    url = f"https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-{i}-2048.jpg"
    images.append(Image.open(requests.get(url, stream=True).raw))
    placeholder += f"<|image_{i}|>\n"

messages = [
    {"role": "user", "content": placeholder + "Summarize the deck of slides."},
]

prompt = processor.tokenizer.apply_chat_template(
  messages,
  tokenize=False,
  add_generation_prompt=True
)

inputs = processor(prompt, images, return_tensors="pt").to("cuda:0")

generation_args = {
    "max_new_tokens": 1000,
    "temperature": 0.0,
    "do_sample": False,
}

generate_ids = model.generate(**inputs,
  eos_token_id=processor.tokenizer.eos_token_id, 
  **generation_args
)

# remove input tokens
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(generate_ids,
  skip_special_tokens=True,
  clean_up_tokenization_spaces=False)[0]

print(response)

# COMMAND ----------

import mlflow
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import base64
import io
from huggingface_hub import snapshot_download

model_path = snapshot_download(repo_id="microsoft/phi-3.5-vision-instruct")

class Phi35VisionWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        # Load the model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(artifacts['model'], torch_dtype=torch.float16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(artifacts['model'])

    def predict(self, context, model_input):
        # Extract text and images from the input
        text = model_input["text"][0]
        image_base64_list = model_input["images"]

        # Decode the base64 images
        images = []
        for image_base64 in image_base64_list:
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            images.append(image)

        # Prepare the input for the model
        inputs = self.tokenizer(text, images, return_tensors="pt").to(self.model.device)

        # Generate the output
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100)

        # Decode and return the result
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [result]

# Save the model
model = Phi35VisionWrapper()
artifacts = {
    "model": model_path
}




# COMMAND ----------

from mlflow.models import ModelSignature
from mlflow.models.model import ModelInfo
from mlflow.types import Schema, ColSpec, DataType

# Define the input schema
input_schema = Schema([
    ColSpec(DataType.string, "text"),
    ColSpec(DataType.string, "images")
])

output_schema = Schema([ColSpec(DataType.string)])

# Create the model signature
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

with mlflow.start_run(run_name="phi-3.5-vision-model"):
    # Log model parameters
    mlflow.log_param("model_name", "microsoft/phi-3.5-vision-instruct")

    # Create and log the model
    model = Phi35VisionWrapper()
    artifacts = {
        "model": model_path
    }

    model_info: ModelInfo = mlflow.pyfunc.log_model(
        artifact_path="phi_3_5_vision_model",
        python_model=model,
        artifacts=artifacts,
        pip_requirements=pip_requirements,#["torch==2.3.0","transformers==4.44.2", "cloudpickle==2.2.1","accelerate==0.30.1","torchvision==0.18.0",],
        signature=signature
        )

    

# COMMAND ----------

import requests
import base64
import io

def download_and_encode_image(url: str):
  response = requests.get(url, timeout=10)
  response.raise_for_status()  # Raise an exception for bad status codes

  # Verify content type is an image
  content_type = response.headers.get('content-type', '')
  if not content_type.startswith('image/'):
      raise ValueError(f"Downloaded content is not an image. Content-Type: {content_type}")

  # Encode the image to base64
  base64_encoded = base64.b64encode(response.content).decode('utf-8')

  return base64_encoded



print(base64_image)

# COMMAND ----------

base64_image: str = download_and_encode_image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg")

sample_input = {
    "text": ["Describe these images:"],
    "images": [
        base64_image,
    ]
}

mlflow.models.predict(
    model_uri=model_info.model_uri,
    input_data=sample_input,
)

result = loaded_model.predict(sample_input)
print(result)

# COMMAND ----------

from mlflow.entities.model_registry.model_version import ModelVersion


mlflow.set_registry_uri("databricks-uc")

registered_model_info: ModelVersion = mlflow.register_model(
    model_uri=model_info.model_uri, 
    name=registered_vision_model
)

# COMMAND ----------

from mlflow import MlflowClient

def get_latest_model_version(model_name):
    mlflow_client = MlflowClient()
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    ServedEntityInput, 
    EndpointCoreConfigInput, 
    AutoCaptureConfigInput, 
    ServedModelInputWorkloadSize, 
    ServedModelInputWorkloadType
)


w = WorkspaceClient()
endpoint_config = EndpointCoreConfigInput(
    name=vision_model_endpoint,
    served_entities=[
        ServedEntityInput(
            entity_name=registered_vision_model,
            entity_version=get_latest_model_version(registered_vision_model),
            workload_size=ServedModelInputWorkloadSize.SMALL.value,
            workload_type=ServedModelInputWorkloadType.GPU_LARGE.value,
            scale_to_zero_enabled=True
        )
    ]
)
\
force_update = False #Set this to True to release a newer version (the demo won't update the endpoint to a newer model version by default)
try:
  existing_endpoint = w.serving_endpoints.get(vision_model_endpoint)
  print(f"endpoint {vision_model_endpoint} already exist - force update = {force_update}...")
  if force_update:
    w.serving_endpoints.update_config_and_wait(served_entities=endpoint_config.served_entities, name=vision_model_endpoint)
except:
    print(f"Creating the endpoint {vision_model_endpoint}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=vision_model_endpoint, config=endpoint_config)

# COMMAND ----------


# Example usage
# Define input example
input_example = pd.DataFrame({"image_urls": [f"https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-{i}-2048.jpg" for i in range(1, 20)]})

# COMMAND ----------

