# Databricks notebook source
# MAGIC %pip install --quiet --upgrade databricks-sdk databricks-vectorsearch mlflow
# MAGIC %restart_python

# COMMAND ----------

from typing import Sequence

from importlib.metadata import version


pip_requirements: Sequence[str] = (
  f"databricks-sdk=={version('databricks-sdk')}",
  f"databricks-vectorsearch=={version('databricks-vectorsearch')}",
  f"mlflow=={version('mlflow')}",
)
print("\n".join(pip_requirements))

# COMMAND ----------

from typing import Any, Sequence

from mlflow.models import ModelConfig


model_config_file: str = "model_config.yaml"
config: ModelConfig = ModelConfig(development_config=model_config_file)

retreiver_config: dict[str, Any] = config.get("retriever")

embedding_model_endpoint_name: str = retreiver_config.get("embedding_model_endpoint_name")
endpoint_name: str = retreiver_config.get("endpoint_name")
index_name: str = retreiver_config.get("index_name")
primary_key: str = retreiver_config.get("primary_key")
embedding_source_column: str = retreiver_config.get("embedding_source_column")
columns: Sequence[str] = retreiver_config.get("columns", [])
search_parameters: dict[str, Any] = retreiver_config.get("search_parameters", {})

data_config: dict[str, Any] = config.get("data")
source_table_name: str = data_config.get("source_table_name")


print(f"embedding_model_endpoint_name: {embedding_model_endpoint_name}")
print(f"endpoint_name: {endpoint_name}")
print(f"index_name: {index_name}")
print(f"primary_key: {primary_key}")
print(f"embedding_source_column: {embedding_source_column}")
print(f"columns: {columns}")
print(f"search_parameters: {search_parameters}")
print(f"source_table_name: {source_table_name}")


# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

def endpoint_exists(vsc: VectorSearchClient, vs_endpoint_name: str) -> bool:
    try:
        return vs_endpoint_name in [e['name'] for e in vsc.list_endpoints().get('endpoints', [])]
    except Exception as e:
        if "REQUEST_LIMIT_EXCEEDED" in str(e):
            print("WARN: couldn't get endpoint status due to REQUEST_LIMIT_EXCEEDED error.")
            return True
        else:
            raise e


vsc: VectorSearchClient = VectorSearchClient()

if not endpoint_exists(vsc, endpoint_name):
    vsc.create_endpoint_and_wait(name=endpoint_name, verbose=True, endpoint_type="STANDARD")

print(f"Endpoint named {endpoint_name} is ready.")


# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.vector_search.index import VectorSearchIndex


def index_exists(vsc: VectorSearchClient, endpoint_name: str, index_full_name: str) -> bool:
    try:
        vsc.get_index(endpoint_name, index_full_name).describe()
        return True
    except Exception as e:
        if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
            print(f'Unexpected error describing the index. This could be a permission issue.')
            raise e
    return False


should_delete: bool = False

if not index_exists(vsc, endpoint_name, index_name):
  print(f"Creating index {index_name} on endpoint {endpoint_name}...")
  vsc.create_delta_sync_index_and_wait(
    endpoint_name=endpoint_name,
    index_name=index_name,
    source_table_name=source_table_name,
    pipeline_type="TRIGGERED",
    primary_key=primary_key,
    embedding_source_column=embedding_source_column, #The column containing our text
    embedding_model_endpoint_name=embedding_model_endpoint_name #The embedding endpoint used to create the embeddings
  )
else:
  if should_delete:
    vsc.delete_index(endpoint_name=endpoint_name, index_name=index_name)
  vsc.get_index(endpoint_name, index_name).sync()

print(f"index {index_name} on table {source_table_name} is ready")

# COMMAND ----------

from typing import Dict, Any, List

import mlflow.deployments
from databricks.vector_search.index import VectorSearchIndex
from mlflow.deployments.databricks import DatabricksDeploymentClient

deploy_client: DatabricksDeploymentClient = mlflow.deployments.get_deploy_client("databricks")

question = "Show me a picture of a dog on a couch"

index: VectorSearchIndex = vsc.get_index(endpoint_name, index_name)
k: int = search_parameters.get("k", 3)

search_results: Dict[str, Any] = index.similarity_search(
  query_text=question,
  columns=columns,
  num_results=k)

chunks: List[str] = search_results.get('result', {}).get('data_array', [])
chunks

# COMMAND ----------

from PIL import Image

chunk: list[str] = chunks[0]
image_path: str = chunk[4]

display(Image.open(image_path))