# Databricks notebook source
from typing import Sequence

pip_requirements: Sequence[str] = [
  "langchain",
  "databricks-langchain",
  "langchain-openai",
  "databricks-sdk",
  "delta-spark",
  "transformers",
  "pillow",
  "PyMuPDF",
  "mlflow",
  "filetype",
  "python-dotenv",

]
pip_requirements = " ".join(pip_requirements)

%pip install --quiet --upgrade {pip_requirements}
%restart_python

# COMMAND ----------

from typing import Sequence

from importlib.metadata import version


pip_requirements: Sequence[str] = [
  f"langchain=={version('langchain')}",
  f"databricks-langchain=={version('databricks-langchain')}",
  f"langchain-openai=={version('langchain-openai')}",
  f"databricks-sdk=={version('databricks-sdk')}",
  f"delta-spark=={version('delta-spark')}",
  f"transformers=={version('transformers')}",
  f"PyMuPDF=={version('PyMuPDF')}",
  f"pillow=={version('pillow')}",
  f"mlflow=={version('mlflow')}",
  f"filetype=={version('filetype')}",
  f"python-dotenv=={version('python-dotenv')}",
]

print("\n".join(pip_requirements))


# COMMAND ----------

import os
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()

openai_api_key: str = os.getenv("OPENAI_API_KEY")
databricks_api_key: str = context.apiToken().get()
base_url: str = context.apiUrl().get()

# COMMAND ----------

from typing import Any
from pathlib import Path

from mlflow.models import ModelConfig


model_config_file: str = "model_config.yaml"
config: ModelConfig = ModelConfig(development_config=model_config_file)

catalog_config: dict[str, Any] = config.get("catalog")
catalog_name: str = catalog_config.get("catalog_name")
database_name: str = catalog_config.get("database_name")
volume_name: str = catalog_config.get("volume_name")

data_config: dict[str, Any] = config.get("data")
document_path: str = Path(data_config.get("document_path"))
image_path: str = Path(data_config.get("image_path"))
source_table_name: str = data_config.get("source_table_name")
primary_key: str = data_config.get("primary_key")


print(f"catalog_name: {catalog_name}")
print(f"database_name: {database_name}")
print(f"volume_name: {volume_name}")
print(f"document_path: {document_path}")
print(f"image_path: {image_path}")
print(f"source_table_name: {source_table_name}")
print(f"primary_key: {primary_key}")


# COMMAND ----------

from typing import (
    Callable, 
    Iterator, 
    Literal, 
    Optional,
    Sequence,
    Tuple,
)
import io
import base64
import hashlib
from pathlib import Path

import pyspark.sql.functions as F
import pyspark.sql.types as T

from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

from PIL import Image
import filetype
import mimetypes

import pandas as pd
from pydantic import BaseModel, Field

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from databricks_langchain import ChatDatabricks

from databricks.sdk import WorkspaceClient


def load_classifications() -> Sequence[str]:
    classifications: Sequence[dict[str, str]] = config.get("classifications")
    return [classification.get("name") for classification in classifications]


def prompt_for_classification(name: str) -> str:
    classifications: Sequence[dict[str, str]] = config.get("classifications")
    prompt: str = ""
    for classification in classifications:
        if classification.get("name") == name:
            prompt = classification.get("prompt") or ""
            break
            
    return prompt
   

def is_diagram(
    image: Image, 
    min_width: int = 500,
    min_height: int = 375,
    max_width: int = 2000,
    max_height: int = 1500
) -> bool:
    width, height = image.size
    return (
        width >= min_width and 
        width <= max_width and 
        height >= min_height and 
        height <= max_height
    )


def format_image_path_factory() -> Callable[[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series], pd.Series]:
    @F.pandas_udf(T.StringType())
    def format_image_path_udf(
        source_paths: pd.Series, 
        page_numbers: pd.Series, 
        ordinals: pd.Series, 
        extensions: pd.Series, 
        destination_base_paths: pd.Series
    ) -> pd.Series:
        def _format_image_path(source_path: str, page_number: int, ordinal: int, extension: str, destination_base_path: str) -> str:
            source_path_obj = Path(source_path)
            stem: str = source_path_obj.stem
            path_hash: str = hashlib.md5(str(source_path_obj).encode()).hexdigest()[:8]
            new_filename: str = f"{stem}-{path_hash}-page{page_number}-index{ordinal}{extension}"
            return str(Path(destination_base_path) / new_filename)
        
        return pd.Series([
            _format_image_path(src, page, ord, ext, base_path) 
            for src, page, ord, ext, base_path in zip(source_paths, page_numbers, ordinals, extensions, destination_base_paths)
        ])

    return format_image_path_udf


def save_image_factory() -> Callable[[Iterator[Tuple[pd.Series, pd.Series]]], Iterator[pd.Series]]:
    @F.pandas_udf(T.StringType())
    def save_image_udf(iterator: Iterator[Tuple[pd.Series, pd.Series]]) -> Iterator[pd.Series]:
    
        w: WorkspaceClient = WorkspaceClient(host=base_url, token=databricks_api_key)

        def _save_image(file_path: str, image_bytes: bytes) -> bool:
            try:
                w.files.upload(file_path, image_bytes, overwrite=True)
                return "success"
            except Exception as e:
                print(f"Error saving image to {file_path}: {e}")
                return str(e)

        for file_path_series, image_bytes_series in iterator:
            results: list[bool] = []
            for file_path, image_bytes in zip(file_path_series, image_bytes_series):
                results.append(_save_image(file_path, image_bytes))
            yield pd.Series(results)

    return save_image_udf


ImageMetadataType: T.StructType = T.StructType([
    T.StructField("extracted_image", T.BinaryType()),
    T.StructField("page_number", T.IntegerType()),
    T.StructField("ordinal", T.IntegerType()),
    T.StructField("image_mime_type", T.StringType()),
    T.StructField("image_ext", T.StringType()),
])


def extract_images_factory() -> Callable[[Iterator[Tuple[pd.Series, pd.Series, pd.Series]]], Iterator[pd.Series]]:
    @F.pandas_udf(T.ArrayType(ImageMetadataType))
    def extract_images_udf(document_metdata_batches: Iterator[Tuple[pd.Series, pd.Series, pd.Series]]) -> Iterator[pd.Series]:
        
        def _handle_jpeg(source_path: Path, content: bytes) -> list[bytes]:
            mime_type: str = "image/jpeg"
            return [{
                "extracted_image": content,
                "page_number": 1,
                "ordinal": 1,
                "image_mime_type": mime_type,
                "image_ext": mimetypes.guess_extension(mime_type)
            }]

        def _handle_pdf(source_path: Path, content: bytes) -> list[bytes]:
            import fitz

            pdf_file = fitz.open(stream=content)

            images: list[dict] = []
            for page_number in range(len(pdf_file)): 

                page=pdf_file[page_number]
                image_list = page.get_images()
                for image_index, img in enumerate(page.get_images(),start=1):
                    xref = img[0] 
                    base_image = pdf_file.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    
                    if is_diagram(pil_image):
                        bytes_io: io.BytesIO = io.BytesIO()
                        pil_image.save(bytes_io, format="JPEG")
                        image_content: bytes = bytes_io.getvalue()
                        mime_type: str = "image/jpeg"
                        images.append({
                            "extracted_image": image_content,
                            "page_number": int(page_number),
                            "ordinal": int(image_index),
                            "image_mime_type": mime_type,
                            "image_ext": mimetypes.guess_extension(mime_type)
                        })
            
            return images
        
        def _extract_images(source_batch: pd.Series, mime_type_batch: pd.Series, document_batch: pd.Series) -> pd.Series:
            images: list[list[bytes]] = []
            for source, mime_type, document in zip(source_batch, mime_type_batch, document_batch):
                print(f"mime_type: {mime_type}, document: {len(document)}")
                source_path: Path = Path(source)
                match mime_type:
                    case "image/jpeg":
                        images.append(_handle_jpeg(source_path, document))
                    case "application/pdf":
                        images.append(_handle_pdf(source_path, document))
                    case _:
                        ...

            return pd.Series(images)

        for source_batch, mime_type_batch, document_batch in document_metdata_batches:
            yield _extract_images(source_batch, mime_type_batch, document_batch)

    return extract_images_udf


def image_size_factory() -> Callable[[pd.Series], pd.Series]:
    @F.pandas_udf(T.ArrayType(T.IntegerType()))
    def image_size_udf(images: pd.Series) -> pd.Series:
        def _image_size(b: bytes):
            try:
                img = Image.open(io.BytesIO(b))
                width, height = img.size
                return [width, height]
            except Exception as e:
                print(f"Error getting dimensions")
                return None

        return images.apply(_image_size)

    return image_size_udf


def resize_image_factory(max_dimension: int = 1120) -> Callable[[pd.Series], pd.Series]:
    @F.pandas_udf(T.BinaryType())
    def resize_image_udf(images: pd.Series) -> pd.Series:
        
        def _resize_image(b: bytes):
            try:
                img = Image.open(io.BytesIO(b))

                original_width, original_height = img.size

                if original_width < max_dimension and original_height < max_dimension:
                    return b

                if original_width > original_height:
                    scaling_factor = max_dimension / original_width
                else:
                    scaling_factor = max_dimension / original_height

                new_width = int(original_width * scaling_factor)
                new_height = int(original_height * scaling_factor)

                resized_img = img.resize((new_width, new_height))

                buffered = io.BytesIO()
                resized_img.save(buffered, format=img.format)
                return buffered.getvalue()

            except Exception as e:
                print(f"Error resizing image: {e}")
                return None

        return images.apply(_resize_image)

    return resize_image_udf


def guess_mime_type_factory() -> Callable[[pd.Series], pd.Series]:
    @F.pandas_udf(T.StringType())
    def guess_mime_type_udf(content: pd.Series) -> pd.Series:
        def _guess_mime_type(b: bytes) -> str:
            kind = filetype.guess(io.BytesIO(b))
            return kind.mime if kind else None

        return content.apply(_guess_mime_type)

    return guess_mime_type_udf


def guess_extension_factory() -> Callable[[pd.Series], pd.Series]:
    @F.pandas_udf(T.StringType())
    def guess_extension_udf(mime_types: pd.Series) -> pd.Series:
        def _guess_extension(mime_type: str) -> str:
            ext: str = mimetypes.guess_extension(m)
            return ext 

        return mime_types.apply(_guess_extension)
    
    return guess_extension_udf


def image_to_base64_factory() -> Callable[[pd.Series], pd.Series]:
    @F.pandas_udf(T.StringType())
    def image_to_base64_udf(image: pd.Series) -> pd.Series:
        def _image_to_base64(b: bytes) -> str:
            try:
                return base64.b64encode(b).decode("utf-8")
            except Exception as e:
                return None

        return image.apply(_image_to_base64)
    
    return image_to_base64_udf


def image_caption_factory() -> Callable[[Iterator[pd.Series]], Iterator[pd.Series]]:

    @F.pandas_udf(T.StringType())
    def image_caption_udf(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
        model_name: str = "Salesforce/blip-image-captioning-large"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

        for image_batch in iterator:
            captions = []
            for img_bytes in image_batch:
                try:
                    bytes_io = io.BytesIO(img_bytes)
                    pil_image = Image.open(bytes_io).convert("RGB")
                    inputs = processor(pil_image, return_tensors="pt").to(device)
                    output = model.generate(**inputs)
                    caption = processor.decode(output[0], skip_special_tokens=True)
                    captions.append(caption)
                except Exception as e:
                    captions.append(f"Error captioning image: {str(e)}")

            yield pd.Series(captions)

    return image_caption_udf


def classify_image_factory(model: str, api_key: str) -> Callable[[Iterator[pd.Series]], Iterator[pd.Series]]:
    @F.pandas_udf(T.StringType())
    def classify_image_udf(image_batches: Iterator[pd.Series]) -> Iterator[pd.Series]:
        
        from typing import Literal
        from langchain_openai import ChatOpenAI
        from pydantic import BaseModel, Field

        llm: LanguageModelLike = ChatOpenAI(model=model, api_key=api_key)
        
        allowed_classifications: list[str] = load_classifications()

        class ImageClassification(BaseModel):
            """
            A model output for image classification.
            """
            classification: Literal[tuple(allowed_classifications)] = Field(..., description="The image classification")

        def _classify_image(base64_image: str) -> str:
            message: BaseMessage = HumanMessage(
                content=[
                    {"type": "text", "text": "Classify this image using provided tools"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            )
            llm_with_tools: LanguageModelLike = llm.with_structured_output(ImageClassification)
            image_classification: ImageClassification = llm_with_tools.invoke(input=[message])
            #image_classification: ImageClassification = ImageClassification(classification="general")
            return image_classification.classification

        for image_batch in image_batches:
            yield image_batch.apply(_classify_image)

    return classify_image_udf



def summarize_image_factory(model: str, api_key: str) -> Callable[[Iterator[Tuple[pd.Series, pd.Series]]], Iterator[pd.Series]]:
    @F.pandas_udf(T.StringType())
    def summarize_image_udf(image_batches: Iterator[Tuple[pd.Series, pd.Series]]) -> Iterator[pd.Series]:
        
        from langchain_openai import ChatOpenAI
 
        llm: LanguageModelLike = ChatOpenAI(model=model, api_key=api_key)

        def _summarize_images(image_classification_batch: pd.Series, base64_image_batch: pd.Series) -> pd.Series:
            summaries = []
            for image_classification, base64_image in zip(image_classification_batch, base64_image_batch):
                try:

                    classification_prompt: str = prompt_for_classification(image_classification)
                    classification_prompt = classification_prompt or prompt_for_classification("default")

                    prompt: str = f"Summarize this image of a {image_classification} using the provided tools."
                    if classification_prompt:
                        prompt = f"{prompt} Instructions: {classification_prompt}"

                    message: BaseMessage = HumanMessage(
                        content=[
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    )
                    response: AIMessage = llm.invoke(input=[message])
                    #response: AIMessage = AIMessage(content="Sample output")
                    content: str = response.content
                    summaries.append(content)
                except Exception as e:
                    summaries.append(f"Error summarizing image: {str(e)}")
            
            return pd.Series(summaries)

        for image_classification, image_batch in image_batches:
            yield _summarize_images(image_classification, image_batch)

    return summarize_image_udf


# COMMAND ----------

from pyspark.sql import DataFrame

import pyspark.sql.functions as F
import pyspark.sql.types as T


spark.conf.set("spark.databricks.optimizer.adaptive.enabled", "false")
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "10")


def process_documents(df: DataFrame) -> DataFrame:

    guess_mime_type_udf = guess_mime_type_factory()
    documents_df = df.select(
        F.regexp_replace("_metadata.file_path", "dbfs:/", "/").alias("source_path"),
        df.modificationTime.alias("source_modification_time"),
        df.length.alias("source_length"),
        df.content.alias("source_content"),
        guess_mime_type_udf(df.content).alias("source_mime_type"),
    )

    extract_images_udf = extract_images_factory()
    documents_df = documents_df.withColumn(
        "extracted_image_with_metadata",
        F.explode(extract_images_udf(documents_df.source_path, documents_df.source_mime_type, documents_df.source_content)),
    )

    documents_df = documents_df.repartition(os.cpu_count())

    documents_df = documents_df.select(
        F.col("*"),
        F.col("extracted_image_with_metadata.*"),
    )

    documents_df = documents_df.drop("extracted_image_with_metadata")

    resize_image_udf = resize_image_factory()
    documents_df = documents_df.withColumn(
        "resized_image", resize_image_udf(documents_df.extracted_image)
    )

    image_size_udf = image_size_factory()
    image_to_base64_udf = image_to_base64_factory()
    image_caption_udf = image_caption_factory()
    documents_df = documents_df.withColumns(
        {
            "original_image_size": image_size_udf(documents_df.extracted_image),
            "resized_image_size": image_size_udf(documents_df.resized_image),
            "base64_image": image_to_base64_udf(documents_df.resized_image),
            "image_caption": image_caption_udf(documents_df.resized_image),
        }
    )

    classify_image_udf = classify_image_factory(model="gpt-4o", api_key=openai_api_key)
    documents_df = documents_df.withColumn(
        "image_classification", classify_image_udf(documents_df.base64_image)
    )

    summarize_image_udf = summarize_image_factory(model="gpt-4o", api_key=openai_api_key)
    documents_df = documents_df.withColumn(
        "image_summary",
        summarize_image_udf(
            documents_df.image_classification, documents_df.base64_image
        ),
    )

    format_image_path_udf = format_image_path_factory()
    documents_df = documents_df.withColumn(
        "image_path",
        format_image_path_udf(
            documents_df.source_path,
            documents_df.page_number,
            documents_df.ordinal,
            documents_df.image_ext,
            F.lit(image_path.as_posix())
        )
    )

    save_image_udf = save_image_factory()
    documents_df = documents_df.withColumn(
        "is_saved",
        save_image_udf(
            documents_df.image_path, documents_df.resized_image
        ),
    )

    return documents_df

# COMMAND ----------

glob_pattern = "*.jpeg"

documents_df: DataFrame = spark.read.format("binaryFile").option("pathGlobFilter", glob_pattern).load(document_path.as_posix())
processed_df: DataFrame = process_documents(documents_df)

display(processed_df)
#display(documents_df)

# COMMAND ----------

import os

from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import pyspark.sql.types as T

from delta.tables import DeltaTable, IdentityGenerator


glob_pattern: str = "*.pdf"

documents_df: DataFrame = spark.read.format("binaryFile").load(document_path.as_posix())
#documents_df: DataFrame = spark.read.format("binaryFile").option("pathGlobFilter", glob_pattern).load(document_path.as_posix())

processed_df: DataFrame = process_documents(documents_df)

index_df: DataFrame = (
  processed_df.select(
    processed_df.source_path,
    processed_df.source_mime_type,
    processed_df.source_modification_time,
    processed_df.source_length,
    processed_df.image_path,
    processed_df.image_mime_type,
    processed_df.image_classification,
    processed_df.image_caption,
    processed_df.page_number,
    processed_df.ordinal,
    processed_df.original_image_size,
    processed_df.resized_image_size,
    processed_df.image_summary.alias("content"),
  )
)


# COMMAND ----------


(
  DeltaTable.createOrReplace(spark)
    .tableName(source_table_name)
    .property("delta.enableChangeDataFeed", "true")
    .addColumn(primary_key, dataType=T.LongType(), nullable=False, generatedAlwaysAs=IdentityGenerator())
    .addColumns(index_df.schema)
    .execute()
)

spark.sql(f"ALTER TABLE {source_table_name} ADD CONSTRAINT {primary_key}_pk PRIMARY KEY ({primary_key})")

index_df.write.mode("append").saveAsTable(source_table_name)

# COMMAND ----------

display(spark.table(source_table_name))

# COMMAND ----------

from langchain_openai import ChatOpenAI
from databricks.sdk import WorkspaceClient
from openai import OpenAI
from langchain_core.messages import HumanMessage


model: str = "gpt-4o"   

base64_image: str = processed_df.select(processed_df.base64_image).first().base64_image

w: WorkspaceClient = WorkspaceClient()
llm: LanguageModelLike = ChatOpenAI(model=model)

message: BaseMessage = HumanMessage(
    content=[
        {"type": "text", "text": "Classify this image using provided tools"},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
            },
        },
    ],
)
llm_with_tools: LanguageModelLike = llm.with_structured_output(ImageClassification)
classification: ImageClassification = llm_with_tools.invoke(input=[message])


classification
