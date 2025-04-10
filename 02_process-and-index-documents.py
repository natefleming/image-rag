# Databricks notebook source
from typing import Sequence

pip_requirements: Sequence[str] = [
  "langchain",
  "databricks-langchain",
  "langchain-docling",
  "langchain-text-splitters",
  "databricks-sdk",
  "delta-spark",
  "docling",
  "mlflow",
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
  f"langchain-docling=={version('langchain-docling')}",
  f"langchain-text-splitters=={version('langchain-text-splitters')}",
  f"databricks-sdk=={version('databricks-sdk')}",
  f"delta-spark=={version('delta-spark')}",
  f"docling=={version('docling')}",
  f"mlflow=={version('mlflow')}",
  f"python-dotenv=={version('python-dotenv')}",
]

print("\n".join(pip_requirements))


# COMMAND ----------

import os
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

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
source_table_name: str = data_config.get("source_table_name")
primary_key: str = data_config.get("primary_key")


print(f"catalog_name: {catalog_name}")
print(f"database_name: {database_name}")
print(f"volume_name: {volume_name}")
print(f"document_path: {document_path}")
print(f"source_table_name: {source_table_name}")
print(f"primary_key: {primary_key}")


# COMMAND ----------

from typing import (
    Any,
    Callable, 
    Iterator, 
    Literal, 
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
)
import io

from pathlib import Path

import pyspark.sql.functions as F
import pyspark.sql.types as T


import mimetypes

import pandas as pd


from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter

from databricks_langchain import ChatDatabricks

from databricks.sdk import WorkspaceClient

from docling.datamodel.document import DoclingDocument, InputFormat, DocumentStream

from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    ConversionResult,
    PdfFormatOption,
    WordFormatOption,
)
from docling_core.types.doc.document import PageItem
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

def document_converter(pipeline_options: Optional[PdfPipelineOptions] = None) -> DocumentConverter:
    if pipeline_options is None:
        pipeline_options = PdfPipelineOptions()

    converter: DocumentConverter = (
        DocumentConverter( 
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.DOCX,
                InputFormat.HTML,
                InputFormat.PPTX,
            ], 
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options, 
                    backend=PyPdfiumDocumentBackend
                ),
                InputFormat.DOCX: WordFormatOption(
                    pipeline_cls=SimplePipeline 
                ),
            },
        )
    )
    return converter


def chunk_document_factory(
    chunk_size: int = 1000, 
    chunk_overlap: int = 200
) -> Callable[[Iterator[pd.Series]], Iterator[pd.Series]]:

    @F.pandas_udf(T.ArrayType(T.StringType()))
    def chunk_document_udf(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
     
        text_splitter: TextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        def _chunk_document(document_batch: pd.Series) -> pd.Series:
            chunk_results: list[list[str]] = []
            
            for content in document_batch:
                chunks: list[str] = text_splitter.split_text(content)
                chunk_results.append(chunks)
                    
            return pd.Series(chunk_results)

        for document_batch in iterator:
            yield _chunk_document(document_batch)
        
    return chunk_document_udf


Format: TypeAlias = Literal["text", "markdown"] 


def export_document_factory(
    format: Optional[Format] = None,
) -> Callable[[Iterator[Tuple[pd.Series, pd.Series]]], Iterator[pd.Series]]:

    if format is None:
        format = "markdown"

    @F.pandas_udf(T.StringType())
    def export_document_udf(iterator: Iterator[Tuple[pd.Series, pd.Series]]) -> Iterator[pd.Series]:
     
        converter: DocumentConverter = document_converter()

        def _export_document(path_batch: pd.Series, content_batch: pd.Series) -> pd.Series:
            results: list[str] = []
            
            for path, content in zip(path_batch, content_batch):
                stream: io.BytesIO = io.BytesIO(content)
                source: DocumentStream = DocumentStream(name=path, stream=stream)
                result: ConversionResult = converter.convert(source)
                content: str
                match format:
                    case "markdown":
                        content = result.document.export_to_markdown()
                    case _:
                        content = result.document.export_to_text()

                results.append(content)
                    
            return pd.Series(results)
    

        for path_batch, content_batch in iterator:
            yield _export_document(path_batch, content_batch)

    return export_document_udf


PageType: T.StructType = T.StructType([
    T.StructField("page_content", T.StringType()),
    T.StructField("page_number", T.IntegerType()),
])

PageData: TypeAlias = dict[str, Any]


def export_pages_factory(
     format: Optional[Format] = None,
) -> Callable[[Iterator[Tuple[pd.Series, pd.Series]]], Iterator[pd.Series]]:

    if format is None:
        format = "markdown"

    @F.pandas_udf(T.ArrayType(PageType))
    def export_pages_udf(iterator: Iterator[Tuple[pd.Series, pd.Series]]) -> Iterator[pd.Series]:
     
        converter: DocumentConverter = document_converter()

        def _export_pages(path_batch: pd.Series, content_batch: pd.Series) -> pd.Series:
            results: list[list[PageData]] = []
            
            for path, content in zip(path_batch, content_batch):

                stream: io.BytesIO = io.BytesIO(content)
                source: DocumentStream = DocumentStream(name=path, stream=stream)
                result: ConversionResult = converter.convert(source)
              
                document: DoclingDocument = result.document
                page_items: list[PageData] = []
                for _, page_item in document.pages.items():
                    page_item: PageItem
                    page_number: int = page_item.page_no
                    page_content: str
                    match format:
                        case "markdown":
                            page_content = document.export_to_markdown(page_no=page_number)
                        case _:
                            page_content = document.export_to_text(page_no=page_number)

                    page_data: PageData = {
                        "page_content": page_content,
                        "page_number": page_number
                    }
                    page_items.append(page_data)

                results.append(page_items)
                    
            return pd.Series(results)
    
        for path_batch, content_batch in iterator:
            yield _export_pages(path_batch, content_batch)

    return export_pages_udf


# COMMAND ----------

from pyspark.sql import DataFrame

import pyspark.sql.functions as F
import pyspark.sql.types as T


spark.conf.set("spark.databricks.optimizer.adaptive.enabled", "false")
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "100")


def process_documents(df: DataFrame) -> DataFrame:

    documents_df: DataFrame = df.select(
        F.regexp_replace("_metadata.file_path", "dbfs:/", "/").alias("source_path"),
        df.content.alias("source_content")
    )

    export_document_udf = export_document_factory(format="markdown")
    documents_df = documents_df.withColumn(
        "content",
        export_document_udf(documents_df.source_path, documents_df.source_content),
    )

    chunk_document_udf = chunk_document_factory(chunk_size=1000, chunk_overlap=200)
    documents_df = documents_df.withColumn(
        "chunks",
        F.explode(chunk_document_udf(documents_df.content)),
    )

    documents_df = documents_df.drop(documents_df.source_content, documents_df.content)

    return documents_df


def process_pages(df: DataFrame) -> DataFrame:

    documents_df: DataFrame = df.select(
        F.regexp_replace("_metadata.file_path", "dbfs:/", "/").alias("source_path"),
        df.content.alias("source_content"),
    )

    # Export pages to markdown
    export_pages_udf = export_pages_factory(format="markdown")
    pages_df: DataFrame = documents_df.withColumn(
        "pages",
        F.explode(export_pages_udf(documents_df.source_path, documents_df.source_content)),
    )
    pages_df = pages_df.select(
        F.col("*"),
        F.col("pages.*"),
    )
    pages_df = pages_df.drop("source_content", "pages")
    
    # Chunk each each page 
    chunk_document_udf = chunk_document_factory(chunk_size=1000, chunk_overlap=200)
    chunks_df: DataFrame = pages_df.withColumn(
        "chunks",
        F.explode(chunk_document_udf(pages_df.page_content)),
    )
    chunks_df = chunks_df.drop("page_content")

    return chunks_df

# COMMAND ----------

document_path: Path = Path("/Volumes/nfleming/mars/documents/attention-is-all-you-need.pdf")

documents_df: DataFrame = spark.read.format("binaryFile").load(document_path.as_posix())
processed_df: DataFrame = process_documents(documents_df)

# COMMAND ----------


document_path: Path = Path("/Volumes/nfleming/mars/documents/attention-is-all-you-need.pdf")

documents_df: DataFrame = spark.read.format("binaryFile").load(document_path.as_posix())
processed_df: DataFrame = process_pages(documents_df)

processed_df = processed_df
display(processed_df)
#display(documents_df)

# COMMAND ----------

import os

from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import pyspark.sql.types as T

from delta.tables import DeltaTable, IdentityGenerator


documents_df: DataFrame = spark.read.format("binaryFile").load(document_path.as_posix())

processed_df: DataFrame = process_documents(documents_df)

index_df: DataFrame = (
  processed_df.select(
    processed_df.source_path,
    processed_df.chunks.alias("content")
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
