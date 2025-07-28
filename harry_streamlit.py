#Imports
import os
import json
import base64
import re
import time
import traceback
from io import BytesIO
from typing import List, Dict, Any, Optional, TypedDict
import html
from langchain.globals import set_verbose, set_debug
from langchain_community.document_loaders import PyPDFDirectoryLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch, helpers as es_helpers
import pandas as pd
import streamlit as st
from PIL import Image
 
#Configuration 
ENABLE_LANGCHAIN_DEBUG = True
ENABLE_LANGCHAIN_VERBOSE = True
LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_PROJECT_NAME_CONFIG = "nvd-agentic-rag-finance-assistant-v18"
LANGCHAIN_API_ENDPOINT_CONFIG = https://api.smith.langchain.com
os.environ['LANGCHAIN_API_KEY'] = "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"
 
LLM_MODEL_ID = "meta/llama-3.1-70b-instruct"
LLM_API_URL = http://llama31-70b-nim-nvidia-nim.apps.ai-ocp.emea.dsc.local/v1
LLM_TEMPERATURE = 0;
 
EMBEDDINGS_API_URL = http://www.nv-embedding.apps.ai-ocp.emea.dsc.local/v1
EMBEDDINGS_MODEL_ID = "nvidia/nv-embedqa-e5-v5"; EMBEDDING_DIMENSIONS = 1024
 
RERANKER_API_URL = http://www.rerankqa-mistral-4b.apps.ai-ocp.emea.dsc.local/v1
RERANKER_MODEL_ID = "nvidia/nv-rerankqa-mistral-4b-v3"
 
ES_HOST = https://www.elasticsearch.apps.ai-ocp.emea.dsc.local; ES_USER = "elastic"
ES_PASSWORD_CONFIG = os.getenv("ES_PASSWORD", "dyak3z319h71u0Ch7y65avQs") # Use env var or default
ES_VERIFY_CERTS = False
ES_CA_CERTS_PATH = "/home/demo/es_http_ca.crt"; ES_INDEX_NAME = "finance_assistant_docs_v18"
USE_EXPLICIT_ES_MAPPING = True
 
BASE_DOC_PATH = "/home/jovyan/financedocs"; STOCK_DATA_CSV_FILENAME = "stockdata.csv"
PRIMARY_CSV_ENCODING = 'utf-8'; FALLBACK_CSV_ENCODING = 'latin1'
CSV_SOURCE_COLUMN = None
RELEVANT_CSV_METADATA_COLUMNS = ["MSFT", "IBM", "SBUX", "AAPL", "GSPC", "Date" ]
MAX_METADATA_VALUE_LENGTH = 500; DOCUMENT_URL_PREFIX = ""
TEXT_SPLITTER_CHUNK_SIZE = 512; TEXT_SPLITTER_CHUNK_OVERLAP = 64
 
TAVILY_API_KEY_CONFIG = os.getenv("TAVILY_API_KEY", "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") # Use env var or default
TAVILY_SEARCH_K = 3
VECTORSTORE_SEARCH_K = 4
 
APP_TITLE = "Agentic RAG Finance Assistant"; DELL_LOGO_PATH = "dell-logo.png"; NVIDIA_LOGO_PATH = "nvidia-logo.png"
 
 
# --- Helper Functions ---
def setup_langsmith_tracing():
    if LANGCHAIN_TRACING_V2 == "true":
        os.environ['LANGCHAIN_TRACING_V2'] = 'true'
        os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT_NAME_CONFIG
        os.environ['LANGCHAIN_ENDPOINT'] = LANGCHAIN_API_ENDPOINT_CONFIG
        # Ensure TAVILY_API_KEY is set for LangSmith if it's configured
        if TAVILY_API_KEY_CONFIG and not os.getenv("TAVILY_API_KEY"):
            os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY_CONFIG
        if os.getenv('LANGCHAIN_API_KEY'):
            print(f"LangSmith tracing enabled for project: {LANGCHAIN_PROJECT_NAME_CONFIG}")
        else:
            print(f"Warning: LANGCHAIN_TRACING_V2 is true for {LANGCHAIN_PROJECT_NAME_CONFIG}, but LANGCHAIN_API_KEY not set in environment.")
    else:
        print("LangSmith tracing is disabled.")
 
def sanitize_es_field_name(name: str) -> str:
    name = str(name)
    name = name.replace(".", "_dot_")
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    if name.startswith("_") and not name.startswith("__"): name = "field" + name
    if name.startswith("-"): name = "field_" + name
    return name if name else "empty_field_name"
 
def pil_to_base64_encoder(img_pil: Image.Image, format="PNG") -> str:
    buffered = BytesIO()
    img_pil.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode()
 
def format_docs_for_llm(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)
 
#Initialization Functions 
#@st.cache_resource
def initialize_llms_and_utils():
    llm = ChatOpenAI(
        base_url=LLM_API_URL,
        api_key=" YOUR API KEY ", #API key for what? The NIM doesn't require API key?
        model=LLM_MODEL_ID,
        temperature=0,
        max_tokens=None,
    )
    embeddings_model = NVIDIAEmbeddings(
        base_url=EMBEDDINGS_API_URL,
        model=EMBEDDINGS_MODEL_ID,
        truncate="END"
    )
    reranker_model = NVIDIARerank(
        base_url=RERANKER_API_URL,
        model=RERANKER_MODEL_ID,
        truncate="END"
    )
    if not os.getenv("TAVILY_API_KEY") and TAVILY_API_KEY_CONFIG:
        os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY_CONFIG
    if not os.getenv("TAVILY_API_KEY"):
        print("CRITICAL WARNING: TAVILY_API_KEY not set in environment or config. Web search will fail.")
        # Optionally, raise an error or provide a dummy tool
        search_tool = None # Indicate failure
    else:
        print(f"Tavily API Key found: {os.getenv('TAVILY_API_KEY')[:5]}...") # Print part of key for confirmation
        search_tool = TavilySearchResults(k=TAVILY_SEARCH_K)
 
    return llm, embeddings_model, reranker_model, search_tool
 
@st.cache_resource
def get_elasticsearch_client():
    try:
        effective_password = ES_PASSWORD_CONFIG
        if not effective_password:
            raise ValueError("Elasticsearch password not configured.")
        
        print("Attempting to connect to Elasticsearch...") # Added
        client = Elasticsearch(
            hosts=ES_HOST,
            basic_auth=(ES_USER, effective_password),
            verify_certs=ES_VERIFY_CERTS if ES_VERIFY_CERTS else None,
            ca_certs=ES_CA_CERTS_PATH if ES_VERIFY_CERTS and ES_CA_CERTS_PATH else None,
            request_timeout=30, # Overall request timeout in seconds
            #  Other timeout options if needed:
        )
        print("Elasticsearch client object created. Pinging...") # Added
        if not client.ping(request_timeout=10): # Specific timeout for ping
            raise ConnectionError("Failed to connect to Elasticsearch (ping failed within 10s).")
        print("Successfully connected to Elasticsearch and ping was successful.")
        return client
    except Exception as e:
        print(f"CRITICAL: Elasticsearch connection failed in get_elasticsearch_client: {type(e).__name__} - {e}")
        traceback.print_exc()
        return None
 
def attempt_csv_load_with_pandas(csv_path: str, streamlit_feedback: bool = True) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    encodings_to_try = [PRIMARY_CSV_ENCODING, FALLBACK_CSV_ENCODING]
    if PRIMARY_CSV_ENCODING == FALLBACK_CSV_ENCODING and FALLBACK_CSV_ENCODING is not None:
        encodings_to_try = [PRIMARY_CSV_ENCODING]
 
    for encoding in encodings_to_try:
        if encoding is None: continue
        print(f"Attempting direct pandas load of ENTIRE CSV: {csv_path} with encoding: {encoding}")
        try:
            df = pd.read_csv(csv_path, encoding=encoding, dtype=str, on_bad_lines='warn', engine='python')
            print(f"Pandas direct load SUCCESS with {encoding}. Shape: {df.shape}")
            return df, encoding
        except FileNotFoundError:
            msg = f"CRITICAL CSV ERROR: File not found at {csv_path}"
            if streamlit_feedback and 'st' in globals(): st.error(msg)
            print(msg)
            return None, None
        except pd.errors.EmptyDataError:
            msg = f"CSV WARNING: The file {csv_path} is empty."
            if streamlit_feedback and 'st' in globals(): st.warning(msg)
            print(msg)
            return pd.DataFrame(), encoding # Return empty DataFrame and encoding
        except UnicodeDecodeError:
            print(f"Pandas direct load FAILED with {encoding}: UnicodeDecodeError.")
            if encoding == encodings_to_try[-1]:
                msg = f"CRITICAL CSV ERROR: All attempted encodings ({encodings_to_try}) failed for '{os.path.basename(csv_path)}'. Verify file encoding."
                if streamlit_feedback and 'st' in globals(): st.error(msg)
                print(msg)
        except pd.errors.ParserError as pe:
            msg = f"CRITICAL CSV ERROR: ParserError for '{os.path.basename(csv_path)}' with {encoding}. File malformed. Error: {str(pe)[:200]}"
            if streamlit_feedback and 'st' in globals(): st.error(msg)
            print(msg)
            return None, None
        except Exception as e:
            msg = f"CRITICAL CSV ERROR: Failed to read '{os.path.basename(csv_path)}' with {encoding}. Error: {type(e).__name__}, {str(e)[:200]}"
            if streamlit_feedback and 'st' in globals(): st.error(msg)
            print(msg); traceback.print_exc()
            return None, None
    return None, None
 
def load_and_process_documents(streamlit_feedback: bool = True) -> List[Document]:
    stock_data_csv_path = os.path.join(BASE_DOC_PATH, STOCK_DATA_CSV_FILENAME)
    csv_docs: List[Document] = []
    pdf_docs: List[Document] = []
 
    df_csv, csv_successful_encoding = attempt_csv_load_with_pandas(stock_data_csv_path, streamlit_feedback)
 
    if df_csv is not None and csv_successful_encoding:
        if df_csv.empty:
            print(f"CSV file '{STOCK_DATA_CSV_FILENAME}' is empty. No CSV documents will be loaded.")
        else:
            print(f"Pandas direct load successful. Proceeding with Langchain CSVLoader, encoding: {csv_successful_encoding}")
            actual_metadata_cols = [col for col in RELEVANT_CSV_METADATA_COLUMNS if col in df_csv.columns]
            actual_source_column = CSV_SOURCE_COLUMN if CSV_SOURCE_COLUMN and CSV_SOURCE_COLUMN in df_csv.columns else None
            try:
                # CSVLoader expects csv_args for csv.reader, not pandas.read_csv
                csv_loader_args = {} # Example: {'delimiter': ';'} if not comma
                csv_loader = CSVLoader(
                    file_path=stock_data_csv_path,
                    encoding=csv_successful_encoding,
                    source_column=actual_source_column,
                    metadata_columns=actual_metadata_cols,
                    csv_args=csv_loader_args
                )
                loaded_csv_langchain_docs = csv_loader.load()
                temp_csv_docs = []
                for lc_csv_doc in loaded_csv_langchain_docs:
                    new_meta = {}
                    original_lc_source = str(lc_csv_doc.metadata.get('source', stock_data_csv_path))
                    new_meta['source'] = sanitize_es_field_name(os.path.basename(original_lc_source))
                    new_meta['display_source'] = DOCUMENT_URL_PREFIX + os.path.basename(original_lc_source)
                    if 'row' in lc_csv_doc.metadata:
                        new_meta['csv_row_number'] = str(lc_csv_doc.metadata['row'])
 
                    for col_name in actual_metadata_cols:
                        if col_name in lc_csv_doc.metadata: # Check if metadata_columns made it to the doc
                            s_key = sanitize_es_field_name(col_name)
                            value = lc_csv_doc.metadata[col_name]
                            s_val_str = str(value if value is not None else "")[:MAX_METADATA_VALUE_LENGTH]
                            new_meta[s_key] = s_val_str
                    lc_csv_doc.metadata = new_meta
                    temp_csv_docs.append(lc_csv_doc)
                csv_docs = temp_csv_docs
                print(f"Langchain CSVLoader processed {len(csv_docs)} CSV docs.")
            except Exception as e_csv_loader:
                error_type_str = type(e_csv_loader).__name__
                error_details_str = str(e_csv_loader)
                msg_for_ui = (f"Langchain CSVLoader failed for '{STOCK_DATA_CSV_FILENAME}'.\n"
                              f"Error Type: {error_type_str}\nDetails: {error_details_str[:200]}...\n"
                              "CSV documents will be skipped. Check console for full traceback.")
                msg_for_console = (f"Langchain CSVLoader failed for '{STOCK_DATA_CSV_FILENAME}'. "
                                   f"Error Type: {error_type_str}, Details: {error_details_str}")
                if streamlit_feedback and 'st' in globals(): st.warning(msg_for_ui)
                print(msg_for_console); print("--- Traceback for CSVLoader failure ---"); traceback.print_exc(); print("--- End Traceback ---")
                csv_docs = []
    elif df_csv is None and streamlit_feedback and 'st' in globals():
        st.warning(f"CSV file '{STOCK_DATA_CSV_FILENAME}' could not be loaded by pandas. CSV processing skipped.")
 
    try:
        pdf_dir_loader = PyPDFDirectoryLoader(BASE_DOC_PATH)
        pdf_docs = pdf_dir_loader.load()
        print(f"Loaded {len(pdf_docs)} PDF docs.")
    except Exception as e_pdf:
        msg = f"Error loading PDF documents: {e_pdf}."
        if streamlit_feedback and 'st' in globals(): st.error(msg)
        print(msg); traceback.print_exc()
 
    raw_documents = pdf_docs + csv_docs
    if not raw_documents:
        msg = "CRITICAL: No documents (PDF/CSV) loaded."
        if streamlit_feedback and 'st' in globals(): st.error(msg)
        print(msg)
        return []
 
    print(f"Total raw documents: {len(raw_documents)} (PDFs: {len(pdf_docs)}, CSV: {len(csv_docs)})")
    processed_docs_list = []
    for doc_item_proc in raw_documents:
        doc_item_proc.page_content = str(doc_item_proc.page_content) # Ensure content is string
        is_csv_doc_already_processed = 'csv_row_number' in doc_item_proc.metadata
 
        if not is_csv_doc_already_processed:
            current_meta = {}
            original_source_path = str(doc_item_proc.metadata.get('source', 'unknown_source'))
            current_meta['source'] = sanitize_es_field_name(os.path.basename(original_source_path))
            current_meta['display_source'] = DOCUMENT_URL_PREFIX + os.path.basename(original_source_path)
            for key, value in doc_item_proc.metadata.items():
                if key.lower() == 'source': continue # Already handled
                s_key = sanitize_es_field_name(key)
                s_val_str = str(value if value is not None else "")[:MAX_METADATA_VALUE_LENGTH]
                current_meta[s_key] = s_val_str
            doc_item_proc.metadata = current_meta
        processed_docs_list.append(doc_item_proc)
 
    print(f"Processed and sanitized {len(processed_docs_list)} documents.")
    return processed_docs_list
 
@st.cache_resource(show_spinner="Splitting documents...")
def split_documents(_processed_docs: List[Document]) -> List[Document]:
    if not _processed_docs: return []
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=TEXT_SPLITTER_CHUNK_SIZE,
        chunk_overlap=TEXT_SPLITTER_CHUNK_OVERLAP
    )
    doc_splits = text_splitter.split_documents(_processed_docs)
    print(f"Split into {len(doc_splits)} chunks.")
    for split in doc_splits:
        split.metadata = {
            sanitize_es_field_name(k): str(v if v is not None else "")[:MAX_METADATA_VALUE_LENGTH]
            for k, v in split.metadata.items()
        }
    return doc_splits
 
def get_elasticsearch_mapping():
    metadata_properties = {
        "source": {"type": "keyword"},
        "display_source": {"type": "keyword"},
        "csv_row_number": {"type": "keyword"},
        # Example PDF metadata fields (use sanitized names)
        sanitize_es_field_name("creationdate"): {"type": "date", "ignore_malformed": True},
        sanitize_es_field_name("moddate"): {"type": "date", "ignore_malformed": True},
        sanitize_es_field_name("producer"): {"type": "keyword"},
        sanitize_es_field_name("creator"): {"type": "keyword"},
        sanitize_es_field_name("title"): {"type": "text"}, # Or keyword if not analyzed
        sanitize_es_field_name("author"): {"type": "keyword"},
        sanitize_es_field_name("total_pages"): {"type": "integer", "ignore_malformed": True}, # Also for numbers
        sanitize_es_field_name("page"): {"type": "integer", "ignore_malformed": True},
        sanitize_es_field_name("page_label"): {"type": "keyword"},
    }
    # Add CSV columns
    for col_name in RELEVANT_CSV_METADATA_COLUMNS:
        metadata_properties[sanitize_es_field_name(col_name)] = {"type": "keyword"}
 
    return {
        "properties": {
            "text": {"type": "text"},
            "vector": {"type": "dense_vector", "dims": EMBEDDING_DIMENSIONS},
            "metadata": {
                "dynamic": True,
                "properties": metadata_properties
            }
        }
    }
@st.cache_resource(show_spinner="Indexing documents to Elasticsearch...")
def get_vectorstore_and_retriever(_embeddings_model, _es_client, _doc_splits: List[Document]):
    if not _doc_splits: # CHECK 3
        print("CRITICAL CHECK: _doc_splits is empty. No documents were passed for indexing.")
        st.warning("No documents were loaded or split. Vectorstore search will be unavailable.") # UI feedback
        return None, None
    if _es_client is None: # CHECK 4
        print("CRITICAL CHECK: _es_client is None. Elasticsearch client failed to initialize.")
        st.error("Elasticsearch client is not available. Cannot index documents.") # UI feedback
        return None, None
 
    print(f"Attempting to index {len(_doc_splits)} document splits.") # Confirm number of splits
 
    print("\n--- Pre-indexing Sample (first 3 splits if available) ---")
    for i, ds in enumerate(_doc_splits[:min(3, len(_doc_splits))]):
        print(f"Split {i}: Src='{ds.metadata.get('display_source', ds.metadata.get('source','N/A'))}', Content='{ds.page_content[:70].replace(chr(10), ' ')}...'")
        print(f"  MetaKeys: {list(ds.metadata.keys())}")
    print("--------------------------------------------\n")
 
    try: # Outer try for general errors
        if _es_client.indices.exists(index=ES_INDEX_NAME, request_timeout = 20):
            print(f"Deleting existing ES index '{ES_INDEX_NAME}'...")
            _es_client.indices.delete(index=ES_INDEX_NAME, request_timeout=30)
            print(f"Index '{ES_INDEX_NAME}' deleted.")
 
        print(f"Creating ES index '{ES_INDEX_NAME}' with explicit mapping...")
        mapping_body = get_elasticsearch_mapping() # Ensure this function is correct
        # print(f"DEBUG: ES Mapping to be applied:\n{json.dumps(mapping_body, indent=2)}") # Optional: print mapping
        _es_client.indices.create(index=ES_INDEX_NAME, mappings=mapping_body)
        print(f"Index '{ES_INDEX_NAME}' created with mapping.")
 
        print(f"Attempting ElasticsearchStore.from_documents for {len(_doc_splits)} splits...")
        vectorstore = ElasticsearchStore.from_documents(
            documents=_doc_splits,
            embedding=_embeddings_model,
            index_name=ES_INDEX_NAME,
            es_connection=_es_client,
            strategy=ElasticsearchStore.ExactRetrievalStrategy(),
            # bulk_kwargs={'request_timeout': 120, 'max_retries': 0} # Adjust as needed, 0 retries for clearer single error
        )
        print("Successfully indexed documents via ElasticsearchStore.from_documents.")
        retriever = vectorstore.as_retriever()
        print("Retriever created successfully.")
        return vectorstore, retriever
 
    except es_helpers.BulkIndexError as e_bulk: # CHECK 1 (Specific ES Bulk Error)
        print(f"CRITICAL: ES BulkIndexError: {len(e_bulk.errors)} document(s) failed to index.")
        print("Detailed errors for failing documents:")
        for i, error_info in enumerate(e_bulk.errors):
            print(f"\n--- Bulk Error Detail #{i+1} ---")
            print(json.dumps(error_info, indent=2))
            if 'index' in error_info and 'error' in error_info['index'] and 'reason' in error_info['index']['error']:
                reason = error_info['index']['error']['reason']
                print(f"Reason from ES: {reason}")
                if "dimension" in reason.lower() and "dense_vector" in reason.lower():
                    print("POSSIBLE VECTOR DIMENSION MISMATCH! Check EMBEDDING_DIMENSIONS.")
                if "mapper_parsing_exception" in error_info['index']['error'].get('type', '').lower():
                    print("POSSIBLE DATA TYPE MISMATCH or unmapped field. Check the 'reason' above against your document metadata and ES mapping.")
        
        print("\n--- All Document Splits (First few, for manual correlation with errors) ---")
        for idx, doc_split_item in enumerate(_doc_splits[:max(10, len(e_bulk.errors) + 2)]): # Print enough to cover errors + a few more
            print(f"\n--- Doc Split Index in Python List: {idx} ---")
            print(f"  Content (first 100 chars): {doc_split_item.page_content[:100].replace(chr(10), ' ')}...")
            print(f"  Metadata:")
            try:
                print(json.dumps(doc_split_item.metadata, indent=2, default=str))
            except TypeError:
                print(str(doc_split_item.metadata))
        traceback.print_exc()
        return None, None # Critical: ensure this path returns None, None
 
    except Exception as e: # CHECK 2 (Other general errors in this function)
        print(f"CRITICAL: An unexpected error occurred in get_vectorstore_and_retriever: {type(e).__name__} - {e}")
        print("This was NOT a BulkIndexError. Traceback:")
        traceback.print_exc()
        return None, None # Critical: ensure this path returns None, None
 
# --- LangGraph Agent Setup ---
class GraphState(TypedDict):
    question: str
    web_search_enabled: bool
    generation: str
    documents: List[Document]
    relevant_docs_count: int
    initial_docs_count: int
    routing_decision: Optional[str]
    retrieval_status: Optional[str]
    relevance_filtering_status: Optional[str]
    web_search_fallback_status: Optional[str]
    hallucination_check_status: Optional[str]
    usefulness_check_status: Optional[str]
    # Internal flag for routing after grading generation
    _last_decision_from_grade_generation: Optional[str]
    # Internal flag for routing after grading documents
    _web_search_decision_from_grade: Optional[str]
 
 
ROUTER_PROMPT_TEMPLATE = PromptTemplate.from_template(
    """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to:
- Specific company stock information (e.g., Microsoft, Apple, IBM)
- General mortgage information
- Financial reports for Bank of Ireland (BOI), AIB, Permanent TSB (PTSB) for the year 2024
 
If the question directly relates to these topics, route to "vectorstore".
For all other questions, including general knowledge, current events outside of the specified financials/sports, or company information not listed, route to "web_search".
Return a JSON object with a single key "datasource" and the value "vectorstore" or "web_search".
 
Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
)
RELEVANCE_GRADER_TEMPLATE = PromptTemplate.from_template(
    """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keywords related to the user question, it is relevant.
It does not need to be a strict answer, but it should be on the same topic.
If the content is an error message, a navigation menu, or clearly unrelated, it is not relevant.
Answer "yes" or "no" to indicate whether the document is relevant to the question.
Return a JSON object with a single key "score" and the value "yes" or "no".
 
Document:
{document}
 
User Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
)
RAG_GENERATION_TEMPLATE = PromptTemplate.from_template(
    """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful and professional financial assistant. 
If the context is insufficient to answer the question, state that you cannot find enough information in the provided documents.
Be direct and factual.
 
Context:
{context}
 
Question: {question}
Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
)
HALLUCINATION_GRADER_TEMPLATE = PromptTemplate.from_template(
    """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a grader assessing whether an LLM generation is grounded in / supported by a set of facts.
The generation is grounded if all claims made in the generation are substantiated by the provided facts.
If the generation introduces new information not found in the facts, or contradicts the facts, it is not grounded.
Answer "yes" or "no". "yes" means the generation is grounded. "no" means it is not.
Return a JSON object with a single key "score" and the value "yes" or "no".
 
Facts:
{documents}
 
LLM Generation: {generation}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
)
USEFULNESS_GRADER_TEMPLATE = PromptTemplate.from_template(
    """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a grader assessing whether an LLM generation is useful and relevant to the user's question.
The generation is useful if it directly addresses the question and provides a pertinent answer.
If the generation is vague, off-topic, or simply states it cannot answer without providing a reason related to the question's scope, it is not useful.
Answer "yes" or "no". "yes" means the generation is useful and relevant. "no" means it is not.
Return a JSON object with a single key "score" and the value "yes" or "no".
 
User Question: {question}
LLM Generation: {generation}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
)
 
def initialize_graph_chains(llm_instance):
    return {
        "router": ROUTER_PROMPT_TEMPLATE | llm_instance | JsonOutputParser(),
        "relevance_grader": RELEVANCE_GRADER_TEMPLATE | llm_instance | JsonOutputParser(),
        "rag_generator": RAG_GENERATION_TEMPLATE | llm_instance | StrOutputParser(),
        "hallucination_grader": HALLUCINATION_GRADER_TEMPLATE | llm_instance | JsonOutputParser(),
        "answer_grader": USEFULNESS_GRADER_TEMPLATE | llm_instance | JsonOutputParser()
    }
 
def build_rag_graph(chains: Dict, retriever_instance, reranker_instance, web_search_tool_instance) -> 'StateGraph':
    from langgraph.graph import StateGraph, END # Import here
 
    #Node and Decision Function Definitions
    def route_question_node(state: GraphState) -> Dict[str, Any]:
        """Routes the question to 'vectorstore' or 'websearch'."""
        print("---NODE: Route Question---")
        question = state["question"]
        web_search_is_enabled = state.get("web_search_enabled", True) # Default to True if not present
 
        if not web_search_is_enabled:
            print("Web search is disabled by user. Forcing route to 'vectorstore'.")
            return {"routing_decision": "vectorstore"}
                  
        try:
            route_result = chains["router"].invoke({"question": question})
            decision = route_result.get("datasource", "websearch").lower()
            if decision not in ["vectorstore", "websearch"]:
                print(f"Warning: Router returned invalid decision '{decision}'. Defaulting to websearch.")
                decision = "websearch"
            print(f"Routing decision: '{decision}'")
            return {"routing_decision": decision}
        except Exception as e:
            print(f"Error in route_question_node: {e}. Defaulting to websearch.")
            traceback.print_exc()
            return {"routing_decision": "websearch"}
 
    def decide_initial_path(state: GraphState) -> str:
        """Reads the 'routing_decision' from state to decide the first path."""
        print(f"---CONDITIONAL EDGE: Decide Initial Path---")
        decision = state.get("routing_decision")
        print(f"Decision from state is: '{decision}'")
        if decision == "vectorstore":
            return "vectorstore"
        else: # Covers "websearch" and any error defaults
            return "websearch"
 
    def retrieve_node(state: GraphState) -> Dict[str, Any]:
        print("---NODE: Retrieve---")
        question = state["question"]
        docs = []
        status = "Retrieval not attempted or failed"
        try:
            if retriever_instance:
                # Use the new constant for the number of docs to retrieve
                retriever_instance.search_kwargs = {'k': VECTORSTORE_SEARCH_K}
                docs = retriever_instance.invoke(question)
                status = f"Retrieved {len(docs)} docs" if docs else "No docs retrieved by vectorstore"
            else:
                status = "Retriever not available, skipping retrieval"
        except Exception as e:
            print(f"Retrieval error: {e}")
            status = f"Error during retrieval: {str(e)[:100]}"
        # Initialize/reset the count and return the retrieved docs
        return {"documents": docs, "retrieval_status": status, "relevant_docs_count": 0}
 
    def rerank_node(state: GraphState) -> Dict[str, Any]:
        print("---NODE: Rerank---")
        question = state["question"]
        docs = state.get("documents", [])
        if not docs:
            print("No docs to rerank.")
            return {"documents": []}
        try:
            reranked_docs = reranker_instance.compress_documents(query=question, documents=docs)
            print(f"Reranked from {len(docs)} to {len(reranked_docs)} docs.")
            return {"documents": reranked_docs}
        except Exception as e:
            print(f"Reranking error: {e}")
            return {"documents": docs}
 
    def grade_documents_node(state: GraphState) -> Dict[str, Any]:
        print("---NODE: Grade Documents---")
        question = state["question"]
        docs = state.get("documents", [])
        web_search_is_enabled = state.get("web_search_enabled", True)
        initial_count = len(docs)
        web_search_needed_decision = "yes"
        relevance_status = "Grading skipped (no docs)"
        relevant_docs_list = []
 
        if docs:
            # ... (the for loop for grading remains the same) ...
            graded_count = 0
            for doc_item_grade in docs:
                try:
                    score_res = chains["relevance_grader"].invoke({
                        "question": question,
                        "document": doc_item_grade.page_content
                    })
                    graded_count +=1
                    if score_res.get("score", "no").lower() == "yes":
                        relevant_docs_list.append(doc_item_grade)
                except Exception as e:
                    print(f"Document relevance grading error for one doc: {e}")
 
            if not relevant_docs_list:
                 relevance_status = f"All {graded_count} docs graded irrelevant, or no docs to grade."
                 web_search_needed_decision = "yes"
            else:
                 relevance_status = f"{len(relevant_docs_list)} of {len(docs)} docs relevant"
                 web_search_needed_decision = "no"
        else:
            web_search_needed_decision = "yes"
        
        # Override decision if web search is globally disabled
        if not web_search_is_enabled:
            print("Web search is disabled, overriding 'web_search_needed_decision' to 'no'.")
            web_search_needed_decision = "no"
 
        print(f"Grade documents result: relevant_docs={len(relevant_docs_list)}, web_search_needed='{web_search_needed_decision}'")
        return {
            "documents": relevant_docs_list,
            "relevance_filtering_status": relevance_status,
            "_web_search_decision_from_grade": web_search_needed_decision,
            "relevant_docs_count": len(relevant_docs_list),
            "initial_docs_count": initial_count
        }
 
    def decide_path_after_grading(state: GraphState) -> str:
        decision = state.get("_web_search_decision_from_grade", "yes")
        routing_decision = state.get("routing_decision")
        web_search_status = state.get("web_search_fallback_status")
 
        print(f"---DECISION: Path After Grading. _web_search_decision_from_grade: '{decision}', initial_route: '{routing_decision}', web_search_status: '{web_search_status}' ---")
 
        if decision == "yes":
            if routing_decision == "vectorstore" and web_search_status not in ["Used", "Used (no results)"]:
                print("Decision: Need web search (fallback from vectorstore).")
                return "websearch_as_fallback"
            else:
                print("Decision: Proceed to generate (web search already primary or tried).")
                return "generate"
        else:
            print("Decision: Relevant docs found, proceed to generate.")
            return "generate"
 
    def web_search_node(state: GraphState, is_fallback: bool = False) -> Dict[str, Any]:
        node_name = "Web Search (Fallback)" if is_fallback else "Web Search (Primary)"
        print(f"---NODE: {node_name}---")
        question = state["question"]
        web_docs: List[Document] = []
        current_documents = state.get("documents", [])
        status_message = "Web search not attempted or tool unavailable"
 
        if not web_search_tool_instance:
            print("Web search tool not available.")
            return {"documents": current_documents if is_fallback else [], "web_search_fallback_status": "Skipped (tool unavailable)"}
 
        try:
            web_results_raw = web_search_tool_instance.invoke({"query": question})
            if isinstance(web_results_raw, list):
                web_docs = [
                    Document(page_content=str(res.get("content", "")), metadata={"source": str(res.get("url", "Web Search")), "display_source": str(res.get("url", "Web Search"))})
                    for res in web_results_raw
                ]
                status_message = "Used" if web_docs else "Used (no results)"
            else:
                print(f"Warning: Tavily search returned non-list data: {type(web_results_raw)}")
                status_message = "Used (unexpected result format)"
 
            final_docs = current_documents + web_docs if is_fallback else web_docs
            print(f"Total docs for generate after {node_name}: {len(final_docs)}")
            return {"documents": final_docs, "web_search_fallback_status": status_message}
        except Exception as e:
            print(f"{node_name} error: {e}")
            status_message = f"Error: {str(e)[:100]}"
            return {"documents": current_documents if is_fallback else [], "web_search_fallback_status": status_message}
 
    def primary_web_search_node(state: GraphState) -> Dict[str, Any]:
        return web_search_node(state, is_fallback=False)
 
    def fallback_web_search_node(state: GraphState) -> Dict[str, Any]:
        return web_search_node(state, is_fallback=True)
 
    def generate_node(state: GraphState) -> Dict[str, Any]:
        print("---NODE: Generate---")
        question = state["question"]
        docs = state.get("documents", [])
        generation_text = "I could not find enough information in the provided documents to answer your question."
 
        if docs:
            context_str = format_docs_for_llm(docs)
            try:
                generation_text = chains["rag_generator"].invoke({"context": context_str, "question": question})
            except Exception as e:
                generation_text = f"Error during answer generation: {str(e)[:150]}"
        else:
            print("No documents provided to generate_node.")
 
        return {"generation": generation_text}
 
    def grade_generation_node(state: GraphState) -> Dict[str, Any]:
        print("---NODE: Grade Generation---")
        question = state["question"]
        docs = state.get("documents", [])
        generation = state.get("generation", "")
        web_search_is_enabled = state.get("web_search_enabled", True)
 
        hall_status = "Skipped"
        useful_status = "Skipped"
        routing_key = "end_not_useful"
 
        if not generation or "Error during answer generation" in generation or "could not find enough information" in generation:
            hall_status = "Skipped (problematic generation)"
            useful_status = "Failed (problematic generation)"
            routing_key = "end_not_useful"
        else:
            # Hallucination Check
            if docs:
                hall_res = chains["hallucination_grader"].invoke({"documents": format_docs_for_llm(docs), "generation": generation})
                if hall_res.get("score", "no").lower() == "yes":
                    hall_status = "Passed"
                else:
                    hall_status = "Failed (hallucination)"
                    # Decide if we should try web search after a hallucination
                    if web_search_is_enabled and state.get("routing_decision") == "vectorstore" and state.get("web_search_fallback_status") not in ["Used", "Used (no results)"]:
                        routing_key = "websearch_after_hallucination_fail"
                    else:
                        routing_key = "end_hallucination"
            else:
                 hall_status = "Skipped (no source documents)" # Cannot check if no docs
 
            # Usefulness Check (only if not hallucinating)
            if hall_status in ["Passed", "Skipped (no source documents)"]:
                useful_res = chains["answer_grader"].invoke({"question": question, "generation": generation})
                if useful_res.get("score", "no").lower() == "yes":
                    useful_status = "Passed"
                    routing_key = "end_useful"
                else:
                    useful_status = "Failed (not useful)"
                    # Decide if we should try web search after a useless answer
                    if web_search_is_enabled and state.get("routing_decision") == "vectorstore" and state.get("web_search_fallback_status") not in ["Used", "Used (no results)"]:
                         routing_key = "websearch_after_usefulness_fail"
                    else:
                         routing_key = "end_not_useful"
 
        return {
            "hallucination_check_status": hall_status,
            "usefulness_check_status": useful_status,
            "_last_decision_from_grade_generation": routing_key
        }
 
    def decide_after_grade_generation(state: GraphState) -> str:
        decision = state.get("_last_decision_from_grade_generation", "end_not_useful")
        print(f"---DECISION: After Grade Generation, routing to: {decision}---")
        return decision
 
    # --- Graph Construction ---
    workflow = StateGraph(GraphState)
 
    # Add all nodes to the graph
    workflow.add_node("route_question", route_question_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("websearch_primary", primary_web_search_node)
    workflow.add_node("websearch_fallback", fallback_web_search_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("grade_generation", grade_generation_node)
 
 
    workflow.set_entry_point("route_question")
 
    workflow.add_conditional_edges(
        "route_question",
        decide_initial_path,
        {
            "vectorstore": "retrieve",
            "websearch": "websearch_primary",
        }
    )
 
 
    # Define the rest of the edges for the graph's flow
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "grade_documents")
 
    workflow.add_conditional_edges(
        "grade_documents",
        decide_path_after_grading,
        {
            "websearch_as_fallback": "websearch_fallback",
            "generate": "generate"
        }
    )
 
    workflow.add_edge("websearch_primary", "grade_documents")
    workflow.add_edge("websearch_fallback", "grade_documents")
 
    workflow.add_edge("generate", "grade_generation")
 
    workflow.add_conditional_edges(
        "grade_generation",
        decide_after_grade_generation,
        {
            "end_useful": END,
            "end_not_useful": END,
            "end_hallucination": END,
            "websearch_after_hallucination_fail": "websearch_fallback",
            "websearch_after_usefulness_fail": "websearch_fallback"
        }
    )
 
    print("Compiling the RAG graph with the corrected entry logic.")
    return workflow.compile()
    
# --- Streamlit Application ---
# --- CSS for the Final Hover Panel UI (Wide Layout) ---
st.markdown("""
<style>
/* --- 1. The Main Layout --- */
div[data-testid="stAppViewContainer"] > .main > .block-container {
    max-width: 75%;
    margin: 0 auto;
    padding: 1rem 1rem 0rem 1rem;
}
 
/* --- 2. The Custom Scrollable Chat Area --- */
.chat-area {
    height: 550px;
    overflow-y: auto;
    padding: 10px;
    /* Border removed as requested */
    border-radius: 8px;
    background-color: #ffffff;
    display: flex;
    flex-direction: column-reverse;
}
.chat-area-inner {
    display: flex;
    flex-direction: column;
}
 
/* --- 3. Chat Bubble Styles --- */
.chat-bubble { 
    padding: 10px 15px; border-radius: 18px; margin-bottom: 10px;
    max-width: 85%; word-wrap: break-word;
    box-shadow: 0 1px 2px rgba(0,0,0,0.08); /* Softened shadow */
    line-height: 1.5;
}
.user-bubble { 
    background-color: #007bff; color: white; float: right; clear: both;
    margin-left: 50%; border-bottom-right-radius: 5px; 
}
.bot-container {
    float: left; clear: both; max-width: 50%;
    display: flex;
    align-items: flex-end;
    gap: 8px;
}
.bot-bubble { 
    /* --- MODIFIED & FIXED: Using !important to override Streamlit's defaults --- */
    background-color: #f0f2f5 !important; 
    color: #343a40; 
    border-bottom-left-radius: 5px; 
}
 
/* --- 4. The Pure CSS Hover Effect (unchanged) --- */
.info-icon-container {
    position: relative;
    cursor: pointer;
    flex-shrink: 0;
    margin-bottom: 10px;
}
.insights-panel {
    display: none;
    position: absolute;
    bottom: 0;
    left: 100%;
    margin-left: 12px; 
    width: 450px;
    max-height: 400px;
    overflow-y: auto;
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    padding: 1rem;
    z-index: 100;
    text-align: left;
    color: #212529;
}
.info-icon-container:hover .insights-panel {
    display: block;
}
.insights-panel::after {
    content: "";
    position: absolute;
    top: auto;
    bottom: 10px;
    right: 100%;
    margin-top: -7px;
    border-width: 7px;
    border-style: solid;
    border-color: transparent #f8f9fa transparent transparent;
}
 
/* --- 5. Styling for Panel Content (unchanged) --- */
.insights-panel h6, .insights-panel p, .insights-panel .agent-insight-value,
.insights-panel .styled-table, .insights-panel details {
    /* (This section is unchanged, styles are fine) */
}
 
 
/* --- NEW & FIXED: 6. Style the Streamlit Chat Input --- */
/* This targets the container that holds the input bar at the bottom of the screen */
div[data-testid="stChatInput"] > div {
    /* --- MODIFIED & FIXED: Using !important to ensure the border is applied --- */
    border: 1px solid #d1d5db !important;
    border-radius: 0.5rem !important;
    background-color: #ffffff;
}
 
/* This styles the input when it is focused (clicked on) */
div[data-testid="stChatInput"]:focus-within > div {
    border-color: #007bff !important;
    box-shadow: 0 0 0 1px #007bff;
}
 
/* This targets the actual text area inside the input */
div[data-testid="stChatInput"] textarea {
    /* The default Streamlit input has its own border, we remove it */
    border: none !important;
    background-color: transparent;
    /* Ensure text color is visible */
    color: #333;
}
 
</style>
""", unsafe_allow_html=True)
 
def _get_status_class_for_html(status_str: Optional[str]) -> str:
    # (This function remains unchanged)
    if status_str is None: return "status-neutral"
    s_lower = str(status_str).lower()
    if "passed" in s_lower or "success" in s_lower or s_lower == "used" or "relevant" in s_lower: return "status-success"
    if "failed" in s_lower or "error" in s_lower or "hallucination" in s_lower or "irrelevant" in s_lower: return "status-fail"
    if "skipped" in s_lower or "pending" in s_lower or "warning" in s_lower or \
       "used (no results)" in s_lower or "not useful" in s_lower or "(tool unavailable)" in s_lower: return "status-warn"
    return "status-neutral"
 
def generate_insights_hover_html(final_state: Dict[str, Any]) -> str:
    # (This function remains unchanged)
    if not final_state:
        return ""
    relevant = final_state.get('relevant_docs_count', 0)
    initial = final_state.get('initial_docs_count', 0)
    relevance_percentage = (relevant / initial) * 100 if initial > 0 else 0
    if relevance_percentage >= 75: color, label = "#28a745", "High"
    elif relevance_percentage >= 40: color, label = "#ffc107", "Medium"
    else: color, label = "#dc3545", "Low"
    meter_html = f"""<div style='font-family: sans-serif;font-size: 13px;color: #333;margin-bottom: 1rem;border: 1px solid #e0e0e0;border-radius: 8px;padding: 8px 12px;background-color: #fff;'><div style='display: flex;align-items: center;justify-content: space-between;margin-bottom: 5px;'><span style='font-weight: 600;'>Reliability Meter</span><span style='font-weight: bold;color: {color};'>{label}</span></div><div style='width: 100%;background-color: #e9ecef;border-radius: 10px;height: 10px;overflow: hidden;'><div style='width: {relevance_percentage}%;background-color: {color};height: 10px;border-radius: 10px;'></div></div></div>"""
    status_map = {"Routing": "routing_decision", "Retrieval": "retrieval_status", "Relevance": "relevance_filtering_status", "Web Search": "web_search_fallback_status", "Hallucination": "hallucination_check_status", "Usefulness": "usefulness_check_status"}
    status_list = [f"<p><strong>{lbl}:</strong> <span class='agent-insight-value {_get_status_class_for_html(final_state.get(key))}'>{html.escape(str(final_state.get(key)) if final_state.get(key) is not None else 'N/A')}</span></p>" for lbl, key in status_map.items()]
    status_html = "".join(status_list)
    docs = final_state.get("documents", [])
    docs_html = ""
    if docs:
        rows = [f"<tr><td><a href='{html.escape(d.metadata.get('source', '#'))}' target='_blank'>{html.escape(d.metadata.get('display_source', 'Unknown'))}</a></td><td class='preview-cell'><small>{html.escape(d.page_content[:100].strip().replace(chr(10),' ') + '...')}</small></td></tr>" for d in docs if isinstance(d, Document)]
        docs_html = f"<details><summary>View {len(docs)} documents</summary><table class='styled-table'><thead><tr><th>Source</th><th>Preview</th></tr></thead><tbody>{''.join(rows)}</tbody></table></details>"
    panel_content = f"<h6>Agent Insights</h6>{meter_html}{status_html}{docs_html}"
    info_icon_svg = """<svg xmlns=http://www.w3.org/2000/svg width="20" height="20" fill="currentColor" viewBox="0 0 16 16" style="color: #6c757d;"><path d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16z"/><path d="m8.93 6.588-2.29.287-.082.38.45.083c.294.07.352.176.288.469l-.738 3.468c-.064.293.006.399.287.47l.45.083.082-.38-2.29-.287a.5.5 0 0 1-.498-.5l.738-3.468a.5.5 0 0 1 .498-.469l2.29.287z"/><path d="M8 4.5a1 1 0 1 1-2 0 1 1 0 0 1 2 0z"/></svg>"""
    return f"""<div class="info-icon-container">{info_icon_svg}<div class="insights-panel">{panel_content}</div></div>"""
 
# --- Logo and Title ---
dell_logo_b64, nvidia_logo_b64 = None, None
try:
    if os.path.exists(DELL_LOGO_PATH): dell_logo_b64 = pil_to_base64_encoder(Image.open(DELL_LOGO_PATH))
except Exception as e: st.warning(f"Could not load Dell logo: {e}")
try:
    if os.path.exists(NVIDIA_LOGO_PATH): nvidia_logo_b64 = pil_to_base64_encoder(Image.open(NVIDIA_LOGO_PATH))
except Exception as e: st.warning(f"Could not load NVIDIA logo: {e}")
 
col1, col2 = st.columns([3, 1])
 
with col1:
    header_html_string = '<div style="display: flex; align-items: center; padding-bottom: 10px; margin-top: 14px;">'
    if dell_logo_b64: header_html_string += f'<img src="data:image/png;base64,{dell_logo_b64}" width="50" style="margin-right: 10px;">'
    if nvidia_logo_b64: header_html_string += f'<img src="data:image/png;base64,{nvidia_logo_b64}" width="170">'
    header_html_string += '</div>'
    if dell_logo_b64 or nvidia_logo_b64: st.markdown(header_html_string, unsafe_allow_html=True)
    st.title(APP_TITLE)
 
with col2:
    st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True) 
    st.toggle("Enable Web Search", value=True, key='web_search_enabled', help="When enabled, the agent can use a web search for questions outside its document knowledgebase. When disabled, it will only use the indexed RAG documents.")
 
 
 
# --- Initialize core components ---
if 'components_initialized' not in st.session_state:
    st.session_state.components_initialized = False
if not st.session_state.components_initialized:
    with st.spinner("Initializing AI models and core components... This may take a moment."):
        try:
            if TAVILY_API_KEY_CONFIG and not os.getenv("TAVILY_API_KEY"): os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY_CONFIG
            llm, embeddings_model, reranker_model, search_tool = initialize_llms_and_utils()
            es_client = get_elasticsearch_client()
            if all([llm, embeddings_model, reranker_model, (es_client is not None)]):
                st.session_state.llm, st.session_state.embeddings_model, st.session_state.reranker_model, st.session_state.search_tool = llm, embeddings_model, reranker_model, search_tool
                st.session_state.es_client, st.session_state.graph_chains = es_client, initialize_graph_chains(llm)
                st.session_state.components_initialized = True; print("Core components initialized successfully.")
                if not search_tool: st.toast("Warning: Web search tool (Tavily) is not available.", icon="⚠️")
            else:
                missing = [comp for comp, present in [("LLM", llm), ("Embeddings", embeddings_model), ("Reranker", reranker_model), ("ES Client", es_client is not None)] if not present]
                st.error(f"CRITICAL: Core components failed: {', '.join(missing)}. App cannot proceed."); st.stop()
        except Exception as e_init: st.error(f"CRITICAL: Init exception: {e_init}"); traceback.print_exc(); st.stop()
 
llm, embeddings_model, reranker_model = st.session_state.llm, st.session_state.embeddings_model, st.session_state.reranker_model
search_tool, es_client, graph_chains = st.session_state.search_tool, st.session_state.es_client, st.session_state.graph_chains
 
# --- Load, process, and index documents ---
if 'documents_processed_and_indexed' not in st.session_state:
    st.session_state.documents_processed_and_indexed = False; st.session_state.retriever = None
if not st.session_state.documents_processed_and_indexed:
    with st.spinner("Loading, processing, and indexing documents..."):
        try:
            processed_docs = load_and_process_documents(streamlit_feedback=True)
            doc_splits = split_documents(processed_docs) if processed_docs else []
            if es_client:
                if doc_splits:
                    _, st.session_state.retriever = get_vectorstore_and_retriever(embeddings_model, es_client, doc_splits)
                    if st.session_state.retriever: print("Retriever from new docs."); st.toast("Docs indexed!", icon="✅")
                    else: st.warning("Failed to create retriever from new docs.")
                elif es_client.indices.exists(index=ES_INDEX_NAME):
                    try:
                        st.session_state.retriever = ElasticsearchStore(index_name=ES_INDEX_NAME, embedding=embeddings_model, es_connection=es_client, strategy=ElasticsearchStore.ExactRetrievalStrategy()).as_retriever()
                        print("Retriever from existing index."); st.toast("Using existing index.", icon="ℹ️")
                    except Exception as e: print(f"Err using existing index: {e}"); st.warning("Failed to use existing index.")
                else: st.warning("No docs, no existing index. Search unavailable.")
            else: st.error("ES client unavailable. Indexing disabled."); st.session_state.retriever = None
            st.session_state.documents_processed_and_indexed = True
        except Exception as e: st.error(f"Doc processing error: {e}"); traceback.print_exc(); st.session_state.retriever = None; st.session_state.documents_processed_and_indexed = True
 
retriever = st.session_state.retriever
if retriever is None:
    class DummyRetriever:
        def invoke(self, q: str, **kwargs) -> List[Document]: return []
        async def ainvoke(self, q: str, **kwargs) -> List[Document]: return []
    if 'dummy_retriever_warning_shown' not in st.session_state: st.toast("Retriever unavailable. Search disabled.", icon="⚠️"); st.session_state.dummy_retriever_warning_shown = True
    retriever = DummyRetriever()
 
# --- Build RAG graph ---
if 'rag_graph_app' not in st.session_state:
    if not all([graph_chains, retriever, reranker_model]): st.error("Cannot build RAG: Missing components."); st.stop()
    with st.spinner("Building RAG agent graph..."):
        try:
            st.session_state.rag_graph_app = build_rag_graph(graph_chains, retriever, reranker_model, search_tool)
            print("RAG graph built."); st.toast("Agent ready!", icon="🧠")
        except Exception as e: st.error(f"Failed to build RAG graph: {e}"); traceback.print_exc(); st.stop()
rag_graph_app = st.session_state.rag_graph_app
if not rag_graph_app: st.error("RAG graph unavailable."); st.stop()
 
 
# --- Initialize Session State for Chat ---
if 'history' not in st.session_state: st.session_state.history = []
 
# --- Chat Input Logic ---
if user_question_input := st.chat_input("Ask the finance assistant..."):
    st.session_state.history.append({"question": user_question_input, "answer": "⏳ Processing..."}) # Changed from "Thinking" for clarity
    st.rerun()
 
# --- Map for friendly status messages ---
NODE_TO_STATUS_MAP = {
    "route_question": "Routing question...",
    "retrieve": "Retrieving documents...",
    "rerank": "Reranking results...",
    "grade_documents": "Grading document relevance...",
    "websearch_primary": "Searching the web...",
    "websearch_fallback": "Using web search as fallback...",
    "generate": "Generating answer...",
    "grade_generation": "Finalizing response..." # This is the last check
}
 
 
# --- Main UI Rendering and Processing Section ---
html_parts = []
processing_message_index = -1
 
for i, chat_item in enumerate(st.session_state.history):
    # Part 1: Render User Bubble
    user_q_escaped = html.escape(str(chat_item.get("question", "")))
    html_parts.append(f'<div class="chat-bubble user-bubble">{user_q_escaped}</div>')
 
    # Part 2: Handle Bot Bubble (Statically or find the one to process)
    if chat_item.get("answer") != "⏳ Processing...":
        bot_answer_html = str(chat_item.get("answer", ""))
        insights_html = generate_insights_hover_html(chat_item.get("final_state"))
        html_parts.append(f'<div class="bot-container"><div class="chat-bubble bot-bubble">{bot_answer_html}</div>{insights_html}</div>')
    else:
        # Found the message to process, save its index and stop.
        processing_message_index = i
        break
 
# Display the static part of the chat history
chat_history_html = "".join(html_parts)
st.markdown(f'<div class="chat-area"><div class="chat-area-inner">{chat_history_html}</div></div>', unsafe_allow_html=True)
 
# --- Part 3: Live Streaming Logic with Status Updates ---
if processing_message_index != -1:
    
    live_bubble_placeholder = st.empty()
    
    # We don't set an initial message; the first status from the loop will be the first thing shown.
    
    user_question = st.session_state.history[processing_message_index]["question"]
    final_answer = "An error occurred during processing."
    final_state = {}
 
    try:
        initial_state = {
            "question": user_question,
            "documents": [],
            "web_search_enabled": st.session_state.get('web_search_enabled', True)
        }
        
        # --- GRAPH EXECUTION WITH LIVE STATUS UPDATES ---
        accumulated_state = initial_state.copy()
        for event in rag_graph_app.stream(initial_state, {"recursion_limit": 25}):
            if not event: continue
            node_name = list(event.keys())[0]
            
            # ** NEW: Update the bubble with the current step's status **
            status_message = NODE_TO_STATUS_MAP.get(node_name, f"⚙️ Processing: {node_name}...")
            live_bubble_placeholder.markdown(
                f'<div class="bot-container"><div class="chat-bubble bot-bubble">{status_message}</div></div>',
                unsafe_allow_html=True
            )
            
            # Continue accumulating state as before
            if isinstance(event[node_name], dict):
                accumulated_state.update(event[node_name])
        final_state = accumulated_state
 
        # --- LIVE TEXT GENERATION (overwrites the last status) ---
        context_str = format_docs_for_llm(final_state.get("documents", []))
        if not context_str:
            final_answer = "I could not find any relevant information to answer your question."
            live_bubble_placeholder.markdown(
                f'<div class="bot-container"><div class="chat-bubble bot-bubble">{final_answer}</div></div>', 
                unsafe_allow_html=True
            )
        else:
            generator_chain = graph_chains["rag_generator"]
            generation_stream = generator_chain.stream({"context": context_str, "question": user_question})
            
            full_response = ""
            for chunk in generation_stream:
                full_response += chunk
                live_bubble_placeholder.markdown(
                    f'<div class="bot-container"><div class="chat-bubble bot-bubble">{full_response}▌</div></div>', 
                    unsafe_allow_html=True
                )
            
            live_bubble_placeholder.markdown(
                f'<div class="bot-container"><div class="chat-bubble bot-bubble">{full_response}</div></div>', 
                unsafe_allow_html=True
            )
            final_answer = full_response
 
    except Exception as e:
        final_answer = f"An error occurred: {html.escape(str(e)[:150])}"
        final_state['error'] = traceback.format_exc()
        live_bubble_placeholder.markdown(
            f'<div class="bot-container"><div class="chat-bubble bot-bubble" style="color: red;">{final_answer}</div></div>',
            unsafe_allow_html=True
        )
 
    # --- Final Step: Update History and Rerun ---
    st.session_state.history[processing_message_index]["answer"] = final_answer
    st.session_state.history[processing_message_index]["final_state"] = final_state
    
    st.rerun()
