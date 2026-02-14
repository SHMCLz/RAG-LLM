import os
import logging
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 设置 HuggingFace 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

load_dotenv()
logger = logging.getLogger(__name__)

api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com")
use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

logger.info(f"配置: use_ollama={use_ollama}, api_base={api_base}")

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    logger.info(f"✓ HuggingFace embeddings ({device})")
except Exception as e:
    logger.error(f"✗ Embeddings 失败: {e}")
    embeddings = None

prompt_template = ChatPromptTemplate.from_template(
    """Expert Research Assistant Guidelines:

1. Source Accuracy:
   - Strictly use ONLY the provided context
   - For missing info: "The article doesn't specify"
   - Never hallucinate facts

2. Response Structure:
   - Core Answer (1 bolded sentence)
   - Key Evidence (3-5 bullet points max)
   - Practical Implications (when relevant)
   - Limitations (if data is incomplete)

3. Technical Content:
   - Code: ```python\n...\n``` 
   - Formulas: $E=mc^2$ format
   - Terms: "API (Application Programming Interface)"

4. Language Rules:
   - Match question's language
   - Auto-correct grammar subtly
   - Use ISO standards for dates/units

Context:
{context}

Question: {question}"""
)

if use_ollama:
    from langchain_ollama import OllamaLLM
    llm = OllamaLLM(model="qwen2.5:7b", base_url="http://localhost:11434", temperature=0.3)
    logger.info("✓ Ollama LLM")
else:
    MODEL_NAME = "deepseek-chat" if "deepseek" in api_base.lower() else "gpt-4o-mini"
    logger.info(f"初始化: model={MODEL_NAME}, base_url={api_base}")
    llm = ChatOpenAI(model=MODEL_NAME, api_key=api_key, base_url=api_base, temperature=0.3)
    logger.info(f"✓ LLM: {llm.model_name}")


def get_vector_store():
    try:
        logger.info("Loading FAISS vector store...")
        if os.path.exists("./historical_knowledge/index.faiss"):
            vector_store = FAISS.load_local(
                folder_path="./historical_knowledge",
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("✓ Historical knowledge loaded")
            return vector_store
        if os.path.exists("./faiss_index/index.faiss"):
            vector_store = FAISS.load_local(
                folder_path="./faiss_index",
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("✓ Vector store loaded")
            return vector_store
        logger.warning("⚠ No vector store found")
        return None
    except Exception as e:
        logger.error(f"✗ Vector store failed: {str(e)}")
        return None


def answer(question: str) -> str:
    try:
        if not llm:
            return "❌ 系统错误: 语言模型未初始化"
        logger.info(f"Processing: '{question}'")
        vector_store = get_vector_store()
        if not vector_store:
            return "❌ 系统错误: 知识库不可用"
        retrieved_docs = vector_store.similarity_search(question, k=4)
        if not retrieved_docs:
            logger.warning("No relevant documents")
            return "❌ 在知识库中未找到相关信息。"
        docs_content = "\n\n---\n\n".join([
            f"Document {i+1}:\n{doc.page_content}" 
            for i, doc in enumerate(retrieved_docs)
        ])
        formatted_prompt = prompt_template.format(question=question, context=docs_content)
        logger.info("Generating answer...")
        if use_ollama:
            response = llm.invoke(formatted_prompt)
            return response if isinstance(response, str) else str(response)
        else:
            response = llm.invoke(formatted_prompt)
            return response.content
    except Exception as e:
        error_msg = f"处理问题时出错: {str(e)}"
        logger.exception(error_msg)
        return f"❌ {error_msg}"
