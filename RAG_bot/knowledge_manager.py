"""
# 先尝试解析 PDF 文本（基于内置文本层）===>对于大文件来说PyPDFLoader时间消耗大
知识库管理模块 - 管理历史知识库和实时知识库
支持大文件（500MB+）快速处理
"""
import os
import re

os.environ['TESSDATA_PREFIX'] = '../Tesseract/tessdata/'
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()
import logging

# 配置基础日志（输出到控制台）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


def with_subdirs_crawling(func):
    """装饰器：为_process_url添加子目录爬取功能"""
    from functools import wraps

    @wraps(func)
    def wrapper(self, url: str, *args, **kwargs):
        # 检查是否启用子目录爬取
        crawl_subdirs = kwargs.pop('crawl_subdirs', False)
        max_subdirs = kwargs.pop('max_subdirs', 5)

        if not crawl_subdirs:
            # 调用原函数
            return func(self, url, *args, **kwargs)

        # 获取子目录并处理
        import requests
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin, urlparse

        all_docs = []

        # 获取子链接
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.text, 'html.parser')

            urls_to_crawl = [url]
            base_domain = urlparse(url).netloc

            # 收集子链接
            for link in soup.find_all('a', href=True):
                if len(urls_to_crawl) >= max_subdirs + 1:  # +1 包含主页面
                    break

                href = link['href']
                full_url = urljoin(url, href)
                parsed = urlparse(full_url)

                if (parsed.netloc == base_domain and
                        full_url not in urls_to_crawl):
                    urls_to_crawl.append(full_url)

            # 处理所有URL
            for target_url in urls_to_crawl:
                try:
                    docs = func(self, target_url, *args, **kwargs)
                    all_docs.extend(docs)
                except Exception as e:
                    logger.error(f"处理 {target_url} 失败: {e}")

            return all_docs

        except Exception as e:
            logger.error(f"爬取子目录失败，回退到单页面: {e}")
            return func(self, url, *args, **kwargs)

    return wrapper


class KnowledgeConfig:
    """知识库配置 - 支持大文件处理"""

    # 文件大小限制（字节）
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

    # 分块配置（适当调大，减少总块数，适合大文件）
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # 批处理配置（调大 batch，减少与 embedding 服务的往返次数）
    BATCH_SIZE = 300          # 每批处理的文本块数（FAISS 添加文档时的批量）
    EMBEDDING_BATCH_SIZE = 128  # embedding 批处理大小（传给 OpenAIEmbeddings）

    # 性能优化
    USE_MULTIPROCESSING = False  # 是否使用多进程（需要额外配置）
    MAX_WORKERS = 4  # 最大工作进程数


class KnowledgeManager:

    """管理历史知识库和实时知识库 - 支持大文件快速处理"""
    def __init__(self, config: KnowledgeConfig = None):
        self.historical_index_dir = "./historical_knowledge"
        self.realtime_index_dir = "./realtime_knowledge"
        self.metadata_file = "./knowledge_metadata.json"

        # 配置
        self.config = config or KnowledgeConfig()

        # 创建目录
        os.makedirs(self.historical_index_dir, exist_ok=True)
        os.makedirs(self.realtime_index_dir, exist_ok=True)

        # 初始化 embeddings
        self.embeddings = self._init_embeddings()

        # 加载元数据
        self.metadata = self._load_metadata()

        logger.info(f"知识库管理器初始化完成")
        logger.info(f"最大文件大小: {self.config.MAX_FILE_SIZE / (1024*1024):.0f} MB")
        logger.info(f"分块大小: {self.config.CHUNK_SIZE}")

    def _init_embeddings(self):
        """初始化 embedding 模型 - 始终使用本地 HuggingFace（DeepSeek 不支持 embeddings）"""
        
        # 保存配置到元数据，确保检索时使用相同的模型
        embedding_config = {
            "model_type": "huggingface",
            "model_name": "paraphrase-multilingual-MiniLM-L12-v2"
        }

        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            import torch
            
            # 自动检测 GPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"✓ 检测到设备: {device}")

            # 使用支持中文的多语言模型
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
            )
            embedding_config["device"] = device
            logger.info(f"✓ 使用 HuggingFace embeddings (本地, {device}, 支持中文)")

            # 保存配置
            self._save_embedding_config(embedding_config)
            return embeddings
            
        except Exception as e:
            logger.error(f"✗ HuggingFace embeddings 加载失败: {e}")
            logger.error("请安装依赖: pip install sentence-transformers torch")
            raise RuntimeError("无法初始化 embeddings 模型")

    def _save_embedding_config(self, config: Dict):
        """保存 embedding 配置"""
        config_file = "./embedding_config.json"
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            logger.info(f"Embedding 配置已保存: {config['model_name']}")
        except Exception as e:
            logger.warning(f"保存 embedding 配置失败: {e}")

    def _load_embedding_config(self) -> Optional[Dict]:
        """加载 embedding 配置"""
        config_file = "./embedding_config.json"
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载 embedding 配置失败: {e}")
        return None

    def _load_metadata(self) -> Dict:
        """加载知识库元数据"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载元数据失败: {e}")

        return {
            "historical": {
                "sources": [],
                "total_chunks": 0,
                "last_updated": None
            },
            "realtime": {
                "sources": [],
                "total_chunks": 0,
                "last_updated": None
            }
        }

    def _save_metadata(self):
        """保存知识库元数据"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存元数据失败: {e}")

    def _process_file(self, file_path: str) -> List[Document]:
        """处理单个文件并返回文档列表"""
        file_ext = file_path.split('.')[-1].lower()

        # 检查文件大小
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        logger.info(f"文件大小: {file_size_mb:.2f} MB")

        if file_ext == 'pdf':
            logger.info(f"加载 PDF: {file_path}")
            # 先尝试解析 PDF 文本（基于内置文本层）===>对于大文件来说PyPDFLoader时间消耗大
            loader = PyMuPDFLoader(file_path)

            docs = loader.load()

            total_text_len = sum(len(d.page_content) for d in docs)
            # docs是按照文件页数进行分割了，双栏可以顺序准确
            """for d in docs:
                print("===========================")
                print(d.page_content)"""
            logger.info(f"PDF 文本总长度: {total_text_len} 字符，页数: {len(docs)}")

            # 如果几乎没有提取到文本，很可能是扫描件，尝试自动 OCR
            if total_text_len < 1000:
                logger.warning(
                    "当前 PDF 几乎没有可提取文本，可能是扫描件或图片 PDF，尝试使用 OCR 自动识别..."
                )
                try:
                    # 延迟导入，只有在需要 OCR 时才依赖这些库
                    from pdf2image import convert_from_path
                    import pytesseract

                    ocr_docs: List[Document] = []
                    # DPI 不宜太高，否则 2000 页会非常慢，这里取适中值
                    pages = convert_from_path(file_path, dpi=200)
                    print("over1")
                    logger.info(f"OCR: 共 {len(pages)} 页，将逐页识别文本（可能较慢）...")
                    # 11.20开始的
                    for idx, page in enumerate(pages, start=1):
                        text = pytesseract.image_to_string(page, lang="chi_sim+eng")
                        text = text.strip()
                        if text:
                            ocr_docs.append(
                                Document(
                                    page_content=text,
                                    metadata={
                                        "source": file_path,
                                        "page": idx,
                                        "ocr": True
                                    }
                                )
                            )
                        if idx % 50 == 0:
                            logger.info(f"OCR 进度: 已处理 {idx}/{len(pages)} 页，当前累计文本页数: {len(ocr_docs)}")

                    ocr_total_len = sum(len(d.page_content) for d in ocr_docs)
                    logger.info(
                        f"OCR 完成，共生成 {len(ocr_docs)} 个文本页，总长度约 {ocr_total_len} 字符"
                    )

                    # 如果 OCR 成功提取到一定量的文本，则优先使用 OCR 结果
                    if ocr_total_len > 0:
                        return ocr_docs
                    else:
                        logger.warning("OCR 未能提取到有效文本，将返回原始（几乎为空）的解析结果。")
                except ImportError as e:
                    logger.error(
                        f"OCR 所需依赖未安装（pdf2image / pytesseract），"
                        f"请先安装后重试。错误信息: {e}"
                    )
                except Exception as e:
                    logger.error(f"OCR 处理 PDF 失败，将退回原始解析结果。错误: {e}")

            # 默认返回基于文本层的解析结果
            return docs
        elif file_ext == 'txt':
            logger.info(f"加载 TXT: {file_path}")
            loader = TextLoader(file_path, encoding='utf-8', autodetect_encoding=True)
            return loader.load()
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")


    @with_subdirs_crawling
    def _process_url(self, url: str) -> List[Document]:
        """原函数保持不变"""
        logger.info(f"加载 URL: {url}")
        loader = WebBaseLoader(
            web_paths=(url,),
            requests_kwargs={
                "headers": {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
            }
        )
        return loader.load()

    def _process_url_list(self, url_list_file: str) -> List[Document]:
        """处理 URL 列表文件"""
        logger.info(f"加载 URL 列表: {url_list_file}")
        all_docs = []

        with open(url_list_file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip() and line.strip().startswith('http')]

        for url in urls:
            try:
                docs = self._process_url(url)
                all_docs.extend(docs)
                logger.info(f"成功加载: {url}")
            except Exception as e:
                logger.error(f"加载 URL 失败 {url}: {e}")

        return all_docs

    def _split_academic_documents(self, docs: List[Document], chunk_size: int = None) -> List[Document]:
        """专门处理学术论文的分割 - 保持逻辑结构"""

        # 默认配置
        if chunk_size is None:
            chunk_size = self.config.CHUNK_SIZE

        chunk_overlap = self.config.CHUNK_OVERLAP

        # 对每个文档（每页）单独处理
        all_splits = []

        for doc in docs:
            text = doc.page_content
            metadata = doc.metadata.copy()

            # 检测是否为学术论文
            if self._is_academic_paper(text):
                # 使用学术论文专用分割
                page_splits = self._split_academic_text(text, chunk_size, chunk_overlap, metadata)
            else:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", "。", ". ", " ", ""]
                )
                page_splits = text_splitter.split_text(text)
                page_splits = [
                    Document(page_content=split, metadata=metadata)
                    for split in page_splits
                ]

            all_splits.extend(page_splits)

        #self._save_to_json(all_splits, 'lunwen.json')

        logger.info(f"✓ 学术论文分割完成: {len(all_splits)} 个文本块")
        return all_splits

    def _is_academic_paper(self, text: str) -> bool:
        """检测是否为学术论文"""
        indicators = [
            "摘要：", "关键词：", "中图分类号：", "文献标志码：", "文章编号：",
            "Abstract:", "Keywords:", "DOI:", "引用格式：",
            "收稿日期：", "基金项目：", "作者简介：",
            "1  ", "2  ", "1.1  ", "2.1  ",  # 章节编号
            "表", "图", "参考文献", "Reference"
        ]

        # 检查前2000字符中是否有学术论文特征
        sample = text[:2000]
        return any(indicator in sample for indicator in indicators)

    def _split_academic_text(self, text: str, chunk_size: int, chunk_overlap: int, metadata: dict) -> List[Document]:
        """按学术论文结构分割"""

        # 步骤1：识别并标记论文的不同部分
        sections = self._identify_academic_sections(text)

        # 步骤2：根据不同部分采用不同的分割策略
        chunks = []
        for section_type, section_text in sections:
            if section_type == "title_author_info":
                # 标题作者信息 - 保持完整
                chunks.append(Document(
                    page_content=section_text,
                    metadata={**metadata, "section": "title_author_info"}
                ))

            elif section_type == "abstract":
                # 摘要 - 保持完整（中英文摘要分开）
                abstracts = self._split_abstract(section_text, chunk_size)
                for i, abstract in enumerate(abstracts):
                    chunks.append(Document(
                        page_content=abstract,
                        metadata={**metadata, "section": "abstract", "language": abstract[:100]}
                    ))

            elif section_type == "main_content":
                # 正文内容 - 按章节分割
                content_chunks = self._split_main_content(section_text, chunk_size, chunk_overlap)
                for chunk in content_chunks:
                    chunks.append(Document(
                        page_content=chunk,
                        metadata={**metadata, "section": "main_content"}
                    ))

            elif section_type == "references":
                # 参考文献 - 每个文献单独或按组分割
                ref_chunks = self._split_references(section_text, chunk_size)
                for i, ref_chunk in enumerate(ref_chunks):
                    chunks.append(Document(
                        page_content=ref_chunk,
                        metadata={**metadata, "section": "references", "ref_count": len(ref_chunks)}
                    ))

            elif section_type == "tables_figures":
                # 表格和图片 - 保持完整
                chunks.append(Document(
                    page_content=section_text,
                    metadata={**metadata, "section": "tables_figures"}
                ))

        return chunks

    def _identify_academic_sections(self, text: str) -> List[Tuple[str, str]]:
        """识别学术论文的不同部分"""
        sections = []

        # 使用正则表达式匹配不同部分
        import re

        # 1. 标题和作者信息（从开头到摘要之前）
        abstract_start = re.search(r'摘要[：:]|Abstract:', text)
        if abstract_start:
            title_section = text[:abstract_start.start()]
            sections.append(("title_author_info", title_section))
            text = text[abstract_start.start():]
        else:
            # 如果没有明确摘要，前500字符作为标题信息
            sections.append(("title_author_info", text[:500]))
            text = text[500:]

        # 2. 摘要部分（中英文）
        abstract_pattern = r'(摘要[：:].*?)(?=关键词|中图分类号|1\s|引言|Key words|Keywords|1\s|Introduction)'
        abstract_match = re.search(abstract_pattern, text, re.DOTALL)
        if abstract_match:
            sections.append(("abstract", abstract_match.group(1)))
            text = text[abstract_match.end():]

        # 3. 关键词部分
        keywords_pattern = r'(关键词[：:].*?)(?=中图分类号|Abstract|1\s|引言|Introduction)'
        keywords_match = re.search(keywords_pattern, text, re.DOTALL)
        if keywords_match:
            # 合并到摘要部分或单独处理
            text = text[keywords_match.end():]

        # 4. 英文摘要（如果有）
        english_abstract_pattern = r'(Abstract:.*?)(?=Key words|Keywords|1\s|引言|Introduction)'
        eng_abstract_match = re.search(english_abstract_pattern, text, re.DOTALL | re.IGNORECASE)
        if eng_abstract_match:
            sections.append(("abstract", eng_abstract_match.group(1)))
            text = text[eng_abstract_match.end():]

        # 5. 英文关键词
        eng_keywords_pattern = r'(Key words:.*?|Keywords:.*?)(?=1\s|引言|Introduction)'
        eng_keywords_match = re.search(eng_keywords_pattern, text, re.DOTALL | re.IGNORECASE)
        if eng_keywords_match:
            text = text[eng_keywords_match.end():]

        # 6. 正文内容（从编号1或引言开始到参考文献之前）
        ref_start = re.search(r'参考文献|References|Reference', text)
        if ref_start:
            main_content = text[:ref_start.start()]
            sections.append(("main_content", main_content))
            text = text[ref_start.start():]
        else:
            sections.append(("main_content", text))
            text = ""

        # 7. 参考文献
        if text:
            sections.append(("references", text))

        return sections

    def _split_main_content(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """分割正文内容，优先按章节分割"""

        # 按章节编号分割（1, 1.1, 2, 2.1等）
        import re

        # 匹配章节标题：数字开头，可能带小数点，后面有空格或标点
        chapter_pattern = r'\n(\d+(?:\.\d+)*)\s+[^\n]+'
        chapter_matches = list(re.finditer(chapter_pattern, text))

        if len(chapter_matches) >= 2:
            # 按章节分割
            chunks = []
            for i in range(len(chapter_matches)):
                start = chapter_matches[i].start()
                end = chapter_matches[i + 1].start() if i + 1 < len(chapter_matches) else len(text)
                chapter_text = text[start:end].strip()

                if len(chapter_text) > chunk_size:
                    # 章节太大，进一步分割
                    sub_chunks = self._split_large_paragraph(chapter_text, chunk_size, chunk_overlap)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(chapter_text)

            return chunks

        else:
            # 没有明显章节，使用普通分割但保持段落完整
            return self._split_with_paragraph_preservation(text, chunk_size, chunk_overlap)

    def _split_with_paragraph_preservation(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """分割时保持段落完整性"""

        # 先按段落分割
        paragraphs = re.split(r'\n\s*\n', text)

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 如果当前块加上新段落会超过限制，且段落本身不太大
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para
            else:
                # 保存当前块
                if current_chunk:
                    chunks.append(current_chunk)

                if len(para) <= chunk_size:
                    current_chunk = para
                else:
                    # 段落本身太大，需要进一步分割
                    sub_chunks = self._split_paragraph(para, chunk_size)
                    # 最后一个子块作为新的current_chunk的开始
                    if sub_chunks:
                        chunks.extend(sub_chunks[:-1])
                        current_chunk = sub_chunks[-1]

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_paragraph(self, paragraph: str, chunk_size: int) -> List[str]:
        """分割大段落，保持句子完整"""

        # 中英文句子分割
        import re

        # 中英文句子结束标点
        sentence_pattern = r'[。！？.!?]+[”"』』]?\s*'
        sentences = re.split(sentence_pattern, paragraph)
        sentence_delimiters = re.findall(sentence_pattern, paragraph)

        # 重建带标点的句子
        full_sentences = []
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                delimiter = sentence_delimiters[i] if i < len(sentence_delimiters) else ""
                full_sentences.append(sentence + delimiter)

        # 按句子组合成块
        chunks = []
        current_chunk = ""

        for sentence in full_sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_abstract(self, abstract_text: str, chunk_size: int) -> List[str]:
        """分割摘要部分，中英文分开"""

        # 检测中英文摘要
        import re

        if "Abstract:" in abstract_text:
            # 分离中英文摘要
            abstract_split = abstract_text.split("Abstract:", 1)
            chinese_abstract = abstract_split[0].strip()
            english_abstract = "Abstract:" + abstract_split[1].strip()

            chunks = []
            if chinese_abstract and len(chinese_abstract) > 0:
                chunks.append(chinese_abstract)
            if english_abstract and len(english_abstract) > 0:
                chunks.append(english_abstract)

            return chunks
        else:
            return [abstract_text] if len(abstract_text) <= chunk_size else [abstract_text[:chunk_size]]

    def _split_references(self, ref_text: str, chunk_size: int) -> List[str]:
        """分割参考文献，每5-10条文献为一组"""

        # 按文献条目分割
        import re

        # 匹配文献编号：[1], [2] 或 1., 2.
        ref_items = re.split(r'\n\s*(?:\[\d+\]|\d+\.)\s+', ref_text)

        # 清理空条目
        ref_items = [item.strip() for item in ref_items if item.strip()]

        # 分组：每5个文献为一组
        chunks = []
        group_size = 5

        for i in range(0, len(ref_items), group_size):
            group = ref_items[i:i + group_size]
            chunk_text = "\n".join([f"[{j + 1}] {ref}" for j, ref in enumerate(group)])

            # 如果组太大，减少组大小
            if len(chunk_text) > chunk_size:
                # 每个文献单独成块
                for j, ref in enumerate(group):
                    chunks.append(f"[{i + j + 1}] {ref}")
            else:
                chunks.append(chunk_text)

        return chunks

    def _is_book(self, sample: str) -> bool:
        """检测是否为书籍内容"""

        # 书籍特有的特征
        book_indicators = [
            # 1. 目录特征
            "目录", "Contents", "CONTENTS", "目  录",
            "第1章", "第一章", "Chapter 1", "CHAPTER 1",
            "第1节", "第一节", "Section 1", "SECTION 1",

            # 2. 书籍元数据特征
            "开本：",  "丛书名：", "责任编辑：","出版社",

            # 3. 章节结构特征（书籍特有的）
            "前言", "序言", "绪论", "导论",
            "后记", "跋", "索引", "术语表",
            "部", "版"

            # 4. 格式特征
            "版权所有", "Copyright", "©", "All rights reserved",
            "翻印必究", "侵权必究",

            # 5. 多级标题（书籍通常有更深的层级）
            "第[一二三四五六七八九十]+章",
            "第[一二三四五六七八九十]+节",
            "第[0-9]+章",
            "第[0-9]+节",

            # 6. 书籍特有的短语
            "本书", "本教材", "本教程", "本专著",
            "读者", "学习者", "教学使用", "自学",
            "习题", "练习题", "思考题",

            # 7. 版权页特征
            "图书在版编目", "CIP数据", "中国版本图书馆",
            "出版发行", "印刷", "装帧", "印张",
        ]

        # 方法1：直接匹配
        direct_matches = sum(1 for indicator in book_indicators
                             if indicator in sample)

        # 方法2：正则匹配
        import re

        regex_patterns = [
            r'第[一二三四五六七八九十]+章',
            r'第[0-9]+章',
            r'ISBN[:：]\s*\d+[- ]\d+[- ]\d+[- ]\d+[- ]\d',  # ISBN
        ]

        regex_matches = 0
        for pattern in regex_patterns:
            if re.search(pattern, sample, re.MULTILINE):
                regex_matches += 1
        # 反向排除（增加论文特征快速校验，减少依赖）
        paper_indicators = ["摘要", "关键词", "参考文献", "DOI:", "基金项目：", "作者简介："]
        paper_matches = sum(1 for p in paper_indicators if p in sample[:3000])
        if paper_matches >= 2 or self._is_academic_paper(sample):
            return False

        has_book_features = (direct_matches >= 2) or (regex_matches >= 2)
        return has_book_features

    def _get_overlap_from_previous(self, previous_chunk: str, overlap_size: int) -> str:
        """从前一个块获取重叠文本"""
        import re

        if overlap_size <= 0 or len(previous_chunk) < 20:
            return ""

        # 计算重叠长度
        target_overlap = min(overlap_size, len(previous_chunk) // 3)

        # 尝试找到句子边界
        # 从末尾向前搜索句子结束标点
        for i in range(len(previous_chunk) - 1, max(0, len(previous_chunk) - target_overlap - 100), -1):
            if i < len(previous_chunk) and previous_chunk[i] in '。！？.!?':
                overlap_text = previous_chunk[i + 1:].strip()
                if len(overlap_text) >= target_overlap // 2:
                    return overlap_text

        # 如果没找到句子边界，取末尾文本
        overlap_text = previous_chunk[-target_overlap:].strip()

        # 确保不以半个词或标点开始
        if overlap_text and overlap_text[0] in ',，;；.。!！?？':
            overlap_text = overlap_text[1:].strip()

        return overlap_text

    def _split_large_paragraph(self, paragraph: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """分割大段落"""
        import re

        # 按句子分割
        # 中英文句子结束标点
        sentence_pattern = r'([。！？\.\?!]+[」》”"]?\s*)'
        parts = re.split(sentence_pattern, paragraph)

        # 重建完整的句子
        sentences = []
        for i in range(0, len(parts) - 1, 2):
            if i + 1 < len(parts):
                sentence = parts[i] + parts[i + 1]
                if sentence.strip():
                    sentences.append(sentence.strip())

        # 如果按句子分割失败，按长度简单分割
        if len(sentences) <= 1:
            return self._split_by_length_with_overlap(paragraph, chunk_size, chunk_overlap)

        # 按句子组合成块
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if not current_chunk:
                current_chunk = sentence
            elif len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk = current_chunk + " " + sentence
            else:
                # 保存当前块
                chunks.append(current_chunk)

                # 新块开始时添加重叠
                if chunk_overlap > 0 and len(chunks) > 1:
                    overlap_text = self._get_overlap_from_previous(chunks[-2], chunk_overlap)
                    if overlap_text:
                        current_chunk = overlap_text + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_by_length_with_overlap(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """按长度分割文本，并添加重叠"""
        chunks = []
        start = 0

        while start < len(text):
            # 计算当前块的结束位置
            end = start + chunk_size

            # 确保不会切到中间的字（对中文重要）
            if end < len(text):
                # 尝试在标点或空格处分段
                for i in range(min(100, len(text) - end)):
                    if text[end + i] in ' 。！？,，;；\n\t':
                        end += i + 1
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # 更新下一个块的起始位置（考虑重叠）
            start = end - chunk_overlap if chunk_overlap > 0 else end

        return chunks

    def _detect_book_type(self, text: str) -> dict:
        """检测书籍类型"""
        book_info = {'is_book': True, 'book_type': 'unknown'}
        sample = text[:5000]

        if re.search(r'(教材|教程|教科书|教学|习题|练习题|例题|思考题)', sample):
            book_info['book_type'] = 'textbook'
        elif re.search(r'(小说|故事|情节|人物|主角|第[一二三四五六七八九十]+回)', sample):
            book_info['book_type'] = 'novel'
        elif re.search(r'(手册|指南|使用说明|操作指南|安装指南|用户手册)', sample):
            book_info['book_type'] = 'manual'
        elif re.search(r'(词典|辞典|百科全书|大全|参考书|资料)', sample):
            book_info['book_type'] = 'reference'
        elif re.search(r'(研究|学术|理论|方法论|专著|论文集)', sample):
            book_info['book_type'] = 'academic'
        elif re.search(r'(情节|人物|角色|叙事|结局|剧情)', sample) and len(sample.split()) > 200:
            book_info['book_type'] = 'novel'
        elif re.search(r'(医|药)', sample):
            book_info['book_type'] = 'drug'

        return book_info

    def _save_to_json(self, data, save_path):
        """保存数据到JSON文件"""
        # 确保保存的是可序列化的数据
        serializable_data = []
        for item in data:
            if hasattr(item, 'to_dict'):
                serializable_data.append(item.to_dict())
            elif isinstance(item, dict):
                serializable_data.append(item)
            else:
                # 尝试转换为字符串或基本类型
                try:
                    serializable_data.append(str(item))
                except:
                    serializable_data.append(repr(item))

        # 创建目录（如果不存在）
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # 保存到文件
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)

        print(f"数据已保存到: {save_path}")



    """def _split_book_text(self, docs, chunk_size, book_type):
        if book_type == 'textbook':
            result = self._split_textbook(docs, chunk_size)
        elif book_type == 'novel':
            result = self._split_novel(docs, chunk_size)
        elif book_type == 'manual':
            result = self._split_manual(docs, chunk_size)
        elif book_type == 'reference':
            result = self._split_reference(docs, chunk_size)
        elif book_type == 'academic':
            result = self._split_academic(docs, chunk_size)
        elif book_type == 'drug':
            result = self._split_drug_documents(docs, chunk_size)
        else:
            result = self._split_default(docs, chunk_size)


        return result
"""
    def _split_book_text(self, docs: Union[List[Document], Document, str], chunk_size: int, book_type: str) -> List[Document]:
        """根据书籍类型分割文本，返回Document对象列表"""

        # 输入验证
        if not docs:
            logger.warning(f"分割书籍文本: 输入为空，书籍类型={book_type}")
            return []

        try:
            # 将输入标准化为字符串
            text_content = self._normalize_input_to_string(docs)
            metadata = self._extract_metadata(docs)

            if not text_content or not text_content.strip():
                logger.warning(f"分割书籍文本: 文本内容为空，书籍类型={book_type}")
                return []

            logger.info(f"开始分割 {book_type} 类型书籍，文本长度: {len(text_content)}")

            # 根据书籍类型调用不同的分割方法
            if book_type == 'textbook':
                chunks = self._split_textbook(text_content, chunk_size)
            elif book_type == 'novel':
                chunks = self._split_novel(text_content, chunk_size)
            elif book_type == 'manual':
                chunks = self._split_manual(text_content, chunk_size)
            elif book_type == 'reference':
                chunks = self._split_reference(text_content, chunk_size)
            elif book_type == 'academic':
                chunks = self._split_academic(text_content, chunk_size)
            elif book_type == 'drug':
                chunks = self._split_drug_documents(text_content, chunk_size)
            else:
                chunks = self._split_default(text_content, chunk_size)

            # 将字符串块转换为Document对象
            result_docs = []
            for i, chunk in enumerate(chunks):
                if chunk and chunk.strip():
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata.update({
                        "book_type": book_type,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "chunk_size": len(chunk)
                    })
                    result_docs.append(Document(
                        page_content=chunk.strip(),
                        metadata=chunk_metadata
                    ))

            logger.info(f"✓ 书籍分割完成: 类型={book_type}, 块数={len(result_docs)}")
            return result_docs

        except Exception as e:
            logger.error(f"书籍文本分割失败 (类型={book_type}): {str(e)}")
            return []

    def _split_drug_documents(self, text: str, chunk_size: int) -> List[str]:
        """分割药品文档"""
        if not text or not text.strip():
            return []

        try:
            # 药品文档常见结构
            drug_patterns = [
                r'【(药品名称|通用名称|商品名称)】',
                r'【(性状|物理性质)】',
                r'【(鉴别|鉴定)】',
                r'【(检查|检验)】',
                r'【(含量测定|有效成分)】',
                r'【(功能与主治|适应症)】',
                r'【(用法用量)】',
                r'【(不良反应)】',
                r'【(禁忌)】',
                r'【(注意事项)】',
                r'【(贮藏)】',
                r'【(包装)】',
                r'【(有效期)】',
                r'【(批准文号)】',
                r'【(生产企业)】',
            ]

            pattern = '|'.join(drug_patterns)
            sections = re.split(pattern, text)
            chunks = []
            current_chunk = ""
            current_title = ""

            for i, section in enumerate(sections):
                if not section:
                    continue

                section = section.strip()
                if not section:
                    continue

                # 检查是否为标题
                is_title = False
                for p in drug_patterns:
                    if re.match(p, section):
                        is_title = True
                        break

                if is_title:
                    # 保存前一节
                    if current_chunk:
                        if current_title:
                            chunk_content = f"【{current_title}】\n{current_chunk}"
                        else:
                            chunk_content = current_chunk
                        chunks.append(chunk_content.strip())

                    current_title = section
                    current_chunk = ""
                else:
                    if len(current_chunk) + len(section) + 2 <= chunk_size:
                        if current_chunk:
                            current_chunk += "\n" + section
                        else:
                            current_chunk = section
                    else:
                        # 保存当前块
                        if current_chunk:
                            if current_title:
                                chunk_content = f"【{current_title}】\n{current_chunk}"
                            else:
                                chunk_content = current_chunk
                            chunks.append(chunk_content.strip())

                        current_chunk = section

            # 保存最后一块
            if current_chunk:
                if current_title:
                    chunk_content = f"【{current_title}】\n{current_chunk}"
                else:
                    chunk_content = current_chunk
                chunks.append(chunk_content.strip())

            return chunks

        except Exception as e:
            logger.error(f"药品文档分割失败: {str(e)}")
            return self._split_by_length(text, chunk_size)

    def _normalize_input_to_string(self, docs: Union[List[Document], Document, str]) -> str:
        """将各种输入格式统一转换为字符串"""
        if docs is None:
            return ""

        if isinstance(docs, str):
            return docs
        elif isinstance(docs, Document):
            return docs.page_content or ""
        elif isinstance(docs, list):
            # 合并所有文档
            texts = []
            for doc in docs:
                if isinstance(doc, Document):
                    content = doc.page_content or ""
                    if content.strip():
                        texts.append(content)
                elif isinstance(doc, str) and doc.strip():
                    texts.append(doc)

            if not texts:
                return ""

            return "\n\n".join(texts)
        else:
            return str(docs) if docs is not None else ""

    def _extract_metadata(self, docs: Union[List[Document], Document, str]) -> dict:
        """从输入中提取元数据"""
        if isinstance(docs, Document):
            return docs.metadata.copy()
        elif isinstance(docs, list) and docs and isinstance(docs[0], Document):
            # 使用第一个文档的元数据
            return docs[0].metadata.copy()
        else:
            return {}

    def _split_textbook(self, text: str, chunk_size: int) -> List[str]:
        """按章节/小节/练习题分割教科书类文本"""
        if not text or not text.strip():
            return []

        try:
            # 改进的教科书分割模式
            textbook_patterns = [
                r'(第[一二三四五六七八九十百千万]+章\s*[^\n]*)',
                r'(\d+\.\d+(?:\.\d+)*\s+[^\n]*)',  # 1.1, 1.1.1 等
                r'(小节\s*[^\n]*)',
                r'(习题\s*\d+[^\n]*)',
                r'(练习题\s*\d+[^\n]*)',
                r'(例题\s*\d+[^\n]*)',
                r'(本章小结|小结|本章要点)',
                r'(思考题|复习题|讨论题)',
            ]

            # 组合所有模式
            pattern = '|'.join(textbook_patterns)

            # 使用更智能的分割
            sections = re.split(pattern, text)
            chunks = []
            current_chunk = ""

            for i, section in enumerate(sections):
                if not section:
                    continue

                section = section.strip()
                if not section:
                    continue

                # 检查是否是标题
                is_title = False
                for p in textbook_patterns:
                    if re.match(p, section):
                        is_title = True
                        break

                # 如果遇到标题且当前块不为空，开始新块
                if is_title and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = section
                elif len(current_chunk) + len(section) + 2 <= chunk_size:
                    if current_chunk:
                        current_chunk += "\n" + section
                    else:
                        current_chunk = section
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = section

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            return chunks

        except Exception as e:
            logger.error(f"教科书文本分割失败: {str(e)}")
            return self._split_by_length(text, chunk_size)

    def _split_novel(self, text: str, chunk_size: int) -> List[str]:
        """按章节/段落/人物描写/场景分割小说类文本"""
        if not text or not text.strip():
            return []

        try:
            # 更完善的小说分割模式
            novel_patterns = [
                r'(第[一二三四五六七八九十百千万]+章\s*[^\n]*)',
                r'(第[一二三四五六七八九十百千万]+回\s*[^\n]*)',
                r'(人物|角色|主要人物|登场人物)',
                r'(场景|地点|时间|背景)',
                r'(序幕|楔子|引子|开场白)',
                r'(尾声|结局|结尾|后记)',
            ]

            pattern = '|'.join(novel_patterns)
            chapters = re.split(pattern, text)
            chunks = []
            current_chunk = ""

            for chapter in chapters:
                if not chapter:
                    continue

                chapter = chapter.strip()
                if not chapter:
                    continue

                # 检查是否为场景分隔符（如 *** 或 ---）
                if re.match(r'^\s*[\*\-]{3,}\s*$', chapter):
                    # 场景分隔符，开始新块
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    chunks.append(chapter)
                    continue

                if len(current_chunk) + len(chapter) + 2 <= chunk_size:
                    if current_chunk:
                        current_chunk += "\n" + chapter
                    else:
                        current_chunk = chapter
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = chapter

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            return chunks

        except Exception as e:
            logger.error(f"小说文本分割失败: {str(e)}")
            return self._split_by_length(text, chunk_size)

    def _split_manual(self, text: str, chunk_size: int) -> List[str]:
        """按步骤/标题/注意事项/提示分割手册类文本"""
        if not text or not text.strip():
            return []

        try:
            # 更完善的手册分割模式
            manual_patterns = [
                r'(\d+\.\s+[^\n]+)',  # 数字步骤
                r'(Step\s+\d+\s*[:：]?\s*[^\n]*)',  # Step 1
                r'(注意事项|注意|警告|WARNING|CAUTION)',
                r'(提示|技巧|小贴士|TIP|HINT)',
                r'(步骤|操作步骤|使用方法)',
                r'(示例|例子|示例代码|EXAMPLE)',
                r'(故障排除|问题解决|FAQ|常见问题)',
            ]

            pattern = '|'.join(manual_patterns)
            sections = re.split(pattern, text)
            chunks = []
            current_chunk = ""

            for section in sections:
                if not section:
                    continue

                section = section.strip()
                if not section:
                    continue

                # 检查是否为重要标题
                is_important = False
                for p in manual_patterns:
                    if re.match(p, section):
                        is_important = True
                        break

                # 如果是重要标题且当前块不为空，开始新块
                if is_important and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = section
                elif len(current_chunk) + len(section) + 2 <= chunk_size:
                    if current_chunk:
                        current_chunk += "\n" + section
                    else:
                        current_chunk = section
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = section

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            return chunks

        except Exception as e:
            logger.error(f"手册文本分割失败: {str(e)}")
            return self._split_by_length(text, chunk_size)

    def _split_reference(self, text: str, chunk_size: int) -> List[str]:
        """按条目/定义/解释分割参考书类文本"""
        if not text or not text.strip():
            return []

        try:
            # 改进的参考书分割模式
            reference_patterns = [
                r'([A-Z][A-Za-z]+\s*[:：])',  # 术语: 解释
                r'(\d+\.\s+定义|Definition\s+\d+)',
                r'(\d+\.\s+定理|Theorem\s+\d+)',
                r'(\d+\.\s+示例|Example\s+\d+)',
                r'(\d+\.\s+练习|Exercise\s+\d+)',
                r'(参考文献|参考书目|References)',
            ]

            pattern = '|'.join(reference_patterns)
            entries = re.split(pattern, text)
            chunks = []
            current_chunk = ""

            for entry in entries:
                if not entry:
                    continue

                entry = entry.strip()
                if not entry:
                    continue

                # 检查是否为条目开始
                is_entry = False
                for p in reference_patterns:
                    if re.match(p, entry):
                        is_entry = True
                        break

                # 如果是条目开始且当前块不为空，开始新块
                if is_entry and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = entry
                elif len(current_chunk) + len(entry) + 2 <= chunk_size:
                    if current_chunk:
                        current_chunk += "\n" + entry
                    else:
                        current_chunk = entry
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = entry

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            return chunks

        except Exception as e:
            logger.error(f"参考书文本分割失败: {str(e)}")
            return self._split_by_length(text, chunk_size)

    def _split_academic(self, text: str, chunk_size: int) -> List[str]:
        """按章节/引言/摘要/方法/结果等分割学术类文本"""
        if not text or not text.strip():
            return []

        try:
            # 改进的学术论文分割模式
            section_patterns = [
                r'^\s*(摘要|ABSTRACT|Summary)\s*$',
                r'^\s*(关键词|KEYWORDS|Key\s+words)\s*$',
                r'^\s*(引言|INTRODUCTION|前言|绪论)\s*$',
                r'^\s*(方法|METHODOLOGY|实验方法|研究方法)\s*$',
                r'^\s*(结果|RESULTS|实验结果)\s*$',
                r'^\s*(讨论|DISCUSSION|分析与讨论)\s*$',
                r'^\s*(结论|CONCLUSION|总结)\s*$',
                r'^\s*(参考文献|REFERENCES|REFERENCE)\s*$',
                r'^\s*(致谢|ACKNOWLEDGEMENTS)\s*$',
                r'^\s*(附录|APPENDIX)\s*$',
            ]

            # 按行处理
            lines = text.split('\n')
            sections = []
            current_section = []
            current_section_title = ""

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 检查是否为新的章节标题
                is_new_section = False
                section_title = ""

                for pattern in section_patterns:
                    match = re.match(pattern, line, re.IGNORECASE)
                    if match:
                        is_new_section = True
                        section_title = match.group(1)
                        break

                if is_new_section:
                    # 保存前一章节
                    if current_section:
                        section_content = current_section_title + "\n" + "\n".join(
                            current_section) if current_section_title else "\n".join(current_section)
                        sections.append(section_content.strip())

                    # 开始新章节
                    current_section_title = f"【{section_title}】"
                    current_section = []
                else:
                    current_section.append(line)

            # 保存最后一章
            if current_section:
                section_content = current_section_title + "\n" + "\n".join(
                    current_section) if current_section_title else "\n".join(current_section)
                sections.append(section_content.strip())

            # 如果没有检测到章节，按段落分割
            if not sections:
                return self._split_by_paragraphs(text, chunk_size)

            # 对每个章节进一步分割
            chunks = []
            for section in sections:
                if len(section) <= chunk_size:
                    chunks.append(section)
                else:
                    # 按段落分割
                    sub_chunks = self._split_by_paragraphs(section, chunk_size)
                    chunks.extend(sub_chunks)

            return chunks

        except Exception as e:
            logger.error(f"学术文本分割失败: {str(e)}")
            return self._split_by_length(text, chunk_size)


    def _split_by_paragraphs(self, text: str, chunk_size: int) -> List[str]:
        """按段落分割"""
        if not text:
            return []

        paragraphs = text.split('\n')
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                # 空行表示段落分隔
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                continue

            if len(current_chunk) + len(para) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n" + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks


    def _split_by_length(self, text, chunk_size):
        """按长度分割文本的辅助函数"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1  # +1 for space

            if current_length + word_length <= chunk_size:
                current_chunk.append(word)
                current_length += word_length
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


    def _split_default(self, docs, chunk_size):
        """默认的分割方式：按段落分割"""
        paragraphs = docs.split('\n')
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += "\n" + para
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _split_documents(self, docs: List[Document], chunk_size: int = None) -> List[Document]:
        """分割文档为块 - 优化大文件处理，改进中文支持

        对特别大的文档，会自动：
        - 放大 chunk_size，减少总块数
        - 适当降低重叠，减少重复计算
        - 过滤掉内容太少的文本块
        """
        # 默认配置
        if chunk_size is None:
            chunk_size = self.config.CHUNK_SIZE

        chunk_overlap = self.config.CHUNK_OVERLAP

        # 输入验证
        if not docs:
            logger.warning("分割文档: 输入文档列表为空")
            return []

        # 过滤掉内容为空的文档
        valid_docs = []
        for doc in docs:
            if doc and doc.page_content and doc.page_content.strip():
                valid_docs.append(doc)
            else:
                logger.debug(f"跳过空文档或内容为空的文档")

        if not valid_docs:
            logger.warning("分割文档: 所有文档内容为空")
            return []

        # 估算总字符数，用于判断是否为"超大文档"
        total_chars = sum(len(d.page_content) for d in valid_docs)

        # 粗略估算原本的块数
        est_chunks = max(1, total_chars // max(1, chunk_size))

        # 如果预估块数非常多（例如 > 5000），动态调大 chunk_size / 减小 overlap
        if est_chunks > 5000:
            scale = min(3.0, est_chunks / 5000)  # 最多放大到 3 倍
            old_chunk_size = chunk_size
            old_overlap = chunk_overlap

            chunk_size = int(chunk_size * scale)
            # overlap 至少 0，最多不超过 chunk_size / 3
            chunk_overlap = min(int(chunk_overlap / scale), chunk_size // 3)

            logger.info(
                f"检测到超大文档，总字符数约 {total_chars}，原预计块数约 {est_chunks}，"
                f"自动调整分块参数: chunk_size {old_chunk_size} -> {chunk_size}, "
                f"chunk_overlap {old_overlap} -> {chunk_overlap}"
            )

        # 改进的分隔符，更好地支持中文
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # 段落
                "\n",  # 换行
                "。",  # 中文句号
                "！",  # 中文感叹号
                "？",  # 中文问号
                ". ",  # 英文句号
                "! ",  # 英文感叹号
                "? ",  # 英文问号
                "；",  # 中文分号
                "; ",  # 英文分号
                "，",  # 中文逗号
                ", ",  # 英文逗号
                " ",  # 空格
                ""  # 字符
            ]
        )

        logger.info(f"开始分割文档（块大小: {chunk_size}，重叠: {chunk_overlap}，支持中文）...")

        # 获取样本文本用于检测文档类型
        sample_text = ""
        for doc in valid_docs:
            if doc.page_content and len(doc.page_content) > 100:
                sample_text = doc.page_content[:3000]
                break

        if not sample_text:
            # 如果没有足够长的内容，直接使用第一个文档的全部内容
            sample_text = valid_docs[0].page_content[:3000] if valid_docs else ""

        # 检测文档类型并使用相应的分割器
        splits = []
        try:
            if self._is_book(sample_text):
                logger.info("检测到书籍，启用专用分割器...")
                book_info = self._detect_book_type(sample_text)
                logger.info(f"书籍类型: {book_info['book_type']}")

                splits = self._split_book_text(valid_docs, chunk_size, book_info['book_type'])

            elif self._is_academic_paper(sample_text):
                logger.info("检测到学术论文，启用专用分割器...")
                splits = self._split_academic_documents(valid_docs, chunk_size)

            else:
                logger.info("使用通用文档分割器...")
                splits = text_splitter.split_documents(valid_docs)
                logger.info(f"通用文档分割完成: {len(splits)} 个文本块")

        except Exception as e:
            logger.error(f"文档分割失败: {str(e)}")
            # 降级到通用分割器
            splits = text_splitter.split_documents(valid_docs)

        # 过滤掉内容太少的文本块
        filtered_splits = self._filter_small_chunks(splits)
        self._save_to_json(filtered_splits, 'filtered_splits.json')

        logger.info(f"过滤后剩余 {len(filtered_splits)}/{len(splits)} 个文本块")

        return filtered_splits

    def _filter_small_chunks(self, docs: List[Document]) -> List[Document]:
        """过滤掉内容太少的文本块

        过滤规则：
        1. 字符数太少（< 10个字符）
        2. 只有标点符号或空格
        3. 只包含数字或特殊字符
        4. 无效的内容
        """
        if not docs:
            return []

        filtered_docs = []
        removed_count = 0

        # 统计信息
        char_counts = []
        word_counts = []

        for doc in docs:
            if not doc or not doc.page_content:
                removed_count += 1
                continue

            content = doc.page_content.strip()

            # 计算各种指标
            char_count = len(content)
            word_count = len(content.split())

            # 规则1：字符数太少
            if char_count < 15:
                removed_count += 1
                logger.debug(f"过滤块: 字符数太少 ({char_count})")
                continue

            # 规则2：检查是否只有标点符号或空格
            import string
            # 中文标点 + 英文标点 + 空格
            punctuation_chars = "，。！？；：「」『』（）【】《》〈〉、·" + string.punctuation + " \t\n\r"

            # 计算非标点字符的比例
            non_punct_count = sum(1 for c in content if c not in punctuation_chars)
            punct_ratio = 1 - (non_punct_count / max(1, char_count))

            if punct_ratio > 0.8:  # 如果超过80%都是标点
                removed_count += 1
                logger.debug(f"过滤块: 标点符号过多 ({punct_ratio:.2%})")
                continue

            # 规则3：检查是否只包含数字或特殊字符
            # 计算字母、中文、数字的比例
            import re
            # 中文字符
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', content))
            # 英文字母
            english_chars = len(re.findall(r'[a-zA-Z]', content))
            # 数字
            digit_chars = len(re.findall(r'\d', content))

            # 有效字符比例（中文、英文、数字）
            valid_ratio = (chinese_chars + english_chars + digit_chars) / max(1, char_count)

            if valid_ratio < 0.3:  # 如果有效字符少于30%
                removed_count += 1
                logger.debug(f"过滤块: 有效字符过少 ({valid_ratio:.2%})")
                continue

            # 规则4：检查是否为无效内容（例如"待补充"、"暂无"等）
            invalid_keywords = [
                "待补充", "暂无", "暂无内容", "内容待补充",
                "NULL", "null", "None", "none",
                "空", "空白", "无内容", "无",
                "待更新", "更新中", "建设中",
                "To be added", "TBD", "TODO", "待办"
            ]

            content_lower = content.lower()
            if any(keyword in content_lower for keyword in [k.lower() for k in invalid_keywords]):
                removed_count += 1
                logger.debug(f"过滤块: 包含无效关键词")
                continue

            # 规则5：检查是否为重复内容（与前一个块相似）
            if filtered_docs:
                last_content = filtered_docs[-1].page_content.strip()
                # 简单的重复检查：如果当前内容完全包含在前一个内容中
                if content in last_content or last_content in content:
                    # 保留更长的那个
                    if len(content) > len(last_content):
                        filtered_docs[-1] = doc
                    removed_count += 1
                    logger.debug(f"过滤块: 与前一内容重复")
                    continue

            # 通过所有检查，添加到结果
            filtered_docs.append(doc)
            char_counts.append(char_count)
            word_counts.append(word_count)

        # 记录统计信息
        if char_counts:
            avg_chars = sum(char_counts) / len(char_counts)
            avg_words = sum(word_counts) / len(word_counts)
            logger.info(f"文本块统计: 平均 {avg_chars:.1f} 字符, {avg_words:.1f} 词")

        if removed_count > 0:
            logger.info(f"过滤掉 {removed_count} 个内容太少的文本块")

        return filtered_docs

    def _filter_small_chunks_simple(self, docs: List[Document], min_chars: int = 20, min_words: int = 3) -> List[Document]:
        """简单的文本块过滤（备选方法）"""
        if not docs:
            return []

        filtered_docs = []

        for doc in docs:
            if not doc or not doc.page_content:
                continue

            content = doc.page_content.strip()

            # 检查字符数和词数
            char_count = len(content)
            word_count = len(content.split())

            # 如果满足最小要求
            if char_count >= min_chars and word_count >= min_words:
                # 额外的检查：确保不是只有标点符号
                has_content = False
                for char in content:
                    if char.isalnum() or '\u4e00-\u9fff' in char:  # 字母数字或中文
                        has_content = True
                        break

                if has_content:
                    filtered_docs.append(doc)

        return filtered_docs

    def build_historical_knowledge(self, sources: List[str], progress_callback=None) -> Dict:
        """
        批量构建历史知识库

        Args:
            sources: 文件路径或 URL 列表
            progress_callback: 进度回调函数 (current, total, message, details)

        Returns:
            包含处理结果的字典
        """
        logger.info(f"=" * 60)
        logger.info(f"开始构建历史知识库")
        logger.info(f"来源数量: {len(sources)}")
        logger.info(f"=" * 60)

        all_docs = []
        processed_sources = []
        failed_sources = []
        processing_details = []

        for idx, source in enumerate(sources):
            source_name = os.path.basename(source) if not source.startswith('http') else source
            try:
                if progress_callback:
                    progress_callback(
                        idx + 1,
                        len(sources),
                        f"[{idx+1}/{len(sources)}] 加载: {source_name}",
                        {"stage": "loading", "source": source_name}
                    )

                # 处理来源
                if source.startswith(('http://', 'https://')):
                    docs = self._process_url(source)
                    source_type = "url"
                elif source.endswith('.txt'):
                    # 检查是否为 URL 列表
                    try:
                        with open(source, 'r', encoding='utf-8') as f:
                            first_lines = [f.readline().strip() for _ in range(5)]
                        if any(line.startswith('http') for line in first_lines):
                            docs = self._process_url_list(source)
                            source_type = "url_list"
                        else:
                            docs = self._process_file(source)
                            source_type = "file"
                    except:
                        docs = self._process_file(source)
                        source_type = "file"
                else:
                    docs = self._process_file(source)
                    source_type = "file"

                doc_count = len(docs)
                all_docs.extend(docs)

                processed_sources.append({
                    "source": source,
                    "type": source_type,
                    "doc_count": doc_count,
                    "added_at": datetime.now().isoformat()
                })

                detail = f"✓ {source_name}: {doc_count} 个文档"
                processing_details.append(detail)
                logger.info(detail)

            except Exception as e:
                error_msg = str(e)
                logger.error(f"✗ 处理失败 {source_name}: {error_msg}")
                failed_sources.append({
                    "source": source,
                    "error": error_msg
                })
                processing_details.append(f"✗ {source_name}: 失败 - {error_msg}")

        if not all_docs:
            raise ValueError("没有成功加载任何文档")

        # 分割文档
        if progress_callback:
            progress_callback(
                len(sources),
                len(sources),
                "正在分割文档...",
                {"stage": "splitting", "doc_count": len(all_docs)}
            )

        logger.info(f"\n{'='*60}")
        logger.info("开始分割文档...")
        splits = self._split_documents(all_docs)
        logger.info(f"✓ 创建了 {len(splits)} 个文本块")

        # 如果没有任何有效文本块，提前给出明确错误（常见于扫描 PDF）
        if len(splits) == 0:
            error_msg = (
                "未从文档中提取到任何有效文本块，"
                "可能是扫描件或图片 PDF，请先通过 OCR 转成可搜索文本（txt/docx）后再入库。"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 输出前几个文本块，便于检查解析是否正常
        #sample_count = min(3, len(splits))
        #sample_count = len(splits)
        """for i in range(sample_count):
            logger.info(f"示例文本块 {i+1}/{sample_count}：{splits[i].page_content[:300].replace(chr(10), ' ')}...")"""
        processing_details.append(f"\n文本块总数: {len(splits)}")

        # 创建或更新向量存储（支持大批量处理）
        if progress_callback:
            progress_callback(
                len(sources),
                len(sources),
                "正在构建向量索引...",
                {"stage": "vectorizing", "chunk_count": len(splits)}
            )

        logger.info(f"{'='*60}")
        logger.info("开始构建向量存储...")

        # 加载现有索引（如果存在）
        vector_store = None
        existing_chunks = 0
        if os.path.exists(os.path.join(self.historical_index_dir, "index.faiss")):
            try:
                vector_store = FAISS.load_local(
                    self.historical_index_dir,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                existing_chunks = len(vector_store.docstore._dict)
                logger.info(f"✓ 已加载现有历史知识库 ({existing_chunks} 个现有文本块)")
                processing_details.append(f"现有文本块: {existing_chunks}")
            except Exception as e:
                logger.warning(f"⚠ 加载现有索引失败: {e}，将创建新索引")
                vector_store = None

        # 批量处理大量文本块
        batch_size = self.config.BATCH_SIZE
        if len(splits) > batch_size:
            logger.info(f"使用批量处理模式（每批 {batch_size} 个）")

            for i in range(0, len(splits), batch_size):
                batch = splits[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(splits) - 1) // batch_size + 1

                logger.info(f"处理批次 {batch_num}/{total_batches} ({len(batch)} 个文本块)")

                if progress_callback:
                    progress_callback(
                        batch_num,
                        total_batches,
                        f"向量化批次 {batch_num}/{total_batches}",
                        {"stage": "batch_vectorizing", "batch": batch_num, "total": total_batches}
                    )

                if vector_store is None:
                    vector_store = FAISS.from_documents(batch, self.embeddings)
                else:
                    vector_store.add_documents(batch)
        else:
            # 小批量直接处理
            logger.info("直接处理所有文本块...")
            if vector_store is None:
                vector_store = FAISS.from_documents(splits, self.embeddings)
            else:
                vector_store.add_documents(splits)

        # 保存
        logger.info(f"{'='*60}")
        logger.info("保存向量存储...")
        vector_store.save_local(self.historical_index_dir)
        logger.info("✓ 向量存储已保存")

        # 更新元数据
        self.metadata["historical"]["sources"].extend(processed_sources)
        self.metadata["historical"]["total_chunks"] = existing_chunks + len(splits)
        self.metadata["historical"]["last_updated"] = datetime.now().isoformat()
        self._save_metadata()

        # 生成摘要
        summary = {
            "success": True,
            "total_sources": len(sources),
            "processed_sources": len(processed_sources),
            "failed_sources": len(failed_sources),
            "new_chunks": len(splits),
            "existing_chunks": existing_chunks,
            "total_chunks": existing_chunks + len(splits),
            "processing_details": processing_details,
            "details": {
                "processed": processed_sources,
                "failed": failed_sources
            }
        }

        logger.info(f"{'='*60}")
        logger.info("✓ 历史知识库构建完成！")
        logger.info(f"处理来源: {len(processed_sources)}/{len(sources)}")
        logger.info(f"新增文本块: {len(splits)}")
        logger.info(f"总文本块数: {existing_chunks + len(splits)}")
        logger.info(f"{'='*60}")

        return summary

    def add_to_realtime_knowledge(self, source: str) -> Dict:
        """
        添加单个来源到实时知识库

        Args:
            source: 文件路径或 URL

        Returns:
            处理结果
        """
        source_name = os.path.basename(source) if not source.startswith('http') else source
        logger.info(f"=" * 60)
        logger.info(f"添加到实时知识库: {source_name}")
        logger.info(f"=" * 60)

        try:
            # 处理来源
            if source.startswith(('http://', 'https://')):
                logger.info("加载 URL...")
                docs = self._process_url(source)
                source_type = "url"
            else:
                logger.info("加载文件...")
                docs = self._process_file(source)
                source_type = "file"

            logger.info(f"✓ 加载了 {len(docs)} 个文档")

            # 分割文档
            logger.info("分割文档...")
            splits = self._split_documents(docs)
            logger.info(f"✓ 创建了 {len(splits)} 个文本块")

            # 如果没有任何有效文本块，提前给出明确错误（常见于扫描 PDF）
            if len(splits) == 0:
                error_msg = (
                    "未从文档中提取到任何有效文本块，"
                    "可能是扫描件或图片 PDF，请先通过 OCR 转成可搜索文本（txt/docx）后再入库。"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            """# 输出前几个文本块，便于检查解析是否正常
            sample_count = min(3, len(splits))
            for i in range(sample_count):
                logger.info(f"示例文本块 {i+1}/{sample_count}：{splits[i].page_content[:300].replace(chr(10), ' ')}...")"""

            # 创建或更新向量存储（支持大文件）
            logger.info("构建向量索引...")
            vector_store = None
            existing_chunks = 0

            if os.path.exists(os.path.join(self.realtime_index_dir, "index.faiss")):
                try:
                    vector_store = FAISS.load_local(
                        self.realtime_index_dir,
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    existing_chunks = len(vector_store.docstore._dict)
                    logger.info(f"✓ 加载现有实时知识库 ({existing_chunks} 个现有文本块)")
                except Exception as e:
                    logger.warning(f"⚠ 加载现有索引失败: {e}")
                    vector_store = None

            # 批量处理（如果文本块很多）
            batch_size = self.config.BATCH_SIZE
            if len(splits) > batch_size:
                logger.info(f"使用批量处理模式（每批 {batch_size} 个）")
                for i in range(0, len(splits), batch_size):
                    batch = splits[i:i + batch_size]
                    batch_num = i//batch_size + 1
                    total_batches = (len(splits)-1)//batch_size + 1
                    logger.info(f"处理批次 {batch_num}/{total_batches}")

                    if vector_store is None:
                        vector_store = FAISS.from_documents(batch, self.embeddings)
                    else:
                        vector_store.add_documents(batch)
            else:
                logger.info("直接处理所有文本块...")
                if vector_store is None:
                    vector_store = FAISS.from_documents(splits, self.embeddings)
                else:
                    vector_store.add_documents(splits)

            # 保存
            logger.info("保存向量存储...")
            vector_store.save_local(self.realtime_index_dir)
            logger.info("✓ 向量存储已保存")

            # 更新元数据
            self.metadata["realtime"]["sources"].append({
                "source": source,
                "type": source_type,
                "added_at": datetime.now().isoformat(),
                "chunks": len(splits)
            })
            self.metadata["realtime"]["total_chunks"] = existing_chunks + len(splits)
            self.metadata["realtime"]["last_updated"] = datetime.now().isoformat()
            self._save_metadata()

            logger.info(f"=" * 60)
            logger.info("✓ 实时知识库更新完成！")
            logger.info(f"新增文本块: {len(splits)}")
            logger.info(f"总文本块数: {existing_chunks + len(splits)}")
            logger.info(f"=" * 60)

            return {
                "success": True,
                "chunks": len(splits),
                "existing_chunks": existing_chunks,
                "total_chunks": existing_chunks + len(splits),
                "source": source,
                "source_name": source_name
            }

        except Exception as e:
            logger.error(f"添加到实时知识库失败: {e}")
            raise

    def merge_realtime_to_historical(self) -> Dict:
        """
        将实时知识库合并到历史知识库

        Returns:
            合并结果
        """
        logger.info("开始合并实时知识库到历史知识库")

        try:
            # 检查实时知识库是否存在
            if not os.path.exists(os.path.join(self.realtime_index_dir, "index.faiss")):
                return {
                    "success": False,
                    "message": "实时知识库为空"
                }

            # 加载实时知识库
            realtime_store = FAISS.load_local(
                self.realtime_index_dir,
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            # 获取所有文档
            realtime_docs = realtime_store.docstore._dict.values()
            realtime_docs_list = list(realtime_docs)

            # 加载或创建历史知识库
            if os.path.exists(os.path.join(self.historical_index_dir, "index.faiss")):
                historical_store = FAISS.load_local(
                    self.historical_index_dir,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                historical_store.add_documents(realtime_docs_list)
            else:
                historical_store = FAISS.from_documents(realtime_docs_list, self.embeddings)

            # 保存历史知识库
            historical_store.save_local(self.historical_index_dir)

            # 更新元数据
            realtime_sources = self.metadata["realtime"]["sources"]
            self.metadata["historical"]["sources"].extend(realtime_sources)
            self.metadata["historical"]["total_chunks"] += self.metadata["realtime"]["total_chunks"]
            self.metadata["historical"]["last_updated"] = datetime.now().isoformat()

            # 清空实时知识库元数据
            merged_count = self.metadata["realtime"]["total_chunks"]
            self.metadata["realtime"] = {
                "sources": [],
                "total_chunks": 0,
                "last_updated": None
            }
            self._save_metadata()

            # 清空实时知识库文件
            self.clear_realtime_knowledge()

            logger.info(f"成功合并 {merged_count} 个文本块到历史知识库")

            return {
                "success": True,
                "merged_chunks": merged_count,
                "message": f"成功合并 {merged_count} 个文本块"
            }

        except Exception as e:
            logger.error(f"合并知识库失败: {e}")
            raise


    def search_knowledge(self, query: str, k: int = 6, use_realtime: bool = True) -> List[Document]:
        """
        在知识库中搜索 - 改进检索策略

        Args:
            query: 查询文本
            k: 返回结果数量（默认增加到 6）
            use_realtime: 是否同时搜索实时知识库

        Returns:
            相关文档列表
        """
        logger.info(f"=" * 60)
        logger.info(f"开始检索知识库")
        logger.info(f"查询: {query[:50]}...")
        logger.info(f"检索数量: {k}")
        logger.info(f"=" * 60)

        results = []
        search_sources = []

        # 搜索历史知识库
        if os.path.exists(os.path.join(self.historical_index_dir, "index.faiss")):
            try:
                logger.info("✓ 搜索历史知识库...")
                historical_store = FAISS.load_local(
                    self.historical_index_dir,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                total_docs = len(historical_store.docstore._dict)
                logger.info(f"  历史知识库文档数: {total_docs}")

                # 增加检索数量以提高召回率
                search_k = min(k * 2, total_docs)
                historical_results = historical_store.similarity_search(query, k=search_k)
                logger.info(f"  检索到 {len(historical_results)} 个相关文档")

                # 标记来源
                for doc in historical_results:
                    doc.metadata['search_source'] = 'historical'

                results.extend(historical_results)
                search_sources.append('historical')
            except Exception as e:
                logger.error(f"✗ 搜索历史知识库失败: {e}")
        else:
            logger.info("⚠ 历史知识库不存在")

        # 搜索实时知识库
        if use_realtime and os.path.exists(os.path.join(self.realtime_index_dir, "index.faiss")):
            try:
                logger.info("✓ 搜索实时知识库...")
                realtime_store = FAISS.load_local(
                    self.realtime_index_dir,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                total_docs = len(realtime_store.docstore._dict)
                logger.info(f"  实时知识库文档数: {total_docs}")

                search_k = min(k * 2, total_docs)
                realtime_results = realtime_store.similarity_search(query, k=search_k)
                logger.info(f"  检索到 {len(realtime_results)} 个相关文档")

                # 标记来源
                for doc in realtime_results:
                    doc.metadata['search_source'] = 'realtime'

                results.extend(realtime_results)
                search_sources.append('realtime')
            except Exception as e:
                logger.error(f"✗ 搜索实时知识库失败: {e}")
        else:
            if use_realtime:
                logger.info("⚠ 实时知识库不存在")

        if not results:
            logger.warning("✗ 未找到任何相关文档")
            return []

        # 去重和排序
        logger.info(f"合并结果，共 {len(results)} 个文档")
        unique_results = []
        seen_content = set()

        for doc in results:
            # 使用更长的内容片段进行去重
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(doc)

        # 取前 k 个
        final_results = unique_results[:k]

        logger.info(f"=" * 60)
        logger.info(f"✓ 检索完成")
        logger.info(f"搜索来源: {', '.join(search_sources)}")
        logger.info(f"去重后文档数: {len(unique_results)}")
        logger.info(f"返回文档数: {len(final_results)}")
        logger.info(f"=" * 60)

        return final_results

    def get_knowledge_status(self) -> Dict:
        """获取知识库状态"""
        return {
            "historical": {
                "exists": os.path.exists(os.path.join(self.historical_index_dir, "index.faiss")),
                "sources_count": len(self.metadata["historical"]["sources"]),
                "total_chunks": self.metadata["historical"]["total_chunks"],
                "last_updated": self.metadata["historical"]["last_updated"]
            },
            "realtime": {
                "exists": os.path.exists(os.path.join(self.realtime_index_dir, "index.faiss")),
                "sources_count": len(self.metadata["realtime"]["sources"]),
                "total_chunks": self.metadata["realtime"]["total_chunks"],
                "last_updated": self.metadata["realtime"]["last_updated"]
            }
        }

    def clear_realtime_knowledge(self):
        """清空实时知识库"""
        try:
            if os.path.exists(self.realtime_index_dir):
                for file in os.listdir(self.realtime_index_dir):
                    file_path = os.path.join(self.realtime_index_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

            self.metadata["realtime"] = {
                "sources": [],
                "total_chunks": 0,
                "last_updated": None
            }
            self._save_metadata()
            logger.info("实时知识库已清空")
            return True
        except Exception as e:
            logger.error(f"清空实时知识库失败: {e}")
            return False

    def clear_historical_knowledge(self):
        """清空历史知识库"""
        try:
            if os.path.exists(self.historical_index_dir):
                for file in os.listdir(self.historical_index_dir):
                    file_path = os.path.join(self.historical_index_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

            self.metadata["historical"] = {
                "sources": [],
                "total_chunks": 0,
                "last_updated": None
            }
            self._save_metadata()
            logger.info("历史知识库已清空")
            return True
        except Exception as e:
            logger.error(f"清空历史知识库失败: {e}")
            return False


    def validate_file_size(self, file_path: str) -> Dict:
        """
        验证文件大小并返回信息

        Returns:
            包含文件信息的字典
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)

        is_large = file_size > 100 * 1024 * 1024  # 超过 100MB 视为大文件
        is_valid = file_size <= self.config.MAX_FILE_SIZE

        estimated_chunks = int(file_size / (self.config.CHUNK_SIZE * 2))  # 粗略估计
        estimated_time = estimated_chunks / 10  # 假设每秒处理 10 个块

        return {
            "file_path": file_path,
            "file_size": file_size,
            "file_size_mb": file_size_mb,
            "is_large": is_large,
            "is_valid": is_valid,
            "estimated_chunks": estimated_chunks,
            "estimated_time_seconds": estimated_time,
            "max_size_mb": self.config.MAX_FILE_SIZE / (1024 * 1024)
        }

    def get_processing_stats(self) -> Dict:
        """获取处理统计信息"""
        stats = {
            "config": {
                "max_file_size_mb": self.config.MAX_FILE_SIZE / (1024 * 1024),
                "chunk_size": self.config.CHUNK_SIZE,
                "chunk_overlap": self.config.CHUNK_OVERLAP,
                "batch_size": self.config.BATCH_SIZE
            },
            "historical": self.metadata["historical"],
            "realtime": self.metadata["realtime"]
        }

        # 计算总大小
        if os.path.exists(self.historical_index_dir):
            hist_size = sum(
                os.path.getsize(os.path.join(self.historical_index_dir, f))
                for f in os.listdir(self.historical_index_dir)
                if os.path.isfile(os.path.join(self.historical_index_dir, f))
            )
            stats["historical"]["index_size_mb"] = hist_size / (1024 * 1024)

        if os.path.exists(self.realtime_index_dir):
            real_size = sum(
                os.path.getsize(os.path.join(self.realtime_index_dir, f))
                for f in os.listdir(self.realtime_index_dir)
                if os.path.isfile(os.path.join(self.realtime_index_dir, f))
            )
            stats["realtime"]["index_size_mb"] = real_size / (1024 * 1024)

        return stats
