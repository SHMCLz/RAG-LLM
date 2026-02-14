"""
# GPU在构建历史数据库和进行推理回答时占用率低====》ollama推理过程中使用的是cpu
# pdf知识入库的结构可不可以ocr直接提取文字用json或txt
# 构建知识库的难度和微调的难度
# http怎么获取到子url中的知识
目前需要解决回答有点智障，应该是检索相关知识的问题，当前的数据库检索方式修改
1. 检索到相关文档片段很快
2.回答速度非常慢，即使只检查6个文档片段
增强版 GUI - 支持历史知识库和实时知识库

# 生成嵌入向量的方法不兼容：固定生成嵌入向量模型（不会追加应该重建？）
#
# 环境创建步骤：虚拟conda环境->python3.10->根据运行提示安装软件包->pip install sentence_transformers->conda install -c conda-forge faiss-gpu
# 检索回答不准确的原因：文本切片存入的数据不准确，比如一个段落会被中间拆分，多个段落会合并
# 存储速度慢的原因：当前已经把构建索引进行加速，但是PDFloader包本身处理速度慢
# 回答速度慢的原因：检索相关存储数据的速度很快，但是给LLM生成回答的速度慢
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'RAG_bot'))

from RAG_bot.knowledge_manager import KnowledgeManager
from RAG_bot.Requests import answer
from conversation_manager import ConversationManager

load_dotenv()


class EnhancedArticleAssistantGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SmartRAG ")
        self.root.geometry("1000x800")
        self.root.minsize(900, 700)

        # 初始化知识库管理器
        self.km = KnowledgeManager()
        
        # 初始化对话管理器
        self.conversation = ConversationManager(max_history=5)

        # 变量
        self.processing = False
        self.selected_files = []

        # 检查 API key
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            messagebox.showwarning("警告", "未在 .env 文件中找到 OPENAI_API_KEY！")

        self.setup_ui()
        self.update_status_display()

    def setup_ui(self):
        # 主容器
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)

        # 标题
        title_label = ttk.Label(main_frame, text="SmartRAG ",
                               font=("Microsoft YaHei", 16, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 10))

        # 创建 Notebook（标签页）
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        # 标签页1: 历史知识库
        self.historical_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.historical_frame, text="📚 历史知识库")
        self.setup_historical_tab()

        # 标签页2: 实时知识库
        self.realtime_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.realtime_frame, text="⚡ 实时知识库")
        self.setup_realtime_tab()

        # 标签页3: 智能问答
        self.chat_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.chat_frame, text="💬 智能问答")
        self.setup_chat_tab()

        # 状态栏
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(5, 0))

        self.status_label = ttk.Label(status_frame, text="就绪", foreground="green")
        self.status_label.pack(side=tk.LEFT)

        self.progress_bar = ttk.Progressbar(status_frame, mode='indeterminate', length=200)
        self.progress_bar.pack(side=tk.RIGHT)
        self.progress_bar.pack_forget()


    def setup_historical_tab(self):
        """设置历史知识库标签页"""
        # 说明
        info_label = ttk.Label(self.historical_frame,
                              text="批量上传文件构建持久化的历史知识库",
                              font=("Microsoft YaHei", 10))
        info_label.grid(row=0, column=0, columnspan=3, pady=(0, 10), sticky=tk.W)

        # 文件选择区域
        file_frame = ttk.LabelFrame(self.historical_frame, text="选择文件", padding="10")
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(0, weight=1)

        ttk.Button(file_frame, text="添加 PDF/TXT 文件",
                  command=self.add_files_to_historical).grid(row=0, column=0, sticky=tk.W, padx=(0, 5))

        ttk.Button(file_frame, text="添加 URL 列表文件",
                  command=self.add_url_list_to_historical).grid(row=0, column=1, sticky=tk.W, padx=(0, 5))

        ttk.Button(file_frame, text="清空选择",
                  command=self.clear_file_selection).grid(row=0, column=2, sticky=tk.W)

        # 已选文件列表
        list_frame = ttk.LabelFrame(self.historical_frame, text="已选文件", padding="10")
        list_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        self.file_listbox = tk.Listbox(list_frame, height=8, font=("Microsoft YaHei", 9))
        self.file_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.file_listbox.config(yscrollcommand=scrollbar.set)

        # 操作按钮
        btn_frame = ttk.Frame(self.historical_frame)
        btn_frame.grid(row=3, column=0, columnspan=3, pady=(0, 10))

        ttk.Button(btn_frame, text="🚀 构建历史知识库",
                  command=self.build_historical_knowledge,
                  style="Accent.TButton").pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(btn_frame, text="🗑️ 清空历史知识库",
                  command=self.clear_historical_knowledge).pack(side=tk.LEFT)

        # 状态显示
        status_frame = ttk.LabelFrame(self.historical_frame, text="知识库状态", padding="10")
        status_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E))

        self.historical_status_text = tk.Text(status_frame, height=4, state='disabled',
                                             font=("Microsoft YaHei", 9))
        self.historical_status_text.grid(row=0, column=0, sticky=(tk.W, tk.E))

    def setup_realtime_tab(self):
        """设置实时知识库标签页"""
        # 说明
        info_label = ttk.Label(self.realtime_frame,
                              text="临时加载单个文件或 URL，可随时合并到历史知识库",
                              font=("Microsoft YaHei", 10))
        info_label.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky=tk.W)

        # URL 输入
        url_frame = ttk.LabelFrame(self.realtime_frame, text="加载 URL", padding="10")
        url_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        url_frame.columnconfigure(1, weight=1)

        ttk.Label(url_frame, text="网址:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.realtime_url_entry = ttk.Entry(url_frame, font=("Microsoft YaHei", 9))
        self.realtime_url_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))

        ttk.Button(url_frame, text="加载",
                  command=self.load_url_to_realtime).grid(row=0, column=2)

        # 文件输入
        file_frame = ttk.LabelFrame(self.realtime_frame, text="加载文件", padding="10")
        file_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)

        ttk.Label(file_frame, text="文件:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.realtime_file_label = ttk.Label(file_frame, text="未选择文件", foreground="gray")
        self.realtime_file_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 5))

        ttk.Button(file_frame, text="浏览并加载",
                  command=self.load_file_to_realtime).grid(row=0, column=2)

        # 操作按钮
        btn_frame = ttk.Frame(self.realtime_frame)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=(10, 10))

        ttk.Button(btn_frame, text="📥 合并到历史知识库",
                  command=self.merge_to_historical,
                  style="Accent.TButton").pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(btn_frame, text="🗑️ 清空实时知识库",
                  command=self.clear_realtime_knowledge).pack(side=tk.LEFT)

        # 状态显示
        status_frame = ttk.LabelFrame(self.realtime_frame, text="知识库状态", padding="10")
        status_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E))

        self.realtime_status_text = tk.Text(status_frame, height=4, state='disabled',
                                           font=("Microsoft YaHei", 9))
        self.realtime_status_text.grid(row=0, column=0, sticky=(tk.W, tk.E))


    def setup_chat_tab(self):
        """设置智能问答标签页"""
        # 说明
        info_label = ttk.Label(self.chat_frame,
                              text="基于历史知识库和实时知识库进行智能问答",
                              font=("Microsoft YaHei", 10))
        info_label.grid(row=0, column=0, pady=(0, 10), sticky=tk.W)

        # 聊天显示区
        chat_display_frame = ttk.Frame(self.chat_frame)
        chat_display_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        chat_display_frame.columnconfigure(0, weight=1)
        chat_display_frame.rowconfigure(0, weight=1)

        self.chat_display = scrolledtext.ScrolledText(chat_display_frame, wrap=tk.WORD,
                                                      height=20, state='disabled',
                                                      font=("Microsoft YaHei", 10))
        self.chat_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置标签样式
        self.chat_display.tag_config("user", foreground="blue", font=("Microsoft YaHei", 10, "bold"))
        self.chat_display.tag_config("assistant", foreground="green", font=("Microsoft YaHei", 10))
        self.chat_display.tag_config("system", foreground="gray", font=("Microsoft YaHei", 9, "italic"))

        # 问题输入区
        input_frame = ttk.Frame(self.chat_frame)
        input_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))
        input_frame.columnconfigure(0, weight=1)

        self.question_entry = ttk.Entry(input_frame, font=("Microsoft YaHei", 10))
        self.question_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        self.question_entry.bind('<Return>', lambda e: self.ask_question())

        self.ask_button = ttk.Button(input_frame, text="提问", command=self.ask_question)
        self.ask_button.grid(row=0, column=1)

        # 选项
        option_frame = ttk.Frame(self.chat_frame)
        option_frame.grid(row=3, column=0, sticky=tk.W, pady=(5, 0))

        self.use_realtime_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(option_frame, text="同时搜索实时知识库",
                       variable=self.use_realtime_var).pack(side=tk.LEFT, padx=(0, 10))
        
        # 添加清空对话按钮
        ttk.Button(option_frame, text="🔄 清空对话", 
                  command=self.clear_conversation).pack(side=tk.LEFT)

        # 初始消息
        self.add_message("system", "欢迎使用 SmartRAG！\n\n"
                        "• 历史知识库：批量构建持久化知识库\n"
                        "• 实时知识库：临时加载单个文件或 URL\n"
                        "• 智能问答：优先从历史知识库检索，可选择是否使用实时知识库\n"
                        "• 多轮对话：支持上下文理解，可以说'这个'、'它'等指代词")

    # ========== 历史知识库功能 ==========

    def add_files_to_historical(self):
        """添加文件到历史知识库构建列表"""
        files = filedialog.askopenfilenames(
            title="选择文件",
            filetypes=[("支持的文件", "*.pdf *.txt"), ("PDF 文件", "*.pdf"),
                      ("文本文件", "*.txt"), ("所有文件", "*.*")]
        )

        for file in files:
            if file not in self.selected_files:
                # 验证文件大小
                try:
                    file_info = self.km.validate_file_size(file)
                    if not file_info['is_valid']:
                        messagebox.showwarning(
                            "文件过大",
                            f"文件 {os.path.basename(file)} 大小为 {file_info['file_size_mb']:.1f} MB\n"
                            f"超过最大限制 {file_info['max_size_mb']:.0f} MB"
                        )
                        continue

                    self.selected_files.append(file)

                    # 显示文件大小
                    size_str = f"{file_info['file_size_mb']:.1f}MB"
                    if file_info['is_large']:
                        size_str += " [大文件]"

                    display_name = f"{os.path.basename(file)} ({size_str})"
                    self.file_listbox.insert(tk.END, display_name)

                except Exception as e:
                    messagebox.showerror("错误", f"验证文件失败:\n{str(e)}")

    def add_url_list_to_historical(self):
        """添加 URL 列表文件"""
        file = filedialog.askopenfilename(
            title="选择 URL 列表文件",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )

        if file and file not in self.selected_files:
            self.selected_files.append(file)
            self.file_listbox.insert(tk.END, f"[URL列表] {os.path.basename(file)}")

    def clear_file_selection(self):
        """清空文件选择"""
        self.selected_files.clear()
        self.file_listbox.delete(0, tk.END)

    def build_historical_knowledge(self):
        """构建历史知识库"""
        if not self.selected_files:
            messagebox.showwarning("警告", "请先选择文件！")
            return

        # 创建详细信息显示窗口
        detail_window = tk.Toplevel(self.root)
        detail_window.title("构建进度")
        detail_window.geometry("600x400")
        detail_window.transient(self.root)

        # 详细信息文本框
        detail_text = scrolledtext.ScrolledText(detail_window, wrap=tk.WORD,
                                                font=("Consolas", 9))
        detail_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        def add_detail(message):
            """添加详细信息"""
            detail_text.insert(tk.END, message + "\n")
            detail_text.see(tk.END)
            detail_text.update()

        def progress_callback(current, total, message, details=None):
            self.set_status(f"[{current}/{total}] {message}", "orange", show_progress=True)
            add_detail(f"[{current}/{total}] {message}")

        def process():
            try:
                self.set_processing(True)
                add_detail("=" * 60)
                add_detail("开始构建历史知识库")
                add_detail(f"文件数量: {len(self.selected_files)}")
                add_detail("=" * 60)
                add_detail("")

                result = self.km.build_historical_knowledge(
                    self.selected_files,
                    progress_callback=progress_callback
                )

                # 显示详细结果
                add_detail("")
                add_detail("=" * 60)
                add_detail("✓ 构建完成！")
                add_detail("=" * 60)
                add_detail(f"总来源数: {result['total_sources']}")
                add_detail(f"成功处理: {result['processed_sources']}")
                add_detail(f"处理失败: {result['failed_sources']}")
                add_detail(f"新增文本块: {result['new_chunks']}")
                add_detail(f"现有文本块: {result['existing_chunks']}")
                add_detail(f"总文本块数: {result['total_chunks']}")
                add_detail("")

                # 显示处理详情
                if result.get('processing_details'):
                    add_detail("处理详情:")
                    for detail in result['processing_details']:
                        add_detail(f"  {detail}")

                add_detail("")
                add_detail("=" * 60)

                self.set_status("历史知识库构建完成", "green", show_progress=False)
                self.update_status_display()

                # 显示摘要对话框
                summary = (
                    f"✓ 历史知识库构建完成！\n\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"📊 统计信息\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"总来源数: {result['total_sources']}\n"
                    f"成功处理: {result['processed_sources']}\n"
                    f"处理失败: {result['failed_sources']}\n\n"
                    f"新增文本块: {result['new_chunks']}\n"
                    f"现有文本块: {result['existing_chunks']}\n"
                    f"总文本块数: {result['total_chunks']}\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"详细信息请查看进度窗口"
                )

                messagebox.showinfo("构建完成", summary)
                self.clear_file_selection()

            except Exception as e:
                add_detail("")
                add_detail("=" * 60)
                add_detail(f"✗ 构建失败: {str(e)}")
                add_detail("=" * 60)
                self.set_status("构建失败", "red", show_progress=False)
                messagebox.showerror("错误", f"构建历史知识库失败:\n{str(e)}")
            finally:
                self.set_processing(False)

        threading.Thread(target=process, daemon=True).start()

    def clear_historical_knowledge(self):
        """清空历史知识库"""
        if messagebox.askyesno("确认", "确定要清空历史知识库吗？此操作不可恢复！"):
            try:
                self.km.clear_historical_knowledge()
                self.update_status_display()
                messagebox.showinfo("成功", "历史知识库已清空")
            except Exception as e:
                messagebox.showerror("错误", f"清空失败:\n{str(e)}")


    # ========== 实时知识库功能 ==========

    def load_url_to_realtime(self):
        """加载 URL 到实时知识库"""
        url = self.realtime_url_entry.get().strip()
        if not url:
            messagebox.showwarning("警告", "请输入网址")
            return

        if not url.startswith(('http://', 'https://')):
            messagebox.showerror("错误", "请输入有效的网址")
            return

        def process():
            try:
                self.set_processing(True)
                self.set_status("正在加载 URL...", "orange", show_progress=True)

                result = self.km.add_to_realtime_knowledge(url)

                self.set_status("URL 加载完成", "green", show_progress=False)
                self.update_status_display()

                messagebox.showinfo("成功",
                    f"成功加载 URL 到实时知识库！\n\n"
                    f"文本块数: {result['chunks']}")

                self.realtime_url_entry.delete(0, tk.END)

            except Exception as e:
                self.set_status("加载失败", "red", show_progress=False)
                messagebox.showerror("错误", f"加载 URL 失败:\n{str(e)}")
            finally:
                self.set_processing(False)

        threading.Thread(target=process, daemon=True).start()

    def load_file_to_realtime(self):
        """加载文件到实时知识库"""
        file_path = filedialog.askopenfilename(
            title="选择文件",
            filetypes=[("PDF 文件", "*.pdf"), ("文本文件", "*.txt"), ("所有文件", "*.*")]
        )

        if not file_path:
            return

        # 验证文件大小
        try:
            file_info = self.km.validate_file_size(file_path)
            if not file_info['is_valid']:
                messagebox.showerror(
                    "文件过大",
                    f"文件大小为 {file_info['file_size_mb']:.1f} MB\n"
                    f"超过最大限制 {file_info['max_size_mb']:.0f} MB"
                )
                return

            # 显示文件信息
            size_str = f"({file_info['file_size_mb']:.1f}MB)"
            if file_info['is_large']:
                size_str += " [大文件，预计 {:.0f} 秒]".format(file_info['estimated_time_seconds'])

            self.realtime_file_label.config(
                text=f"{os.path.basename(file_path)} {size_str}",
                foreground="black"
            )
        except Exception as e:
            messagebox.showerror("错误", f"验证文件失败:\n{str(e)}")
            return

        def process():
            try:
                self.set_processing(True)
                self.set_status("正在加载文件...", "orange", show_progress=True)

                result = self.km.add_to_realtime_knowledge(file_path)

                self.set_status("文件加载完成", "green", show_progress=False)
                self.update_status_display()

                # 显示详细摘要
                summary = (
                    f"✓ 文件加载完成！\n\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"📄 文件信息\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"文件名: {result['source_name']}\n"
                    f"新增文本块: {result['chunks']}\n"
                    f"现有文本块: {result['existing_chunks']}\n"
                    f"总文本块数: {result['total_chunks']}\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"现在可以在问答页面提问了！"
                )

                messagebox.showinfo("加载成功", summary)

            except Exception as e:
                self.set_status("加载失败", "red", show_progress=False)
                messagebox.showerror("错误", f"加载文件失败:\n{str(e)}")
            finally:
                self.set_processing(False)

        threading.Thread(target=process, daemon=True).start()

    def merge_to_historical(self):
        """合并实时知识库到历史知识库"""
        if messagebox.askyesno("确认", "确定要将实时知识库合并到历史知识库吗？\n合并后实时知识库将被清空。"):
            def process():
                try:
                    self.set_processing(True)
                    self.set_status("正在合并知识库...", "orange", show_progress=True)

                    result = self.km.merge_realtime_to_historical()

                    if result['success']:
                        self.set_status("合并完成", "green", show_progress=False)
                        self.update_status_display()
                        messagebox.showinfo("成功", result['message'])
                    else:
                        messagebox.showwarning("提示", result['message'])

                except Exception as e:
                    self.set_status("合并失败", "red", show_progress=False)
                    messagebox.showerror("错误", f"合并失败:\n{str(e)}")
                finally:
                    self.set_processing(False)

            threading.Thread(target=process, daemon=True).start()

    def clear_realtime_knowledge(self):
        """清空实时知识库"""
        if messagebox.askyesno("确认", "确定要清空实时知识库吗？"):
            try:
                self.km.clear_realtime_knowledge()
                self.update_status_display()
                messagebox.showinfo("成功", "实时知识库已清空")
            except Exception as e:
                messagebox.showerror("错误", f"清空失败:\n{str(e)}")

    # ========== 智能问答功能 ==========

    def ask_question(self):
        """提问"""
        original_question = self.question_entry.get().strip()
        if not original_question:
            return

        # 检查是否有知识库
        status = self.km.get_knowledge_status()
        if not status['historical']['exists'] and not status['realtime']['exists']:
            messagebox.showwarning("警告", "请先构建历史知识库或加载实时知识库！")
            return

        # 显示原始问题
        self.add_message("user", original_question)
        self.question_entry.delete(0, tk.END)
        
        # 指代消解
        resolved_question = self.conversation.resolve_reference(original_question)
        if resolved_question != original_question:
            self.add_message("system", f"💡 理解为: {resolved_question}")

        def process():
            try:
                self.set_processing(True)
                self.set_status("正在检索知识库...", "orange", show_progress=True)

                # 使用解析后的问题进行检索
                search_query = resolved_question

                # 使用知识管理器搜索（增加检索数量）
                use_realtime = self.use_realtime_var.get()
                docs = self.km.search_knowledge(search_query, k=6, use_realtime=use_realtime)

                if not docs:
                    error_msg = "❌ 未找到相关信息\n\n可能原因：\n1. 知识库中没有相关内容\n2. 问题表述不够清晰\n3. 尝试换个问法"
                    self.add_message("system", error_msg)
                    self.conversation.add_message("assistant", error_msg)
                    self.set_status("未找到相关信息", "orange", show_progress=False)
                    return

                # 显示检索信息
                search_info = f"[检索到 {len(docs)} 个相关文档片段]"
                self.add_message("system", search_info)
                
                # 保存检索到的文档
                self.conversation.last_retrieved_docs = docs

                self.set_status("正在生成回答...", "orange", show_progress=True)

                # 构建上下文
                context = "\n\n---\n\n".join([
                    f"文档片段 {i+1}:\n{doc.page_content}"
                    for i, doc in enumerate(docs)
                ])

                from RAG_bot.Requests import llm
                from langchain_core.prompts import ChatPromptTemplate

                # 判断是否需要对话上下文
                use_context = self.conversation.should_use_context(original_question)
                
                if use_context and len(self.conversation.history) > 0:
                    # 包含对话历史的 Prompt
                    chinese_prompt = ChatPromptTemplate.from_template(
                        """你是一位经验丰富的中医药专家。请基于你的专业知识回答用户的问题。

对话历史：
{conversation_history}

背景资料：
{context}

当前问题：{question}

回答要求：
1. 结合对话历史理解问题中的指代词（如"这个"、"它"、"该"）
2. 综合背景资料和你的专业知识给出完整、准确的答案
3. 直接回答问题，不要提及"文档"、"资料"、"参考"等词汇
4. 以专业、自信的口吻回答，就像你是该领域的权威专家
5. 保持详细、清晰、专业

请直接回答："""
                    )
                    
                    # 构建对话历史
                    history_text = "\n".join([
                        f"{'用户' if msg['role'] == 'user' else '助手'}: {msg['content'][:150]}"
                        for msg in self.conversation.history[-4:]  # 最近2轮
                    ])
                    
                    formatted_prompt = chinese_prompt.format(
                        conversation_history=history_text,
                        question=original_question,
                        context=context
                    )
                else:
                    # 不包含对话历史的 Prompt
                    chinese_prompt = ChatPromptTemplate.from_template(
                        """你是一位经验丰富的中医药专家。请基于你的专业知识回答用户的问题。

背景资料：
{context}

用户问题：{question}

回答要求：
1. 综合背景资料和你的专业知识给出完整、准确的答案
2. 直接回答问题，不要提及"文档"、"资料"、"参考"等词汇
3. 以专业、自信的口吻回答，就像你是该领域的权威专家
4. 保持详细、清晰、专业
5. 如果问题是中文，用中文回答；如果是英文，用英文回答

请直接回答："""
                    )

                    formatted_prompt = chinese_prompt.format(
                        question=resolved_question,
                        context=context
                    )

                use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
                if use_ollama:
                    response = llm.invoke(formatted_prompt)
                    response_text = response if isinstance(response, str) else str(response)
                else:
                    response = llm.invoke(formatted_prompt)
                    response_text = response.content
                
                # 过滤掉不必要的前缀和短语
                unwanted_prefixes = [
                    "根据提供的文档内容，",
                    "根据文档内容，",
                    "根据提供的文档，",
                    "参考提供的文档信息，",
                    "根据背景资料，",
                    "文档中提到，",
                    "文档显示，",
                    "根据上述文档，",
                    "从文档中可以看出，",
                    "根据我的专业知识，",
                    "抱歉，",
                ]
                
                for prefix in unwanted_prefixes:
                    if response_text.startswith(prefix):
                        response_text = response_text[len(prefix):]
                        break
                
                # 过滤掉包含"文档"、"资料"等提及的句子（只在开头）
                import re
                # 移除开头提及文档/资料的句子
                response_text = re.sub(
                    r'^[^。！？]*?(文档|资料|参考)[^。！？]*?[。！？]\s*',
                    '',
                    response_text
                )

                final_answer = response_text.strip()
                self.add_message("assistant", final_answer)
                
                # 添加到对话历史
                self.conversation.add_message("user", original_question, {
                    "resolved_question": resolved_question,
                    "docs_count": len(docs)
                })
                self.conversation.add_message("assistant", final_answer)
                
                self.set_status("就绪", "green", show_progress=False)

            except Exception as e:
                error_msg = f"❌ 处理问题时出错\n\n错误信息: {str(e)}\n\n请检查：\n1. 网络连接\n2. API Key 是否有效\n3. 知识库是否正常"
                self.add_message("system", error_msg)
                self.conversation.add_message("assistant", error_msg)
                self.set_status("错误", "red", show_progress=False)
            finally:
                self.set_processing(False)

        threading.Thread(target=process, daemon=True).start()
    
    def clear_conversation(self):
        """清空对话历史"""
        if messagebox.askyesno("确认", "确定要清空对话历史吗？"):
            self.conversation.clear_history()
            self.chat_display.config(state='normal')
            self.chat_display.delete(1.0, tk.END)
            self.chat_display.config(state='disabled')
            self.add_message("system", "对话历史已清空")


    # ========== 辅助功能 ==========

    def add_message(self, sender, message):
        """添加消息到聊天显示"""
        self.chat_display.config(state='normal')

        if sender == "user":
            self.chat_display.insert(tk.END, "你: ", "user")
        elif sender == "assistant":
            self.chat_display.insert(tk.END, "助手: ", "assistant")
        else:
            self.chat_display.insert(tk.END, "系统: ", "system")

        self.chat_display.insert(tk.END, message + "\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state='disabled')

    def set_status(self, message, color="black", show_progress=False):
        """设置状态栏"""
        self.status_label.config(text=message, foreground=color)
        if show_progress:
            self.progress_bar.pack(side=tk.RIGHT)
            self.progress_bar.start(10)
        else:
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
        self.root.update()

    def set_processing(self, is_processing):
        """设置处理状态"""
        self.processing = is_processing
        state = 'disabled' if is_processing else 'normal'
        self.ask_button.config(state=state)
        self.question_entry.config(state=state)

    def update_status_display(self):
        """更新知识库状态显示"""
        status = self.km.get_knowledge_status()

        # 更新历史知识库状态
        historical_text = f"""存在: {'是' if status['historical']['exists'] else '否'}
来源数量: {status['historical']['sources_count']}
文本块数: {status['historical']['total_chunks']}
最后更新: {status['historical']['last_updated'] or '从未'}"""

        self.historical_status_text.config(state='normal')
        self.historical_status_text.delete(1.0, tk.END)
        self.historical_status_text.insert(1.0, historical_text)
        self.historical_status_text.config(state='disabled')

        # 更新实时知识库状态
        realtime_text = f"""存在: {'是' if status['realtime']['exists'] else '否'}
来源数量: {status['realtime']['sources_count']}
文本块数: {status['realtime']['total_chunks']}
最后更新: {status['realtime']['last_updated'] or '从未'}"""

        self.realtime_status_text.config(state='normal')
        self.realtime_status_text.delete(1.0, tk.END)
        self.realtime_status_text.insert(1.0, realtime_text)
        self.realtime_status_text.config(state='disabled')


def main():
    root = tk.Tk()
    app = EnhancedArticleAssistantGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
