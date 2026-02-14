"""
对话上下文管理器 - 支持多轮对话和上下文理解
"""
import re
from typing import List, Dict, Optional
from datetime import datetime


class ConversationManager:
    """管理对话历史和上下文"""
    
    def __init__(self, max_history: int = 5):
        """
        初始化对话管理器
        
        Args:
            max_history: 保留的最大历史轮数
        """
        self.history: List[Dict] = []
        self.max_history = max_history
        self.current_topic: Optional[str] = None
        self.last_retrieved_docs: List = []
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """
        添加消息到历史
        
        Args:
            role: 'user' 或 'assistant'
            content: 消息内容
            metadata: 额外元数据（如检索到的文档数量）
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.history.append(message)
        
        # 保持历史长度
        if len(self.history) > self.max_history * 2:  # user + assistant = 2条
            self.history = self.history[-self.max_history * 2:]
        
        # 如果是用户消息，提取话题
        if role == "user":
            self.current_topic = self._extract_topic(content)
    
    def _extract_topic(self, text: str) -> Optional[str]:
        """
        从文本中提取主要话题（实体）
        
        简单实现：提取名词性短语
        """
        # 移除常见的疑问词
        text = re.sub(r'^(什么是|介绍一下|请问|能否|如何|怎么|为什么)', '', text)
        
        # 提取可能的主题词（中文：2-10个字的词组）
        matches = re.findall(r'[\u4e00-\u9fa5]{2,10}', text)
        
        if matches:
            # 返回最长的词组作为话题
            return max(matches, key=len)
        
        return None
    
    def resolve_reference(self, question: str) -> str:
        """
        解析指代词，将"这个"、"它"等替换为具体话题
        
        Args:
            question: 用户问题
            
        Returns:
            解析后的问题
        """
        # 检测是否包含指代词
        reference_patterns = [
            r'^(这个|这|那个|那|它|其|该)',
            r'(这个|这|那个|那|它|其|该)(药材|文档|论文|内容|方法|技术|概念|理论)',
        ]
        
        has_reference = any(re.search(pattern, question) for pattern in reference_patterns)
        
        if has_reference and self.current_topic:
            # 只替换指代词为话题名称，不拼接整个问题
            resolved = re.sub(
                r'^(这个|这|那个|那|它|其|该)',
                self.current_topic,
                question,
                count=1  # 只替换第一个
            )
            
            resolved = re.sub(
                r'(这个|这|那个|那|它|其|该)(药材|文档|论文|内容|方法|技术|概念|理论)',
                f'{self.current_topic}\\2',
                resolved,
                count=1  # 只替换第一个
            )
            
            print(f"[对话管理] 指代消解: '{question}' -> '{resolved}'")
            return resolved
        
        return question
    
    def get_context_prompt(self, question: str, max_turns: int = 2) -> str:
        """
        构建包含历史上下文的 Prompt
        
        Args:
            question: 当前问题
            max_turns: 包含的最大历史轮数
            
        Returns:
            包含上下文的完整问题
        """
        if not self.history:
            return question
        
        # 获取最近的对话历史
        recent_history = self.history[-(max_turns * 2):]
        
        # 构建上下文
        context_lines = []
        for msg in recent_history:
            role_name = "用户" if msg["role"] == "user" else "助手"
            # 截断过长的内容
            content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
            context_lines.append(f"{role_name}: {content}")
        
        context = "\n".join(context_lines)
        
        # 构建完整 Prompt
        full_prompt = f"""对话历史:
{context}

当前问题: {question}

请基于对话历史和当前问题，理解用户的真实意图并回答。如果当前问题中有指代词（如"这个"、"它"），请结合对话历史理解其指代的内容。"""
        
        return full_prompt
    
    def should_use_context(self, question: str) -> bool:
        """
        判断是否需要使用对话上下文
        
        Args:
            question: 用户问题
            
        Returns:
            是否需要上下文
        """
        # 检测指代词
        reference_patterns = [
            r'(这个|这|那个|那|它|其|该)',
            r'(还有|另外|此外|而且|并且)',
            r'(继续|接着|然后)',
        ]
        
        return any(re.search(pattern, question) for pattern in reference_patterns)
    
    def clear_history(self):
        """清空对话历史"""
        self.history.clear()
        self.current_topic = None
        self.last_retrieved_docs.clear()
        print("[对话管理] 对话历史已清空")
    
    def get_summary(self) -> str:
        """获取对话摘要"""
        if not self.history:
            return "暂无对话历史"
        
        total_turns = len(self.history) // 2
        current_topic = self.current_topic or "未识别"
        
        return f"对话轮数: {total_turns}, 当前话题: {current_topic}"
