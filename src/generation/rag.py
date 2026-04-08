import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Changed: replaced DashScope with Hugging Face InferenceClient
# LLM: Qwen/Qwen2.5-7B-Instruct served via HF Serverless Inference API
# Uses the OpenAI-compatible chat_completion endpoint for structured I/O
from huggingface_hub import InferenceClient

from src.vector_database.chroma_vector_db import ChromaVectorDB
from src.embeddings.embedding_generator import EmbeddingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Represents the result of RAG generation with citations"""
    query: str
    response: str
    sources_used: List[Dict[str, Any]]
    retrieval_count: int
    generation_tokens: Optional[int] = None

    def get_citation_summary(self) -> str:
        if not self.sources_used:
            return "No sources cited"

        source_summary = []
        for source in self.sources_used:
            source_info = (
                f"• {source.get('source_file', 'Unknown')} "
                f"({source.get('source_type', 'unknown')})"
            )
            if source.get("page_number"):
                source_info += f" - Page {source['page_number']}"
            source_summary.append(source_info)

        return "\n".join(source_summary)


class RAGGenerator:
    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        vector_db: ChromaVectorDB,
        # Changed: primary param is now hf_api_token
        hf_api_token: Optional[str] = None,
        # Legacy kwarg aliases kept so callers don't break immediately
        dashscope_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        # Changed: default model is now Qwen2.5-7B-Instruct on HF
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ):
        self.embedding_generator = embedding_generator
        self.vector_db = vector_db
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Resolve the HF token: explicit arg → env var
        resolved_token = (
            hf_api_token
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )

        # Build the InferenceClient once — reused for every call
        self._client = InferenceClient(
            model=self.model_name,
            token=resolved_token,
        )

        logger.info(
            f"RAG Generator initialised with HF model: {self.model_name}"
        )

    # ------------------------------------------------------------------
    # Private: thin wrapper around HF chat_completion
    # ------------------------------------------------------------------
    def _call_hf(self, prompt: str) -> str:
        """
        Call Qwen2.5-7B-Instruct via the HF Serverless Inference API.

        Uses the OpenAI-compatible chat_completion interface which:
        - Accepts a messages list (system + user turns)
        - Returns a ChatCompletionOutput with choices[0].message.content
        - Works on any HF-hosted text-generation model that supports chat

        Replaces: DashScope Generation.call() / crewai LLM.call()
        """
        completion = self._client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return completion.choices[0].message.content

    # ------------------------------------------------------------------
    # Public API (signatures unchanged)
    # ------------------------------------------------------------------
    def generate_response(
        self,
        query: str,
        max_chunks: int = 8,
        max_context_chars: int = 4000,
        top_k: int = 10,
    ) -> RAGResult:

        if not query.strip():
            return RAGResult(
                query=query,
                response="Please provide a valid question.",
                sources_used=[],
                retrieval_count=0,
            )

        try:
            logger.info(f"Generating response for: '{query[:50]}...'")

            # Step 1: Retrieve relevant chunks
            query_vector = self.embedding_generator.generate_query_embedding(query)
            search_results = self.vector_db.search(
                query_vector=query_vector.tolist(),
                limit=top_k,
            )

            if not search_results:
                return RAGResult(
                    query=query,
                    response=(
                        "I couldn't find any relevant information in the available "
                        "documents to answer your question."
                    ),
                    sources_used=[],
                    retrieval_count=0,
                )

            # Step 2: Format context with numbered citations
            context, sources_info = self._format_context_with_citations(
                search_results, max_chunks, max_context_chars
            )

            # Step 3: Build the prompt
            prompt = self._create_rag_prompt(query, context)

            # Step 4: Generate via Qwen on HF (was: DashScope / crewai LLM)
            response = self._call_hf(prompt)

            rag_result = RAGResult(
                query=query,
                response=response,
                sources_used=sources_info,
                retrieval_count=len(search_results),
            )

            logger.info(
                f"Response generated successfully using {len(sources_info)} sources"
            )
            return rag_result

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return RAGResult(
                query=query,
                response=f"I encountered an error while processing your question: {str(e)}",
                sources_used=[],
                retrieval_count=0,
            )

    def _format_context_with_citations(
        self,
        search_results: List[Dict[str, Any]],
        max_chunks: int,
        max_context_chars: int,
    ) -> Tuple[str, List[Dict[str, Any]]]:

        context_parts: List[str] = []
        sources_info: List[Dict[str, Any]] = []
        total_chars = 0

        for i, result in enumerate(search_results[:max_chunks]):
            citation_info = result["citation"]
            source_file = citation_info.get("source_file", "Unknown Source")
            source_type = citation_info.get("source_type", "unknown")
            page_number = citation_info.get("page_number")

            citation_ref = f"[{i + 1}]"
            chunk_text = f"{citation_ref} {result['content']}"

            if total_chars + len(chunk_text) > max_context_chars and context_parts:
                break

            context_parts.append(chunk_text)
            total_chars += len(chunk_text)

            sources_info.append(
                {
                    "reference": citation_ref,
                    "source_file": source_file,
                    "source_type": source_type,
                    "page_number": page_number,
                    "chunk_id": result["id"],
                    "relevance_score": result["score"],
                }
            )

        return "\n\n".join(context_parts), sources_info

    def _create_rag_prompt(self, query: str, context: str) -> str:
        return f"""You are an AI assistant that answers questions based on provided source material. Follow these citation rules carefully:

CITATION REQUIREMENTS:
1. For each factual claim, include the citation reference in square brackets [1], [2], etc.
2. Only use information from the CONTEXT below — do not add external knowledge.
3. If you cannot find relevant information in the context, say so clearly.
4. Be precise and accurate in citations.
5. When multiple sources support a point, list all: [1], [2], [3].

CONTEXT (with citation references):
{context}

QUESTION: {query}

Please provide a comprehensive, well-cited answer:"""

    def generate_summary(
        self,
        max_chunks: int = 15,
        summary_length: str = "medium",
    ) -> RAGResult:
        try:
            summary_query = "main topics key findings important information overview"
            query_vector = self.embedding_generator.generate_query_embedding(summary_query)
            search_results = self.vector_db.search(
                query_vector=query_vector.tolist(),
                limit=max_chunks,
            )

            if not search_results:
                return RAGResult(
                    query="Document Summary",
                    response="No documents available for summarization.",
                    sources_used=[],
                    retrieval_count=0,
                )

            context, sources_info = self._format_context_with_citations(
                search_results, max_chunks, 6000
            )

            length_instructions = {
                "short": "Provide a concise 2-3 paragraph summary highlighting the most important points.",
                "medium": "Provide a comprehensive 4-5 paragraph summary covering key topics and findings.",
                "long": "Provide a detailed summary with multiple sections covering all major topics and supporting details.",
            }

            summary_prompt = f"""You are tasked with creating a summary of the provided document content. Guidelines:

1. {length_instructions.get(summary_length, length_instructions['medium'])}
2. Include citations [1], [2], etc. for all factual claims.
3. Organise information logically with clear topics.
4. Focus on the most important and relevant information.
5. Maintain accuracy and cite sources properly.

DOCUMENT CONTENT (with citation references):
{context}

Please provide a well-structured summary with proper citations:"""

            # Generate via Qwen on HF (was: DashScope / crewai LLM)
            response = self._call_hf(summary_prompt)

            return RAGResult(
                query="Document Summary",
                response=response,
                sources_used=sources_info,
                retrieval_count=len(search_results),
            )

        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return RAGResult(
                query="Document Summary",
                response=f"Error generating summary: {str(e)}",
                sources_used=[],
                retrieval_count=0,
            )


if __name__ == "__main__":
    import os
    from src.document_processing.doc_processor import DocumentProcessor
    from src.embeddings.embedding_generator import EmbeddingGenerator
    from src.vector_database.milvus_vector_db import MilvusVectorDB

    # Requires HUGGINGFACEHUB_API_TOKEN in your environment / .env file
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        print("Please set HUGGINGFACEHUB_API_TOKEN environment variable")
        exit(1)

    try:
        embedding_gen = EmbeddingGenerator()
        vector_db = MilvusVectorDB()
        rag_generator = RAGGenerator(
            embedding_generator=embedding_gen,
            vector_db=vector_db,
            hf_api_token=hf_token,
        )

        test_query = "What are the main findings discussed in the documents?"
        result = rag_generator.generate_response(test_query)

        print(f"Query: {result.query}")
        print(f"Response: {result.response}")
        print(f"\nSources Used ({len(result.sources_used)}):")
        print(result.get_citation_summary())

    except Exception as e:
        print(f"Error in RAG pipeline example: {e}")