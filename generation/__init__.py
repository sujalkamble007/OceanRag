# generation/__init__.py

from generation.prompt_builder import (
    build_prompt,
    build_context_block,
    extract_sources,
    format_answer_with_sources,
    build_hf_prompt_string,
    SYSTEM_PROMPT
)
from generation.llm_handler import (
    LLM_CONFIGS,
    get_available_llms,
    generate_answer,
    stream_answer,
    print_llm_response
)
from generation.answer_store import (
    save_qa,
    save_comparison,
    print_qa_history
)
from generation.generation_pipeline import (
    run_rag_query,
    stream_rag_query,
    run_multimodel_comparison,
    print_rag_result
)
