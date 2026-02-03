"""
Views for the chat application.

This module provides views for:
- Chat interface (sync and async)
- Settings configuration
- API endpoints for chat messages (sync and async)
- Index status
"""

import asyncio
import json
import logging
import uuid

from django.contrib import messages
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from chat.models import ChatMessage, RAGConfiguration
from chat.services.langchain_rag_service import (
    get_rag_service as get_langchain_rag_service,
)
from chat.services.rag_service import get_rag_service as get_legacy_rag_service
from chat.utils import get_rag_config, validate_api_keys

logger = logging.getLogger(__name__)

# Toggle between legacy and LangChain RAG service
USE_LANGCHAIN = True


def chat_view(request):
    """
    Main chat interface view (sync).

    Displays the chat interface with message history.
    """
    session_id = request.session.get("chat_session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session["chat_session_id"] = session_id

    conversation = ChatMessage.get_conversation(uuid.UUID(session_id))
    index_status = _get_index_status()

    context = {
        "session_id": session_id,
        "conversation": conversation,
        "index_status": index_status,
        "api_keys_configured": validate_api_keys(),
    }

    return render(request, "chat/chat.html", context)


async def chat_view_async(request):
    """
    Main chat interface view (async).

    Displays the chat interface with message history.
    Uses async operations for better performance.
    """
    session_id = request.session.get("chat_session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session["chat_session_id"] = session_id

    # Get conversation history asynchronously
    conversation = await asyncio.to_thread(
        ChatMessage.get_conversation, uuid.UUID(session_id)
    )

    # Get index status asynchronously
    index_status = await _get_index_status_async()

    context = {
        "session_id": session_id,
        "conversation": conversation,
        "index_status": index_status,
        "api_keys_configured": validate_api_keys(),
    }

    return render(request, "chat/chat.html", context)


def settings_view(request):
    """
    Settings view for configuring RAG parameters.
    """
    config = RAGConfiguration.objects.order_by("-created_at").first()

    if request.method == "POST":
        if config:
            config.rag_folder_name = request.POST.get(
                "rag_folder_name", "default"
            )
            config.embedding_provider = request.POST.get(
                "embedding_provider", "openai"
            )
            config.llm_provider = request.POST.get("llm_provider", "openai")
            config.chunk_size = int(request.POST.get("chunk_size", 1000))
            config.chunk_overlap = int(request.POST.get("chunk_overlap", 200))
            config.top_k = int(request.POST.get("top_k", 5))
            config.save()
            messages.success(request, "Configuration updated successfully!")
        else:
            RAGConfiguration.objects.create(
                rag_folder_name=request.POST.get("rag_folder_name", "default"),
                embedding_provider=request.POST.get(
                    "embedding_provider", "openai"
                ),
                llm_provider=request.POST.get("llm_provider", "openai"),
                chunk_size=int(request.POST.get("chunk_size", 1000)),
                chunk_overlap=int(request.POST.get("chunk_overlap", 200)),
                top_k=int(request.POST.get("top_k", 5)),
            )
            messages.success(request, "Configuration created successfully!")

        return redirect("chat:settings")

    context = {
        "config": config,
        "api_keys_configured": validate_api_keys(),
    }

    return render(request, "chat/settings.html", context)


@csrf_exempt
@require_http_methods(["POST"])
def chat_api(request):
    """
    API endpoint for sending chat messages (sync).

    Expects JSON body with:
        - message: The user's message
        - session_id: Optional session ID
        - use_langchain: Optional boolean to toggle LangChain service

    Returns JSON response with:
        - answer: The AI's response
        - sources: List of source documents
        - context: List of context chunks
    """
    try:
        data = json.loads(request.body)
        user_message = data.get("message", "").strip()
        session_id = data.get("session_id")
        use_langchain = data.get("use_langchain", USE_LANGCHAIN)

        if not user_message:
            return JsonResponse({"error": "Message is required"}, status=400)

        if not session_id:
            session_id = request.session.get("chat_session_id")
            if not session_id:
                session_id = str(uuid.uuid4())
                request.session["chat_session_id"] = session_id

        session_uuid = uuid.UUID(session_id)

        # Save user message
        ChatMessage.create_user_message(session_uuid, user_message)

        # Get RAG service
        config = get_rag_config()

        if use_langchain:
            rag_service = get_langchain_rag_service(config)
        else:
            rag_service = get_legacy_rag_service(config)

        # Check if index is ready
        if not rag_service.is_index_ready():
            return JsonResponse(
                {
                    "error": "No documents indexed. Please run the index_documents command first.",
                    "answer": "I don't have any documents to search through. Please index some documents first.",
                    "sources": [],
                    "context": [],
                }
            )

        # Query RAG service
        result = rag_service.query(user_message, session_id)

        # Save assistant message
        ChatMessage.create_assistant_message(
            session_uuid,
            result["answer"],
            context=result.get("context", []),
            sources=result.get("sources", []),
        )

        return JsonResponse(
            {
                "answer": result["answer"],
                "sources": result.get("sources", []),
                "context": result.get("context", []),
                "session_id": session_id,
            }
        )

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        logger.exception(f"Error in chat API: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
async def chat_api_async(request):
    """
    API endpoint for sending chat messages (async).

    Supports streaming responses and uses async operations throughout.
    """
    try:
        data = json.loads(request.body)
        user_message = data.get("message", "").strip()
        session_id = data.get("session_id")
        stream = data.get("stream", False)
        use_langchain = data.get("use_langchain", USE_LANGCHAIN)

        if not user_message:
            return JsonResponse({"error": "Message is required"}, status=400)

        if not session_id:
            session_id = request.session.get("chat_session_id")
            if not session_id:
                session_id = str(uuid.uuid4())
                request.session["chat_session_id"] = session_id

        session_uuid = uuid.UUID(session_id)

        # Save user message asynchronously
        await asyncio.to_thread(
            ChatMessage.create_user_message, session_uuid, user_message
        )

        # Get RAG service
        config = get_rag_config()

        if use_langchain:
            rag_service = get_langchain_rag_service(config)
        else:
            rag_service = get_legacy_rag_service(config)

        # Check if index is ready
        if not rag_service.is_index_ready():
            return JsonResponse(
                {
                    "error": "No documents indexed.",
                    "answer": "I don't have any documents to search through. Please index some documents first.",
                    "sources": [],
                    "context": [],
                }
            )

        # Handle streaming response
        if stream:
            return _stream_response(
                rag_service, user_message, session_id, session_uuid
            )

        # Non-streaming async response
        result = await rag_service.aquery(user_message, session_id)

        # Save assistant message
        await asyncio.to_thread(
            ChatMessage.create_assistant_message,
            session_uuid,
            result["answer"],
            context=result.get("context", []),
            sources=result.get("sources", []),
        )

        return JsonResponse(
            {
                "answer": result["answer"],
                "sources": result.get("sources", []),
                "context": result.get("context", []),
                "session_id": session_id,
            }
        )

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        logger.exception(f"Error in async chat API: {e}")
        return JsonResponse({"error": str(e)}, status=500)


def _stream_response(
    rag_service, user_message: str, session_id: str, session_uuid
):
    """
    Create a streaming response for the chat API.

    Args:
        rag_service: RAG service instance with streaming support
        user_message: The user's message
        session_id: Session ID
        session_uuid: UUID version of session ID

    Returns:
        StreamingHttpResponse
    """

    async def generate():
        try:
            async for chunk in rag_service.astream(user_message, session_id):
                yield chunk
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            yield f"Error: {str(e)}"

    response = StreamingHttpResponse(
        generate(), content_type="text/event-stream"
    )
    response["X-Accel-Buffering"] = "no"
    return response


@require_http_methods(["GET"])
def conversation_api(request, session_id):
    """
    API endpoint for retrieving conversation history.
    """
    try:
        session_uuid = uuid.UUID(session_id)
        messages_qs = ChatMessage.get_conversation(session_uuid)

        message_list = []
        for msg in messages_qs:
            message_list.append(
                {
                    "id": str(msg.id),
                    "role": msg.role,
                    "content": msg.content,
                    "sources": msg.sources,
                    "created_at": msg.created_at.isoformat(),
                }
            )

        return JsonResponse(
            {
                "messages": message_list,
                "session_id": session_id,
            }
        )

    except ValueError:
        return JsonResponse({"error": "Invalid session ID"}, status=400)
    except Exception as e:
        logger.exception(f"Error retrieving conversation: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@require_http_methods(["GET"])
def index_status_api(request):
    """
    API endpoint for getting index status.
    """
    try:
        config = get_rag_config()

        if USE_LANGCHAIN:
            rag_service = get_langchain_rag_service(config)
            stats = rag_service.get_index_stats()
        else:
            rag_service = get_legacy_rag_service(config)
            stats = rag_service.get_index_stats()

        return JsonResponse(
            {
                "is_ready": rag_service.is_index_ready(),
                "vector_count": stats.get("vector_count", 0),
                "dimension": stats.get("dimension"),
                "index_path": stats.get("index_path"),
            }
        )

    except Exception as e:
        logger.exception(f"Error getting index status: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@require_http_methods(["GET"])
def new_session_api(request):
    """
    API endpoint for creating a new chat session.
    """
    session_id = str(uuid.uuid4())
    request.session["chat_session_id"] = session_id

    return JsonResponse(
        {
            "session_id": session_id,
        }
    )


def _get_index_status():
    """
    Get the current index status (sync).
    """
    try:
        config = get_rag_config()

        if USE_LANGCHAIN:
            rag_service = get_langchain_rag_service(config)
        else:
            rag_service = get_legacy_rag_service(config)

        stats = rag_service.get_index_stats()

        return {
            "is_ready": rag_service.is_index_ready(),
            "vector_count": stats.get("vector_count", 0),
            "dimension": stats.get("dimension"),
            "index_path": stats.get("index_path"),
        }
    except Exception as e:
        logger.error(f"Error getting index status: {e}")
        return {
            "is_ready": False,
            "vector_count": 0,
            "dimension": None,
            "index_path": None,
        }


async def _get_index_status_async():
    """
    Get the current index status (async).
    """
    try:
        config = get_rag_config()

        if USE_LANGCHAIN:
            rag_service = get_langchain_rag_service(config)
        else:
            rag_service = get_legacy_rag_service(config)

        # Run in thread pool for sync service
        stats = await asyncio.to_thread(rag_service.get_index_stats)

        return {
            "is_ready": rag_service.is_index_ready(),
            "vector_count": stats.get("vector_count", 0),
            "dimension": stats.get("dimension"),
            "index_path": stats.get("index_path"),
        }
    except Exception as e:
        logger.error(f"Error getting index status: {e}")
        return {
            "is_ready": False,
            "vector_count": 0,
            "dimension": None,
            "index_path": None,
        }
