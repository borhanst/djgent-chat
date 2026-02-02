"""
Views for the chat application.

This module provides views for:
- Chat interface
- Settings configuration
- API endpoints for chat messages
- Index status
"""

import uuid
import json
import logging
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib import messages
from django.db.models import Q

from chat.models import RAGConfiguration, ChatMessage
from chat.services.rag_service import get_rag_service
from chat.utils import get_rag_config, validate_api_keys

logger = logging.getLogger(__name__)


def chat_view(request):
    """
    Main chat interface view.
    
    Displays the chat interface with message history.
    """
    # Get or create session ID
    session_id = request.session.get('chat_session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session['chat_session_id'] = session_id
    
    # Get conversation history
    conversation = ChatMessage.get_conversation(uuid.UUID(session_id))
    
    # Get index status
    index_status = _get_index_status()
    
    context = {
        'session_id': session_id,
        'conversation': conversation,
        'index_status': index_status,
        'api_keys_configured': validate_api_keys(),
    }
    
    return render(request, 'chat/chat.html', context)


def settings_view(request):
    """
    Settings view for configuring RAG parameters.
    """
    # Get current configuration
    config = RAGConfiguration.objects.order_by('-created_at').first()
    
    if request.method == 'POST':
        # Create or update configuration
        if config:
            # Update existing
            config.rag_folder_name = request.POST.get('rag_folder_name', 'default')
            config.embedding_provider = request.POST.get('embedding_provider', 'openai')
            config.llm_provider = request.POST.get('llm_provider', 'openai')
            config.chunk_size = int(request.POST.get('chunk_size', 1000))
            config.chunk_overlap = int(request.POST.get('chunk_overlap', 200))
            config.top_k = int(request.POST.get('top_k', 5))
            config.save()
            messages.success(request, 'Configuration updated successfully!')
        else:
            # Create new
            RAGConfiguration.objects.create(
                rag_folder_name=request.POST.get('rag_folder_name', 'default'),
                embedding_provider=request.POST.get('embedding_provider', 'openai'),
                llm_provider=request.POST.get('llm_provider', 'openai'),
                chunk_size=int(request.POST.get('chunk_size', 1000)),
                chunk_overlap=int(request.POST.get('chunk_overlap', 200)),
                top_k=int(request.POST.get('top_k', 5)),
            )
            messages.success(request, 'Configuration created successfully!')
        
        return redirect('chat:settings')
    
    context = {
        'config': config,
        'api_keys_configured': validate_api_keys(),
    }
    
    return render(request, 'chat/settings.html', context)


@csrf_exempt
@require_http_methods(["POST"])
def chat_api(request):
    """
    API endpoint for sending chat messages.
    
    Expects JSON body with:
        - message: The user's message
        - session_id: Optional session ID (uses current session if not provided)
    
    Returns JSON response with:
        - answer: The AI's response
        - sources: List of source documents
        - context: List of context chunks
    """
    try:
        # Parse request body
        data = json.loads(request.body)
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id')
        
        if not user_message:
            return JsonResponse({
                'error': 'Message is required'
            }, status=400)
        
        # Get or use session ID
        if not session_id:
            session_id = request.session.get('chat_session_id')
            if not session_id:
                session_id = str(uuid.uuid4())
                request.session['chat_session_id'] = session_id
        
        session_uuid = uuid.UUID(session_id)
        
        # Save user message
        ChatMessage.create_user_message(session_uuid, user_message)
        
        # Get RAG service
        config = get_rag_config()
        rag_service = get_rag_service(config)
        
        # Check if index is ready
        if not rag_service.is_index_ready():
            return JsonResponse({
                'error': 'No documents indexed. Please run the index_documents command first.',
                'answer': 'I don\'t have any documents to search through. Please index some documents first.',
                'sources': [],
                'context': [],
            })
        
        # Query RAG service
        result = rag_service.query(user_message, session_id)
        
        # Save assistant message
        ChatMessage.create_assistant_message(
            session_uuid,
            result['answer'],
            context=result['context'],
            sources=result['sources']
        )
        
        return JsonResponse({
            'answer': result['answer'],
            'sources': result['sources'],
            'context': result['context'],
            'session_id': session_id,
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'error': 'Invalid JSON'
        }, status=400)
    except Exception as e:
        logger.exception(f"Error in chat API: {e}")
        return JsonResponse({
            'error': str(e)
        }, status=500)


@require_http_methods(["GET"])
def conversation_api(request, session_id):
    """
    API endpoint for retrieving conversation history.
    
    Args:
        session_id: The session UUID.
    
    Returns JSON response with:
        - messages: List of messages in the conversation
    """
    try:
        session_uuid = uuid.UUID(session_id)
        messages = ChatMessage.get_conversation(session_uuid)
        
        message_list = []
        for msg in messages:
            message_list.append({
                'id': str(msg.id),
                'role': msg.role,
                'content': msg.content,
                'sources': msg.sources,
                'created_at': msg.created_at.isoformat(),
            })
        
        return JsonResponse({
            'messages': message_list,
            'session_id': session_id,
        })
        
    except ValueError:
        return JsonResponse({
            'error': 'Invalid session ID'
        }, status=400)
    except Exception as e:
        logger.exception(f"Error retrieving conversation: {e}")
        return JsonResponse({
            'error': str(e)
        }, status=500)


@require_http_methods(["GET"])
def index_status_api(request):
    """
    API endpoint for getting index status.
    
    Returns JSON response with:
        - is_ready: Whether the index is ready
        - vector_count: Number of vectors in the index
        - dimension: Dimension of the vectors
        - index_path: Path to the index
    """
    try:
        config = get_rag_config()
        rag_service = get_rag_service(config)
        
        stats = rag_service.get_index_status() if hasattr(rag_service, 'get_index_status') else rag_service.get_index_stats()
        
        return JsonResponse({
            'is_ready': rag_service.is_index_ready(),
            'vector_count': stats.get('vector_count', 0),
            'dimension': stats.get('dimension'),
            'index_path': stats.get('index_path'),
        })
        
    except Exception as e:
        logger.exception(f"Error getting index status: {e}")
        return JsonResponse({
            'error': str(e)
        }, status=500)


@require_http_methods(["GET"])
def new_session_api(request):
    """
    API endpoint for creating a new chat session.
    
    Returns JSON response with:
        - session_id: The new session ID
    """
    session_id = str(uuid.uuid4())
    request.session['chat_session_id'] = session_id
    
    return JsonResponse({
        'session_id': session_id,
    })


def _get_index_status():
    """
    Get the current index status.
    
    Returns:
        A dictionary with index status information.
    """
    try:
        config = get_rag_config()
        rag_service = get_rag_service(config)
        
        stats = rag_service.get_index_stats()
        
        return {
            'is_ready': rag_service.is_index_ready(),
            'vector_count': stats.get('vector_count', 0),
            'dimension': stats.get('dimension'),
            'index_path': stats.get('index_path'),
        }
    except Exception as e:
        logger.error(f"Error getting index status: {e}")
        return {
            'is_ready': False,
            'vector_count': 0,
            'dimension': None,
            'index_path': None,
        }
