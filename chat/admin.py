"""
Django admin configuration for the chat application.
"""

from django.contrib import admin
from .models import RAGConfiguration, ChatMessage


@admin.register(RAGConfiguration)
class RAGConfigurationAdmin(admin.ModelAdmin):
    """
    Admin interface for RAGConfiguration model.
    """
    list_display = [
        'rag_folder_name',
        'embedding_provider',
        'llm_provider',
        'chunk_size',
        'chunk_overlap',
        'top_k',
        'created_at',
        'updated_at',
    ]
    list_filter = [
        'embedding_provider',
        'llm_provider',
        'created_at',
    ]
    search_fields = [
        'rag_folder_name',
    ]
    readonly_fields = [
        'created_at',
        'updated_at',
    ]
    fieldsets = (
        ('Document Configuration', {
            'fields': ('rag_folder_name',)
        }),
        ('Provider Configuration', {
            'fields': ('embedding_provider', 'llm_provider')
        }),
        ('Chunking Configuration', {
            'fields': ('chunk_size', 'chunk_overlap', 'top_k')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    ordering = ['-created_at']


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    """
    Admin interface for ChatMessage model.
    """
    list_display = [
        'id',
        'session_id',
        'role',
        'content_preview',
        'created_at',
    ]
    list_filter = [
        'role',
        'created_at',
    ]
    search_fields = [
        'content',
        'session_id',
    ]
    readonly_fields = [
        'id',
        'session_id',
        'created_at',
        'context_summary',
    ]
    fieldsets = (
        ('Message Information', {
            'fields': ('id', 'session_id', 'role', 'created_at')
        }),
        ('Content', {
            'fields': ('content',)
        }),
        ('Context & Sources', {
            'fields': ('context_used', 'sources', 'context_summary'),
            'classes': ('collapse',)
        }),
    )
    ordering = ['-created_at']
    
    def content_preview(self, obj):
        """Display a preview of the message content."""
        preview = obj.content[:100]
        if len(obj.content) > 100:
            preview += '...'
        return preview
    content_preview.short_description = 'Content'
    
    def context_summary(self, obj):
        """Display a summary of the context used."""
        return obj.get_context_summary()
    context_summary.short_description = 'Context Summary'
    
    def get_queryset(self, request):
        """Optimize queryset with select_related."""
        return super().get_queryset(request).select_related()
