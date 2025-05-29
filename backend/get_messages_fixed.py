@app.get("/get_messages/{session_id}")
async def get_messages(session_id: str):
    """Get all messages for a session.

    Args:
        session_id: Unique session identifier.

    Returns:
        List of messages for the session.

    Raises:
        HTTPException: If session_id is invalid or session not found.
    """
    validate_uuid(session_id)
    log = logger.bind(session_id=session_id)  # Fixed variable name

    # Check session existence
    session_lock: Optional[asyncio.Lock] = None
    async with global_sessions_lock:
        if session_id not in sessions:
            log.error("Session not found", session_id=session_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            )
        session_lock = per_session_locks[session_id]

    # Get messages
    async with session_lock:
        session: Session = sessions[session_id]
        
        # Convert messages to a format suitable for the frontend
        messages = [{
            "message_id": msg.message_id,
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat(),
            "metadata": msg.metadata
        } for msg in session.messages]
        
        log.info(
            "Retrieved messages for session",
            session_id=session_id,
            message_count=len(messages)
        )
        
        return messages
