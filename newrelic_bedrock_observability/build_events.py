import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from newrelic_bedrock_observability.bedrock_monitoring import (
    BEDROCK_EVENT_NAME,
    BEDROCK_SUMMARY_EVENT_NAME,
    BEDROCK_EMBEDDING_EVENT_NAME,
)

logger = logging.getLogger("newrelic_bedrock_observability")


def build_invocation_events(
    response: Any, 
    request: Dict[str, Any], 
    response_headers: Optional[Dict[str, Any]] = None, 
    response_time: Optional[float] = None
) -> Dict[str, Any]:
    """
    Bedrock InvokeModel/InvokeModelWithResponseStream 응답으로부터 이벤트 생성
    
    Args:
        response: Bedrock API 응답
        request: API 요청 파라미터
        response_headers: 응답 헤더
        response_time: 응답 시간
    
    Returns:
        생성된 이벤트 데이터 딕셔너리
    """
    try:
        # 요청 정보
        model_id = request.get("modelId", "unknown")
        model_provider = model_id.split(".")[0] if "." in model_id else "unknown"
        
        # 공통 이벤트 데이터
        common_data = {
            "request_id": str(uuid.uuid4()),
            "model_id": model_id,
            "model_provider": model_provider,
            "response_time": response_time if response_time is not None else 0,
        }
        
        # 요청 바디 파싱
        request_body = {}
        if "body" in request:
            try:
                if isinstance(request["body"], bytes):
                    request_body = json.loads(request["body"].decode("utf-8"))
                elif isinstance(request["body"], str):
                    request_body = json.loads(request["body"])
                else:
                    request_body = request["body"]
            except Exception as ex:
                logger.warning(f"Failed to parse request body: {ex}")
        
        # 응답 바디 파싱
        response_body = {}
        if hasattr(response, "body"):
            try:
                body_content = response.body.read() if hasattr(response.body, "read") else response.body
                if isinstance(body_content, bytes):
                    response_body = json.loads(body_content.decode("utf-8"))
                else:
                    response_body = body_content
            except Exception as ex:
                logger.warning(f"Failed to parse response body: {ex}")
        
        # 모델별 응답 처리
        messages = []
        completion_data = {**common_data}
        
        # Claude 모델 처리
        if "anthropic" in model_id.lower():
            if "prompt" in request_body:
                completion_data["prompt"] = request_body.get("prompt", "")
                completion_data["prompt_tokens"] = len(completion_data["prompt"].split())
            
            if "messages" in request_body:
                for i, msg in enumerate(request_body.get("messages", [])):
                    message_data = {
                        **common_data,
                        "message_index": i,
                        "role": msg.get("role", "unknown"),
                        "content": msg.get("content", ""),
                    }
                    messages.append(message_data)
                
            if "completion" in response_body:
                completion_data["completion"] = response_body.get("completion", "")
                completion_data["completion_tokens"] = len(completion_data["completion"].split())
            
            if "content" in response_body:
                completion_data["completion"] = response_body.get("content", "")
                completion_data["completion_tokens"] = len(completion_data["completion"].split())
                
        # Titan 모델 처리
        elif "titan" in model_id.lower():
            if "inputText" in request_body:
                completion_data["prompt"] = request_body.get("inputText", "")
                completion_data["prompt_tokens"] = len(completion_data["prompt"].split())
            
            if "results" in response_body:
                completion_data["completion"] = response_body.get("results", [{}])[0].get("outputText", "")
                completion_data["completion_tokens"] = len(completion_data["completion"].split())
                
        # LLama2 모델 처리
        elif "llama" in model_id.lower():
            if "prompt" in request_body:
                completion_data["prompt"] = request_body.get("prompt", "")
                completion_data["prompt_tokens"] = len(completion_data["prompt"].split())
            
            if "generation" in response_body:
                completion_data["completion"] = response_body.get("generation", "")
                completion_data["completion_tokens"] = len(completion_data["completion"].split())
        
        # 기타 모델 (일반적인 처리)
        else:
            # 일반적인 프롬프트 처리
            if "prompt" in request_body:
                completion_data["prompt"] = request_body.get("prompt", "")
                completion_data["prompt_tokens"] = len(completion_data["prompt"].split())
            elif "inputText" in request_body:
                completion_data["prompt"] = request_body.get("inputText", "")
                completion_data["prompt_tokens"] = len(completion_data["prompt"].split())
            
            # 응답 처리
            if "completion" in response_body:
                completion_data["completion"] = response_body.get("completion", "")
                completion_data["completion_tokens"] = len(completion_data["completion"].split())
            elif "generation" in response_body:
                completion_data["completion"] = response_body.get("generation", "")
                completion_data["completion_tokens"] = len(completion_data["completion"].split())
            elif "outputText" in response_body:
                completion_data["completion"] = response_body.get("outputText", "")
                completion_data["completion_tokens"] = len(completion_data["completion"].split())
        
        # 토큰 개수 합계
        completion_data["total_tokens"] = completion_data.get("prompt_tokens", 0) + completion_data.get("completion_tokens", 0)
        
        # 응답 헤더 처리
        if response_headers:
            if "x-amzn-requestid" in response_headers:
                completion_data["aws_request_id"] = response_headers.get("x-amzn-requestid")
            
            if "x-amzn-bedrock-invocation-latency" in response_headers:
                completion_data["aws_invocation_latency"] = response_headers.get("x-amzn-bedrock-invocation-latency")
        
        return {
            "messages": messages,
            "completion": completion_data
        }
    except Exception as ex:
        logger.error(f"Error building invocation events: {ex}")
        return {
            "messages": [],
            "completion": {
                "error": str(ex),
                "model_id": request.get("modelId", "unknown"),
            }
        }


def build_invocation_error_events(
    request: Dict[str, Any], 
    error: Exception
) -> Dict[str, Any]:
    """
    Bedrock API 호출 오류에 대한 이벤트 생성
    
    Args:
        request: API 요청 파라미터
        error: 발생한 예외
    
    Returns:
        오류 이벤트 데이터 딕셔너리
    """
    try:
        # 요청 정보
        model_id = request.get("modelId", "unknown")
        model_provider = model_id.split(".")[0] if "." in model_id else "unknown"
        
        # 오류 이벤트 데이터
        error_data = {
            "request_id": str(uuid.uuid4()),
            "model_id": model_id,
            "model_provider": model_provider,
            "error": str(error),
            "error_type": error.__class__.__name__,
        }
        
        # 요청 바디 파싱
        if "body" in request:
            try:
                if isinstance(request["body"], bytes):
                    body = json.loads(request["body"].decode("utf-8"))
                elif isinstance(request["body"], str):
                    body = json.loads(request["body"])
                else:
                    body = request["body"]
                
                # 프롬프트 정보 추가
                if "prompt" in body:
                    error_data["prompt"] = body.get("prompt", "")
                elif "inputText" in body:
                    error_data["prompt"] = body.get("inputText", "")
                
                # 메시지 정보 추가
                if "messages" in body:
                    messages = []
                    for i, msg in enumerate(body.get("messages", [])):
                        message_data = {
                            **error_data,
                            "message_index": i,
                            "role": msg.get("role", "unknown"),
                            "content": msg.get("content", ""),
                        }
                        messages.append(message_data)
                    return {
                        "messages": messages,
                        "completion": error_data
                    }
            except Exception as ex:
                logger.warning(f"Failed to parse request body in error event: {ex}")
        
        return {
            "messages": [],
            "completion": error_data
        }
    except Exception as ex:
        logger.error(f"Error building error events: {ex}")
        return {
            "messages": [],
            "completion": {
                "error": str(ex),
                "original_error": str(error),
                "model_id": request.get("modelId", "unknown"),
            }
        }


def build_embedding_event(
    response: Any, 
    request: Dict[str, Any], 
    response_headers: Optional[Dict[str, Any]] = None, 
    response_time: Optional[float] = None
) -> Dict[str, Any]:
    """
    Bedrock 임베딩 API 응답으로부터 이벤트 생성
    
    Args:
        response: Bedrock API 응답
        request: API 요청 파라미터
        response_headers: 응답 헤더
        response_time: 응답 시간
    
    Returns:
        생성된 임베딩 이벤트 데이터
    """
    try:
        # 요청 정보
        model_id = request.get("modelId", "unknown")
        model_provider = model_id.split(".")[0] if "." in model_id else "unknown"
        
        # 공통 이벤트 데이터
        embedding_data = {
            "request_id": str(uuid.uuid4()),
            "model_id": model_id,
            "model_provider": model_provider,
            "response_time": response_time if response_time is not None else 0,
        }
        
        # 요청 바디 파싱
        request_body = {}
        if "body" in request:
            try:
                if isinstance(request["body"], bytes):
                    request_body = json.loads(request["body"].decode("utf-8"))
                elif isinstance(request["body"], str):
                    request_body = json.loads(request["body"])
                else:
                    request_body = request["body"]
            except Exception as ex:
                logger.warning(f"Failed to parse request body: {ex}")
        
        # 응답 바디 파싱
        response_body = {}
        if hasattr(response, "body"):
            try:
                body_content = response.body.read() if hasattr(response.body, "read") else response.body
                if isinstance(body_content, bytes):
                    response_body = json.loads(body_content.decode("utf-8"))
                else:
                    response_body = body_content
            except Exception as ex:
                logger.warning(f"Failed to parse response body: {ex}")
        
        # 입력 텍스트 처리
        if "inputText" in request_body:
            embedding_data["input_text"] = request_body.get("inputText", "")
            embedding_data["input_text_tokens"] = len(embedding_data["input_text"].split())
        elif "texts" in request_body:
            # 여러 텍스트 처리
            texts = request_body.get("texts", [])
            embedding_data["input_text_count"] = len(texts)
            if texts:
                embedding_data["input_text"] = str(texts)
                embedding_data["input_text_tokens"] = sum(len(text.split()) for text in texts)
        
        # 임베딩 차원 정보
        if "embedding" in response_body:
            embedding = response_body.get("embedding", [])
            embedding_data["embedding_dimensions"] = len(embedding)
        elif "embeddings" in response_body:
            embeddings = response_body.get("embeddings", [])
            if embeddings:
                embedding_data["embedding_count"] = len(embeddings)
                embedding_data["embedding_dimensions"] = len(embeddings[0]) if embeddings[0] else 0
        
        # 응답 헤더 처리
        if response_headers:
            if "x-amzn-requestid" in response_headers:
                embedding_data["aws_request_id"] = response_headers.get("x-amzn-requestid")
            
            if "x-amzn-bedrock-invocation-latency" in response_headers:
                embedding_data["aws_invocation_latency"] = response_headers.get("x-amzn-bedrock-invocation-latency")
        
        return embedding_data
    except Exception as ex:
        logger.error(f"Error building embedding event: {ex}")
        return {
            "error": str(ex),
            "model_id": request.get("modelId", "unknown"),
        }


def build_embedding_error_event(
    request: Dict[str, Any], 
    error: Exception
) -> Dict[str, Any]:
    """
    Bedrock 임베딩 API 호출 오류에 대한 이벤트 생성
    
    Args:
        request: API 요청 파라미터
        error: 발생한 예외
    
    Returns:
        오류 임베딩 이벤트 데이터
    """
    try:
        # 요청 정보
        model_id = request.get("modelId", "unknown")
        model_provider = model_id.split(".")[0] if "." in model_id else "unknown"
        
        # 오류 이벤트 데이터
        error_data = {
            "request_id": str(uuid.uuid4()),
            "model_id": model_id,
            "model_provider": model_provider,
            "error": str(error),
            "error_type": error.__class__.__name__,
        }
        
        # 요청 바디 파싱
        if "body" in request:
            try:
                if isinstance(request["body"], bytes):
                    body = json.loads(request["body"].decode("utf-8"))
                elif isinstance(request["body"], str):
                    body = json.loads(request["body"])
                else:
                    body = request["body"]
                
                # 입력 텍스트 처리
                if "inputText" in body:
                    error_data["input_text"] = body.get("inputText", "")
                    error_data["input_text_tokens"] = len(error_data["input_text"].split())
                elif "texts" in body:
                    # 여러 텍스트 처리
                    texts = body.get("texts", [])
                    error_data["input_text_count"] = len(texts)
                    if texts:
                        error_data["input_text"] = str(texts)
                        error_data["input_text_tokens"] = sum(len(text.split()) for text in texts)
            except Exception as ex:
                logger.warning(f"Failed to parse request body in error event: {ex}")
        
        return error_data
    except Exception as ex:
        logger.error(f"Error building embedding error event: {ex}")
        return {
            "error": str(ex),
            "original_error": str(error),
            "model_id": request.get("modelId", "unknown"),
        } 