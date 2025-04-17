import inspect
import logging
import sys
import time
import uuid
from argparse import ArgumentError
from typing import Any, Dict, Optional

import boto3
import botocore

from newrelic_telemetry_sdk import Span

from newrelic_bedrock_observability.bedrock_monitoring import (
    BEDROCK_EVENT_NAME,
    BEDROCK_SUMMARY_EVENT_NAME,
    BEDROCK_EMBEDDING_EVENT_NAME,
    bedrock_monitor,
)
from newrelic_bedrock_observability.build_events import (
    build_invocation_events,
    build_invocation_error_events,
    build_embedding_event,
    build_embedding_error_event,
)
from newrelic_bedrock_observability.error_handling_decorator import handle_errors

logger = logging.getLogger("newrelic_bedrock_observability")


def _patched_call(original_fn, patched_fn):
    """
    함수를 패치하여 모니터링 기능 추가
    
    Args:
        original_fn: 원본 함수
        patched_fn: 패치할 함수
    
    Returns:
        패치된 함수
    """
    if hasattr(original_fn, "is_patched_by_monitor"):
        return original_fn

    def _inner_patch(*args, **kwargs):
        try:
            return patched_fn(original_fn, *args, **kwargs)
        except Exception as ex:
            raise ex

    _inner_patch.is_patched_by_monitor = True

    return _inner_patch


def _patched_call_async(original_fn, patched_fn):
    """
    비동기 함수를 패치하여 모니터링 기능 추가
    
    Args:
        original_fn: 원본 비동기 함수
        patched_fn: 패치할 비동기 함수
    
    Returns:
        패치된 비동기 함수
    """
    if hasattr(original_fn, "is_patched_by_monitor"):
        return original_fn

    async def _inner_patch(*args, **kwargs):
        try:
            return await patched_fn(original_fn, *args, **kwargs)
        except Exception as ex:
            raise ex

    _inner_patch.is_patched_by_monitor = True

    return _inner_patch


def patcher_invoke_model(original_fn, *args, **kwargs):
    """
    Bedrock InvokeModel 함수 패치
    
    Args:
        original_fn: 원본 InvokeModel 함수
        args: 위치 인자
        kwargs: 키워드 인자
    
    Returns:
        API 호출 결과
    """
    logger.debug(
        f"Running the original function: '{original_fn.__qualname__}'. args:{args}; kwargs: {kwargs}"
    )

    result, time_delta = None, None
    span = bedrock_monitor.create_span()
    try:
        timestamp = time.time()
        result = original_fn(*args, **kwargs)
        time_delta = time.time() - timestamp
    except Exception as ex:
        span.finish()
        handle_invoke_model(result, kwargs, ex, time_delta, span)
        raise ex
    span.finish()

    logger.debug(f"Finished running function: '{original_fn.__qualname__}'.")

    return handle_invoke_model(result, kwargs, None, time_delta, span)


async def patcher_invoke_model_async(original_fn, *args, **kwargs):
    """
    Bedrock InvokeModel 비동기 함수 패치
    
    Args:
        original_fn: 원본 InvokeModel 비동기 함수
        args: 위치 인자
        kwargs: 키워드 인자
    
    Returns:
        API 호출 결과
    """
    logger.debug(
        f"Running the original function: '{original_fn.__qualname__}'. args:{args}; kwargs: {kwargs}"
    )
    result, time_delta = None, None
    span = bedrock_monitor.create_span()
    try:
        timestamp = time.time()
        result = await original_fn(*args, **kwargs)
        time_delta = time.time() - timestamp
    except Exception as ex:
        span.finish()
        handle_invoke_model(result, kwargs, ex, time_delta, span)
        raise ex
    span.finish()

    logger.debug(f"Finished running function: '{original_fn.__qualname__}'.")

    return handle_invoke_model(result, kwargs, None, time_delta, span)


def patcher_invoke_model_with_response_stream(original_fn, *args, **kwargs):
    """
    Bedrock InvokeModelWithResponseStream 함수 패치
    
    Args:
        original_fn: 원본 InvokeModelWithResponseStream 함수
        args: 위치 인자
        kwargs: 키워드 인자
    
    Returns:
        API 호출 결과
    """
    logger.debug(
        f"Running the original function: '{original_fn.__qualname__}'. args:{args}; kwargs: {kwargs}"
    )

    result, time_delta = None, None
    span = bedrock_monitor.create_span()
    try:
        timestamp = time.time()
        # 스트리밍 응답은 다른 방식으로 처리해야 할 수 있음
        result = original_fn(*args, **kwargs)
        time_delta = time.time() - timestamp
    except Exception as ex:
        span.finish()
        handle_invoke_model(result, kwargs, ex, time_delta, span)
        raise ex
    span.finish()

    logger.debug(f"Finished running function: '{original_fn.__qualname__}'.")

    # 스트리밍 응답 처리
    return handle_invoke_model(result, kwargs, None, time_delta, span)


async def patcher_invoke_model_with_response_stream_async(original_fn, *args, **kwargs):
    """
    Bedrock InvokeModelWithResponseStream 비동기 함수 패치
    
    Args:
        original_fn: 원본 InvokeModelWithResponseStream 비동기 함수
        args: 위치 인자
        kwargs: 키워드 인자
    
    Returns:
        API 호출 결과
    """
    logger.debug(
        f"Running the original function: '{original_fn.__qualname__}'. args:{args}; kwargs: {kwargs}"
    )
    result, time_delta = None, None
    span = bedrock_monitor.create_span()
    try:
        timestamp = time.time()
        result = await original_fn(*args, **kwargs)
        time_delta = time.time() - timestamp
    except Exception as ex:
        span.finish()
        handle_invoke_model(result, kwargs, ex, time_delta, span)
        raise ex
    span.finish()

    logger.debug(f"Finished running function: '{original_fn.__qualname__}'.")

    return handle_invoke_model(result, kwargs, None, time_delta, span)


@handle_errors
def handle_invoke_model(
    response, request, error, response_time, span: Optional[Span] = None
):
    """
    Bedrock InvokeModel/InvokeModelWithResponseStream 응답 처리
    
    Args:
        response: API 응답
        request: API 요청 파라미터
        error: 발생한 예외 (없으면 None)
        response_time: 응답 시간
        span: Span 객체
    
    Returns:
        원본 응답
    """
    events = None
    
    if error:
        events = build_invocation_error_events(request, error)
    else:
        # 응답 헤더 추출
        response_headers = {}
        if hasattr(response, "ResponseMetadata"):
            response_headers = response.ResponseMetadata.get("HTTPHeaders", {})
        
        events = build_invocation_events(
            response, request, response_headers, response_time
        )

    for event in events["messages"]:
        bedrock_monitor.record_event(event, BEDROCK_EVENT_NAME)
    
    bedrock_monitor.record_event(events["completion"], BEDROCK_SUMMARY_EVENT_NAME)
    
    if span:
        span["attributes"].update(events["completion"])
        span["attributes"]["name"] = BEDROCK_SUMMARY_EVENT_NAME
        bedrock_monitor.record_span(span)

    return response


def patcher_create_embedding(original_fn, *args, **kwargs):
    """
    Bedrock CreateEmbedding 함수 패치
    
    Args:
        original_fn: 원본 CreateEmbedding 함수
        args: 위치 인자
        kwargs: 키워드 인자
    
    Returns:
        API 호출 결과
    """
    logger.debug(
        f"Running the original function: '{original_fn.__qualname__}'. args:{args}; kwargs: {kwargs}"
    )

    result, time_delta = None, None
    span = bedrock_monitor.create_span()
    try:
        timestamp = time.time()
        result = original_fn(*args, **kwargs)
        time_delta = time.time() - timestamp
    except Exception as ex:
        span.finish()
        handle_create_embedding(result, kwargs, ex, time_delta, span)
        raise ex
    span.finish()

    logger.debug(f"Finished running function: '{original_fn.__qualname__}'.")

    return handle_create_embedding(result, kwargs, None, time_delta, span)


async def patcher_create_embedding_async(original_fn, *args, **kwargs):
    """
    Bedrock CreateEmbedding 비동기 함수 패치
    
    Args:
        original_fn: 원본 CreateEmbedding 비동기 함수
        args: 위치 인자
        kwargs: 키워드 인자
    
    Returns:
        API 호출 결과
    """
    logger.debug(
        f"Running the original function: '{original_fn.__qualname__}'. args:{args}; kwargs: {kwargs}"
    )

    result, time_delta = None, None
    span = bedrock_monitor.create_span()
    try:
        timestamp = time.time()
        result = await original_fn(*args, **kwargs)
        time_delta = time.time() - timestamp
    except Exception as ex:
        span.finish()
        handle_create_embedding(result, kwargs, ex, time_delta, span)
        raise ex
    span.finish()

    logger.debug(f"Finished running function: '{original_fn.__qualname__}'.")

    return handle_create_embedding(result, kwargs, None, time_delta, span)


@handle_errors
def handle_create_embedding(
    response, request, error, response_time, span: Optional[Span] = None
):
    """
    Bedrock CreateEmbedding 응답 처리
    
    Args:
        response: API 응답
        request: API 요청 파라미터
        error: 발생한 예외 (없으면 None)
        response_time: 응답 시간
        span: Span 객체
    
    Returns:
        원본 응답
    """
    event = None
    
    if error:
        event = build_embedding_error_event(request, error)
    else:
        # 응답 헤더 추출
        response_headers = {}
        if hasattr(response, "ResponseMetadata"):
            response_headers = response.ResponseMetadata.get("HTTPHeaders", {})
        
        event = build_embedding_event(
            response, request, response_headers, response_time
        )

    bedrock_monitor.record_event(event, BEDROCK_EMBEDDING_EVENT_NAME)
    
    if span:
        span["attributes"].update(event)
        span["attributes"]["name"] = BEDROCK_EMBEDDING_EVENT_NAME
        bedrock_monitor.record_span(span)

    return response


def perform_patch():
    """
    Bedrock 클라이언트 메소드를 패치하여 모니터링 추가
    """
    try:
        # AWS 서비스 클라이언트를 생성할 때 패치
        original_client = boto3.client
        
        def patched_client(*args, **kwargs):
            client = original_client(*args, **kwargs)
            
            # Bedrock 클라이언트인 경우에만 패치
            if args and args[0] == "bedrock-runtime":
                try:
                    # InvokeModel 패치
                    if hasattr(client, "invoke_model"):
                        client.invoke_model = _patched_call(
                            client.invoke_model, patcher_invoke_model
                        )
                    
                    # InvokeModelWithResponseStream 패치
                    if hasattr(client, "invoke_model_with_response_stream"):
                        client.invoke_model_with_response_stream = _patched_call(
                            client.invoke_model_with_response_stream, patcher_invoke_model_with_response_stream
                        )
                    
                    # CreateEmbedding 패치
                    if hasattr(client, "create_embedding"):
                        client.create_embedding = _patched_call(
                            client.create_embedding, patcher_create_embedding
                        )
                    
                    logger.info("AWS Bedrock 클라이언트 메소드 패치 완료")
                except Exception as ex:
                    logger.error(f"AWS Bedrock 클라이언트 메소드 패치 실패: {ex}")
            
            return client
        
        boto3.client = patched_client
        
        # 비동기 클라이언트도 지원하는 경우 패치
        try:
            import aioboto3
            
            original_aioclient = aioboto3.client
            
            async def patched_aioclient(*args, **kwargs):
                client = await original_aioclient(*args, **kwargs)
                
                # Bedrock 클라이언트인 경우에만 패치
                if args and args[0] == "bedrock-runtime":
                    try:
                        # 비동기 메소드 패치
                        if hasattr(client, "invoke_model"):
                            client.invoke_model = _patched_call_async(
                                client.invoke_model, patcher_invoke_model_async
                            )
                        
                        if hasattr(client, "invoke_model_with_response_stream"):
                            client.invoke_model_with_response_stream = _patched_call_async(
                                client.invoke_model_with_response_stream, patcher_invoke_model_with_response_stream_async
                            )
                        
                        if hasattr(client, "create_embedding"):
                            client.create_embedding = _patched_call_async(
                                client.create_embedding, patcher_create_embedding_async
                            )
                        
                        logger.info("AWS Bedrock 비동기 클라이언트 메소드 패치 완료")
                    except Exception as ex:
                        logger.error(f"AWS Bedrock 비동기 클라이언트 메소드 패치 실패: {ex}")
                
                return client
            
            aioboto3.client = patched_aioclient
            
        except ImportError:
            # aioboto3가 설치되지 않은 경우 무시
            pass
        
    except Exception as ex:
        logger.error(f"AWS Bedrock 클라이언트 패치 중 오류 발생: {ex}")
        raise 