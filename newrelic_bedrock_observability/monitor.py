import logging
from typing import Any, Callable, Dict, Optional

from newrelic_bedrock_observability.bedrock_monitoring import bedrock_monitor
from newrelic_bedrock_observability.patcher import perform_patch

logger = logging.getLogger("newrelic_bedrock_observability")


def initialization(
    application_name: str,
    license_key: Optional[str] = None,
    metadata: Dict[str, Any] = {},
    event_client_host: Optional[str] = None,
    parent_span_id_callback: Optional[Callable] = None,
    metadata_callback: Optional[Callable] = None,
):
    """
    AWS Bedrock 모니터링 초기화
    
    Args:
        application_name: 애플리케이션 이름
        license_key: New Relic 라이센스 키 (없으면 환경 변수에서 읽음)
        metadata: 추가 메타데이터
        event_client_host: 이벤트 클라이언트 호스트 (EU 계정 등에서 필요)
        parent_span_id_callback: 부모 스팬 ID 콜백 함수
        metadata_callback: 메타데이터 콜백 함수
    
    Returns:
        모니터 객체
    """
    bedrock_monitor.start(
        application_name,
        license_key,
        metadata,
        event_client_host,
        parent_span_id_callback,
        metadata_callback,
    )
    perform_patch()
    return bedrock_monitor 