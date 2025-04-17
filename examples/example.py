import os
import json
import boto3

from newrelic_bedrock_observability import monitor

# 모니터링 초기화
monitor.initialization(
    application_name="AWS Bedrock Observability Example",
    metadata={"environment": "development"},
)

# Bedrock 클라이언트 설정
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",  # 사용하는 리전으로 변경
)

# Claude 모델 호출 예제
def invoke_claude():
    response = bedrock_runtime.invoke_model(
        modelId="anthropic.claude-v2",  # 사용할 모델 ID
        body=json.dumps({
            "prompt": "\n\nHuman: Write a short poem about observability.\n\nAssistant:",
            "max_tokens_to_sample": 500,
            "temperature": 0.7,
            "top_p": 0.9,
        })
    )
    
    response_body = json.loads(response.get("body").read())
    print("\nClaude Response:")
    print(response_body.get("completion", "No completion returned"))


# Titan 텍스트 모델 호출 예제
def invoke_titan():
    response = bedrock_runtime.invoke_model(
        modelId="amazon.titan-text-express-v1",  # 사용할 모델 ID
        body=json.dumps({
            "inputText": "Write a short poem about observability.",
            "textGenerationConfig": {
                "maxTokenCount": 500,
                "temperature": 0.7,
                "topP": 0.9,
            }
        })
    )
    
    response_body = json.loads(response.get("body").read())
    print("\nTitan Response:")
    print(response_body.get("results")[0].get("outputText", "No output text returned"))


# 임베딩 생성 예제
def create_embedding():
    response = bedrock_runtime.invoke_model(
        modelId="amazon.titan-embed-text-v1",  # 임베딩 모델 ID
        body=json.dumps({
            "inputText": "This is a sample text for embedding generation.",
        })
    )
    
    response_body = json.loads(response.get("body").read())
    print("\nEmbedding Created:")
    embedding = response_body.get("embedding", [])
    print(f"Embedding dimensions: {len(embedding)}")


# 모델 스트리밍 응답 호출 예제
def invoke_model_with_streaming():
    response = bedrock_runtime.invoke_model_with_response_stream(
        modelId="anthropic.claude-v2",  # 사용할 모델 ID
        body=json.dumps({
            "prompt": "\n\nHuman: Explain in 3 paragraphs what is AI observability and why it's important.\n\nAssistant:",
            "max_tokens_to_sample": 500,
            "temperature": 0.7,
            "top_p": 0.9,
        })
    )
    
    stream = response.get("body")
    
    # 스트리밍 응답 처리
    print("\nStreaming Response:")
    for event in stream:
        chunk = event.get("chunk")
        if chunk:
            chunk_data = json.loads(chunk.get("bytes").decode())
            print(chunk_data.get("completion", ""), end="", flush=True)
    print("\n")


if __name__ == "__main__":
    # API 키와 리전이 설정되어 있는지 확인
    if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        print("AWS 인증 정보가 환경 변수에 설정되어 있지 않습니다.")
        print("AWS_ACCESS_KEY_ID와 AWS_SECRET_ACCESS_KEY를 설정하세요.")
    else:
        try:
            # 예제 실행
            invoke_claude()
            invoke_titan()
            create_embedding()
            invoke_model_with_streaming()
        except Exception as e:
            print(f"오류 발생: {e}") 