# v2.18.0 ModelEvo 经验库服务镜像(dev 文档 §8.4)
# 构建上下文为 skill 根目录:
#   docker build -f service/Dockerfile -t modelevo-library:2.18.0 .
FROM python:3.11-slim

WORKDIR /app
COPY service/requirements.txt /app/service/requirements.txt
RUN pip install --no-cache-dir -r /app/service/requirements.txt \
        "psycopg[binary]>=3.1" "s3fs>=2023.1"

# 服务只需要 runtime/library 与 service/;runtime/__init__.py 不引入重依赖
COPY runtime/__init__.py /app/runtime/__init__.py
COPY runtime/library /app/runtime/library
COPY service /app/service

ENV LIBRARY_ROOT=/var/lib/modelevo-library
VOLUME ["/var/lib/modelevo-library"]
EXPOSE 18765

# 容器内监听 0.0.0.0,由宿主 / 编排层控制暴露范围
CMD ["sh", "-c", "python service/server.py --root \"$LIBRARY_ROOT\" --host 0.0.0.0 --port 18765 ${LIBRARY_DB_URL:+--db-url \"$LIBRARY_DB_URL\"} ${LIBRARY_ARTIFACTS_URL:+--artifacts-url \"$LIBRARY_ARTIFACTS_URL\"}"]
