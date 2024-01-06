FROM alpine:latest
WORKDIR /app
RUN apk --no-cache add curl
COPY drift_monitor/check_file.sh /app
RUN chmod +x /app/check_file.sh

CMD ["/app/check_file.sh"]
