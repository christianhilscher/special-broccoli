# Use a base image with curl installed, like Alpine
FROM alpine:latest

# Install curl
RUN apk --no-cache add curl

# Copy the script into the container
COPY check_file.sh /check_file.sh

# Make the script executable
RUN chmod +x /check_file.sh

# Command to run when the container starts
CMD ["/check_file.sh"]
