#!/bin/sh

# Check for the file
if [ ! -f /path/to/test.txt ]; then
    echo "File not found, raising an issue."
    curl -X POST -H "Authorization: token $GIT_TOKEN" \
         -H "Accept: application/vnd.github.v3+json" \
         https://api.github.com/repos/christianhilscher/special-broccoli/issues \
         -d '{"title":"File test.txt Not Found","body":"The file test.txt was not found."}'
else
    echo "File found. No action needed."
fi
