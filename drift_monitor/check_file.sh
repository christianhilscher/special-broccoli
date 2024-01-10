#!/bin/sh

curl -X POST -H "Authorization: token $GIT_TOKEN" \
        -H "Accept: application/vnd.github.v3+json" \
        https://api.github.com/repos/christianhilscher/special-broccoli/issues \
        -d '{"title":"Drift-Monitor: Retraining Needed","body":"The drift monitor recommends a retraining."}'

