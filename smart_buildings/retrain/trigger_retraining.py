import os
import requests
import datetime

git_token = os.environ.get("GIT_TOKEN", "")
repo_name = os.environ.get("REPO_NAME", "")

file_path = "/home/test.txt"

if not os.path.isfile(file_path):
    print("File not found, raising an issue.")

    github_api_url = f"https://api.github.com/repos/{repo_name}/issues"

    issue_data = {
        "title": f"Drift-Monitor: Retraining needed - {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "body": "The file test.txt was not found ",
    }
    headers = {
        "Authorization": f"token {git_token}",
        "Accept": "application/vnd.github.v3+json",
    }
    response = requests.post(github_api_url, headers=headers, json=issue_data)
    print(f"GitHub API Response Status Code: {response.status_code}")
else:
    print("File found. No action needed.")
