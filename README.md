## ML Model Deployment and Retraining Pipeline Demo

### Overview
This project demonstrates an automated machine learning (ML) model deployment and retraining pipeline. Developed within a constrained timeframe and using fundamental tools such as Docker and GitHub Actions, the pipeline showcases an efficient approach to handling data and model drift.

### Workflow
The pipeline operates as follows:

1. **Model Serving**: A Docker container, hosting a FastAPI endpoint, is deployed on an EC2 instance to serve ML model predictions.

2. **Drift Monitoring**: A regular monitoring process checks for data and model drift.

3. **Issue Triggering**: Upon detection of drift, an issue is automatically created in the repository, possibly including additional diagnostic information.

4. **Model Retraining**: The creation of the issue triggers a retraining workflow. This process retrains the ML model using the latest available data.

5. **Branch Creation and PR**: Post successful retraining, a new branch is created, the updated model is committed, and a pull request (PR) is raised for review.

6. **Model Evaluation and Merge**: A key test in the PR is the comparison of the new model's performance against the current production model. If the new model shows improvement, the PR is merged into the master branch.

7. **Deployment**: The merge into master initiates a deployment action. This action involves building a new Docker container with the updated model, stopping the previous container, and launching the new container to resume prediction services.

### Automation and Self-Healing
This pipeline exemplifies an automated, self-healing approach to ML model management, efficiently addressing both data and model drift with minimal manual intervention.
