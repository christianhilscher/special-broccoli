# General Thoughts

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

6. **Model Evaluation and Merge**: A test in the PR is the comparison of the new model's performance against the current production model. If the new model shows improvement, the PR is merged into the main branch.

7. **Deployment**: The merge into main initiates a deployment action. This action involves building a new Docker container with the updated model, stopping the previous container, and launching the new container to resume prediction services.


![workflow](https://github.com/christianhilscher/special-broccoli/assets/42186988/126f4cc6-6cdc-4c95-8532-8150cb48eb4e) 


### Automation and Self-Healing

This pipeline exemplifies an automated, self-healing approach to ML model management, efficiently addressing both data and model drift with minimal manual intervention.

### Thoughts on data

- Only have data for February which where it is likely that turning on a light is a good predictor for occupancy. This could be very different in the summer months which could lead to model drift.

- Same for winter/summer time switches

- Recommended office temperatures UK winter:
  - Occupied: 20C
  - Unoccupied: 16C-18C

- We currently have:
  - Occupied: 22C
  - Unoccupied: 20.5C

- Evaluation of model:
  - classic metrics like accuracy, precision, recall & f1-score
  - also include metric of actual energy savings:
    - look at counterfactual if nothing happened
    - provide rough estimate of savings
