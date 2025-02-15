The following tutorial was used for deployment:
https://cloud.google.com/run/docs/quickstarts/build-and-deploy/deploy-python-service

Region was set to asia-southeast1

After that memory allocation was increased to 1GB with the following command:
gcloud run services update kebunjio-flask --memory=1Gi --region=asia-southeast1

It is currently deployed on the following URL:
http://kebunjio-flask-819110013955.asia-southeast1.run.app/