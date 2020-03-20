gcloud ai-platform jobs submit training YOUR_JOB_ID \
--module-name=trainer.task \
--package-path=./trainer \
--job-dir=gs://your-storage-bucket \
--region=us-central1 \
--config=trainer/cloudml-gpu.yaml \
--python-version 3.5
