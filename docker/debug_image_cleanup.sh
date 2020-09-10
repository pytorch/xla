IMAGE="gcr.io/tpu-pytorch/xla_debug"
DATE=$(date --date='-90 days' +"%Y-%m-%dT%H:%M:%S")

for digest in $(gcloud container images list-tags ${IMAGE} --limit=999999 --sort-by=TIMESTAMP --filter="timestamp.datetime < '${DATE}'" --format='get(digest)'); do
  echo $digest
  gcloud container images delete -q --force-delete-tags "${IMAGE}@${digest}"
done
