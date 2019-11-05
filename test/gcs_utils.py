"""Utility methods to interact with Google Cloud Storage."""
from os import path

from google.cloud import storage

_GCS_PREFIX = "gcs://"


def write_to_gcs(service_account_filename,
                 output_gcs_path,
                 contents):
  """Write `contents` to a file in GCS.
  
  Args:
    service_account_filename (string): Filename of the JSON service account
      key to authenticate writes.
    output_gcs_path (string): GCS path to write to. Should consist of prefix,
      bucket name, and filename. E.g. "gcs://bucket_name/filename".
    contents (string): Payload to write to the file.

  Raises:
    ValueError if output_gcs_path has an unexpected format.
  """
  output_gcs_path = output_gcs_path.replace(_GCS_PREFIX, "")
  path_components = path.normpath(output_gcs_path).split(path.sep)
  if len(path_components) < 2:
    raise ValueError("output_gcs_path should contain at least "
                     "bucket and filename.")
  output_bucket_name = path_components[0]

  # GCS has a flat file structure. Slashes can be used in the filename
  # to simulate nested directories, e.g. 'metrics/model2/20191104.txt'.
  # Combine everything after the bucket name into a single slash-delimited
  # filename.
  output_filename = path.sep.join(path_components[1:])

  storage_client = storage.Client.from_service_account_json(
      service_account_filename)
  bucket = storage_client.get_bucket(output_bucket_name)
  blob = bucket.blob(output_filename)
  blob.upload_from_string(contents)
