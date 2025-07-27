import ibm_boto3
from ibm_botocore.client import Config, ClientError

class IBMCOSUtils:
    """
    A utility class for interacting with IBM Cloud Object Storage (COS).

    This class handles the connection to IBM COS, allowing for operations like
    uploading, downloading, and listing files in a specified bucket.
    """
    def __init__(self, api_key, service_instance_id, endpoint_url, bucket_name):
        """
        Initializes the IBM COS utility with necessary credentials.

        Args:
            api_key (str): Your IBM Cloud API key.
            service_instance_id (str): The service instance ID of your COS bucket.
            endpoint_url (str): The public endpoint URL for your COS bucket.
            bucket_name (str): The name of your COS bucket.
        """
        self.api_key = api_key
        self.service_instance_id = service_instance_id
        self.endpoint_url = endpoint_url
        self.bucket_name = bucket_name
        self.cos_client = self._create_cos_client()

    def _create_cos_client(self):
        """
        Creates and configures the IBM COS client.

        Returns:
            ibm_boto3.client: An authenticated client for interacting with COS.
        """
        print("Connecting to IBM Cloud Object Storage...")
        try:
            cos_client = ibm_boto3.client(
                "s3",
                ibm_api_key_id=self.api_key,
                ibm_service_instance_id=self.service_instance_id,
                config=Config(signature_version="oauth"),
                endpoint_url=self.endpoint_url
            )
            print("Successfully connected to IBM COS.")
            return cos_client
        except Exception as e:
            print(f"Error connecting to IBM COS: {e}")
            return None

    def upload_file(self, local_file_path, object_key):
        """
        Uploads a file from the local filesystem to the COS bucket.

        Args:
            local_file_path (str): The path to the local file to upload.
            object_key (str): The desired key (name) for the object in COS.
        """
        if not self.cos_client:
            print("COS client not initialized. Cannot upload file.")
            return

        print(f"Uploading {local_file_path} to {self.bucket_name}/{object_key}...")
        try:
            self.cos_client.upload_file(local_file_path, self.bucket_name, object_key)
            print("File uploaded successfully.")
        except ClientError as e:
            print(f"Error uploading file: {e}")

    def download_file(self, object_key, local_file_path):
        """
        Downloads a file from the COS bucket to the local filesystem.

        Args:
            object_key (str): The key of the object to download from COS.
            local_file_path (str): The local path where the file will be saved.
        """
        if not self.cos_client:
            print("COS client not initialized. Cannot download file.")
            return

        print(f"Downloading {self.bucket_name}/{object_key} to {local_file_path}...")
        try:
            self.cos_client.download_file(self.bucket_name, object_key, local_file_path)
            print("File downloaded successfully.")
        except ClientError as e:
            print(f"Error downloading file: {e}")

    def list_files(self):
        """
        Lists all files currently in the COS bucket.

        Returns:
            list: A list of object keys (file names) in the bucket.
        """
        if not self.cos_client:
            print("COS client not initialized. Cannot list files.")
            return []

        print(f"Listing files in bucket: {self.bucket_name}")
        try:
            return [item["Key"] for item in self.cos_client.list_objects(Bucket=self.bucket_name).get("Contents", [])]
        except ClientError as e:
            print(f"Error listing files: {e}")
            return []
