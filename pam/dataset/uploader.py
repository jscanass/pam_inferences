import os
from dataset.utils import *


def upload_blob_to_container(file_path):
    """
    Upload a blob file to an Azure container

    @param file_path: File path to upload
    @return: None
    """
    try:
        service_client = get_blob_client(account_key=KEY) #Get Azure blob client
        blob_client = service_client.get_blob_client(container=CONTAINER_NAME_PUBLIC, blob='inferences/' + os.path.basename(file_path))

        with open(file_path, "rb") as data: #Opening de file to upload
            blob_client.upload_blob(data, overwrite=True)
            print(f"Uploaded {os.path.basename(file_path)}.")
    except Exception as e:
        print(e)
        # TO DO: Manage the exception
        pass

#Example
#upload_blob_to_container('/home/jrudascas/Descargas/Febrero.pdf')
