import os
from dataset.utils import *


def upload_blob_to_container(file_path):
    try:
        service_client = get_blob_client(account_key=KEY)
        blob_client = service_client.get_blob_client(container=CONTAINER_NAME_PUBLIC, blob='inferences/' + os.path.basename(file_path))

        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
            print(f"Uploaded {os.path.basename(file_path)}.")
    except Exception as e:
        print(e)
        # TO DO: Manage the exception
        pass


#upload_blob_to_container('/home/jrudascas/Descargas/Febrero.pdf')
