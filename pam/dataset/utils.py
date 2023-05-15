from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, ContainerClient, generate_account_sas, ResourceTypes, \
    AccountSasPermissions

ACCOUNT_URL = "https://chorus.blob.core.windows.net" #Account Azure path
KEY = "" #Pase key account here

#Prefixes for variables
PREFIX_ANNOTATIONS_BOUNDING_BOX = 'INCT.Annotations/bounding_boxes/'
PREFIX_ANNOTATIONS_PRESENCE_ABSENCE = 'INCT.Annotations/presence_absence/'
PREFIX_ENVIRONMENTAL_VARIABLES_PLANETARYCOMPUTER = 'INCT.EnvironmentalVariables/PlanetaryComputer/'
PREFIX_ENVIRONMENTAL_VARIABLES_WEATHERSTATIONS = 'INCT.EnvironmentalVariables/WeatherStations/'
PREFIX_DATALOGGERS = {'INCT20955': 'INCT.selvino/1_Locais/INCT20995/dataloggers/'}
PREFIX_RECORDS = {'INCT20955': 'INCT.selvino/1_Locais/INCT20995/gravador/',
                  'INCT4': 'INCT.ftoledo/1_Locais/INCT04/gravador/',
                  'INCTTEST': 'INCT.test/1_Locais/INCT.test/gravador/'}

CONTAINER_NAME_PUBLIC = 'public'
CONTAINER_NAME_BACKUP = 'backup'

def get_container_client(account_name, account_key, container_name):
    """
    Create an Azure container client

    @param account_name: Name of Storage Account
    @param account_key: Key access to the Storage Account
    @param container_name: Name of the container

    @return: A instance of a container client

    """
    try:
        sas_token = generate_account_sas(
            account_name=account_name,
            account_key=account_key,
            resource_types=ResourceTypes(service=True),
            permission=AccountSasPermissions(read=True, list=True),
            expiry=datetime.utcnow() + timedelta(hours=2)
        )
        container_client = ContainerClient(account_url=ACCOUNT_URL,
                                           container_name=container_name,
                                           account_key=sas_token)
        return container_client
    except Exception as ex:
        print('Exception: ' + ex.__str__())
        raise ex


def get_blob_client(account_key):
    """
    Create an Azure blob client

    @param account_key: Account key in Azure

    @return: A instance of a blob client

    """
    blob_service_client = BlobServiceClient(account_url=ACCOUNT_URL, credential=account_key)

    return blob_service_client
