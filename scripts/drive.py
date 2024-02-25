from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
import io
import json

def search_files(folder_id, service):
    results = service.files().list(
        q=f"'{folder_id}' in parents",
        fields="nextPageToken, files(id, name, mimeType)").execute()
    items = results.get('files', [])

    jsons = []

    if not items:
        print(f'No se encontraron archivos en {folder_id}')
        return jsons


    print(f"Files inside the folder , {items}")

    for item in items:
        # Asume que todos los archivos en la carpeta son videos
        file_id = item['id']
        file_name = item['name']
        
        if item['mimeType'] == 'application/vnd.google-apps.folder':
            jsons += search_files(file_id, service)
        
        if item['mimeType'] == "application/json":
            fh = io.BytesIO()
            request = service.files().get_media(fileId=file_id)
            downloader = MediaIoBaseDownload(fh, request, 1024*1024*5)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                print(f"Descargando {file_name} {int(status.progress() * 100)}%.")
            fh.seek(0)
            dataset_json = json.loads(fh.read())
            jsons.append(dataset_json)

        if item['mimeType'] == "video/mp4":
            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
        
            # Descarga el archivo
            downloader = MediaIoBaseDownload(fh, request, 1024*1024*5)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                print(f"Descargando {file_name} {int(status.progress() * 100)}%.")
        
            # Escribe el contenido del video en un archivo
            with open(f'data/{file_name}', 'wb') as f:
                f.write(fh.getbuffer())
    return jsons

def download_files(folder_id):
    
    credentials = service_account.Credentials.from_service_account_file(
            'credentials.json', scopes=['https://www.googleapis.com/auth/drive']
    )
    service = build("drive", "v3", credentials=credentials)

    jsons = search_files(folder_id, service)

    dataset = {}
    for j in jsons:
        dataset.update(j)

    
    print(json.dumps(dataset))
