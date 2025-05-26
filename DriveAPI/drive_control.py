from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import re
import os

"""
This script was created to effectively resolve an issue in which
the dataset was uploaded directly to Drive and the images could not
be located in one file to be erased, so it had to be done through the API
"""

SCOPES = ['https://www.googleapis.com/auth/drive']

def auth_drive():

    creds = None
    if os.path.exists('code/credentials.json'):
        creds = Credentials.from_authorized_user_file("code/credentials.json",SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("code/client_secret.apps.googleusercontent.com.json",SCOPES)
            creds = flow.run_local_server(port=39000)
        with open("code/credentials.json","w") as token:
            token.write(creds.to_json())
    
    return creds

def del_jpg_files(service):
    query = "modifiedTime < '2021-03-27T12:00:00' and (mimeType contains 'image/jpeg')"
    results = service.files().list(pageSize=1000,fields="nextPageToken, files(id, name)",q=query).execute()
    items = results.get('files',[])

    if not items:
        print('No files found.')
    else:
        print('Files to be deleted:')
        i=0
        for item in items:
            if re.match(r'\d+\.jpg$',item['name']):
                print(i)
                service.files().delete(fileId=item['id']).execute()
                i+=1
    service.files().emptyTrash().execute()
    print("Files deleted and trash can emptied.")
    
def main():
    creds = auth_drive()
    service = build('drive','v3',credentials=creds,developerKey="DEVELOPER_KEY")
    del_jpg_files(service)

if __name__ ==  '__main__':
    main()