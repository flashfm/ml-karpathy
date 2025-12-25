import requests, json
from pathlib import Path

def get_file(filename, url, is_json = False):
  filename = Path.home() / ".ml" / filename
  if not filename.exists():
      filename.parent.mkdir(parents=True, exist_ok=True)
      response = requests.get(url)
      with open(filename, 'wb') as f:
          f.write(response.content)

  with open(filename, 'r', encoding='utf-8') as f:
    if (is_json):
      result = json.load(f)
    else:
      result = f.read()

  return result
