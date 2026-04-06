import pandas as pd
import requests, os, time
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urljoin

# --- RUTAS DINÁMICAS ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CSV_PATH = os.path.join(BASE_DIR, 'temasekwreck-temasekblueandwhites.csv')
FOLDER = os.path.join(BASE_DIR, 'data', 'raw')

os.makedirs(FOLDER, exist_ok=True)
# -----------------------

sesion = requests.Session()
reintento = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504], allowed_methods=["HEAD", "GET", "OPTIONS"])

adaptado = HTTPAdapter(max_retries=reintento)
sesion.mount("http://", adaptado)
sesion.mount("https://", adaptado)

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}
sesion.headers.update(headers)

res = pd.read_csv(CSV_PATH)

# Filtro solo las piezas con informacion suficiente
condicionesUtiles = ['Intact', 'Half intact', 'Base', 'Rim to Base', 'Base with stem']
seleccion_estado = res[res['Condition'].isin(condicionesUtiles)]

formasSencillas = ['Dish', 'Dish?', 'Bowl & dish']
seleccionadosFinales = seleccion_estado[seleccion_estado['Description/Shape'].isin(formasSencillas)]

url = "https://epress.nus.edu.sg/sitereports/temasekwreck/images/"

countok = 0
counterror = 0

for index, row in seleccionadosFinales.iterrows():
    aux = row['UIN']
    rowImgString = row['Image']

    if pd.isna(rowImgString):
        continue

    # parseo del csv
    listaAdd = rowImgString.replace('.jpg','.jpg|').replace('.JPG', '.JPG|').split('|')
    lista = [x.strip() for x in listaAdd if x.strip()]

    for nom in lista:
        full_url = url + nom
        ruta_guardado = os.path.join(FOLDER, nom)

        if os.path.exists(ruta_guardado):
            continue

        try:
            response = sesion.get(full_url, timeout=20)
            
            if response.status_code == 200:
                with open(ruta_guardado, 'wb') as handler:
                    handler.write(response.content)
                print(f"   -> {nom}: DESCARGADA")
                countok += 1
            elif response.status_code == 404:
                print(f"   -> {nom}: 404 No encontrada. Probando variante...")
                counterror += 1
            else:
                print(f"   -> {nom}: Error {response.status_code}")
                counterror += 1

        except Exception as e:
            print(f"   -> Error conexión: {e}")
            counterror += 1

    time.sleep(0.2)

print(f"\n--- RESUMEN ---")
print(f"Descargadas: {countok}")
print(f"Errores/No encontradas: {counterror}")