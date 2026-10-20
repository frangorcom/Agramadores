# AgroCebada 2026 — Dashboard web

Dashboard multipágina creado con Streamlit. El proyecto está preparado para ejecutarse localmente o publicarse en Streamlit Community Cloud. Se retiraron los emojis del contenido visible y se mantuvieron los ajustes de contraste.

## Ejecutar localmente

1. Instala Python 3.10 o posterior.
2. En una terminal, entra en la carpeta del proyecto.
3. Instala las dependencias: `pip install -r requirements.txt`.
4. Ejecuta desde la raíz del proyecto: `streamlit run Inicio.py`.

## Publicar como página web

1. Descomprime el ZIP y sube el contenido de esta carpeta a un repositorio de GitHub. Incluye `Inicio.py`, `pages/`, `data/`, `.streamlit/config.toml` y `requirements.txt`.
2. Abre https://share.streamlit.io/ e inicia sesión con GitHub.
3. Selecciona **Create app** y elige el repositorio, la rama `main` y el archivo principal `Inicio.py`.
4. Pulsa **Deploy**. Streamlit generará una dirección pública terminada en `streamlit.app`.

No se puede crear la dirección pública desde este paquete sin acceso a tu cuenta de GitHub/Streamlit; esos pasos requieren que inicies sesión y autorices la publicación.

## Datos

Los archivos CSV están en `data/`. El mapa usa predicciones simuladas cuando no existe `data/resultados_modelo.csv`; el archivo `data/importancia_features.csv` también es opcional.
