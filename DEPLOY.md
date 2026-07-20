# Despliegue de SIEMIA

La aplicación se despliega en dos servicios gratuitos:

| Componente | Plataforma | Qué se sube |
|---|---|---|
| Backend (Flask + PyTorch + SQLite) | **Render** | Todo el repositorio |
| Frontend (Flutter web) | **Vercel** | La carpeta `frontend/build/web` |

El orden importa: **primero Render**, porque necesitas su URL para compilar el frontend.

---

## Paso 0 — Subir los cambios a GitHub

```bash
git add .
git commit -m "Preparar despliegue en Render y Vercel"
git push origin mari
```

> La base de datos `ecg_app.db` ya **no** se sube (se añadió a `.gitignore`), porque
> contiene datos de pacientes.

---

## Paso 1 — Backend en Render

1. Entra a [render.com](https://render.com) y crea una cuenta (puedes usar GitHub).
2. **New +** → **Blueprint**.
3. Conecta el repositorio `Arrythmia-detection-categorization-by-machine-learning`.
4. Render detecta el archivo `render.yaml` y propone el servicio `siemia-api`.
   Confirma con **Apply**.

   > El `render.yaml` fija `branch: mari`, así que Render despliega **esa rama**,
   > no `main`. Si más adelante fusionas a `main`, cambia esa línea.
5. Espera el primer despliegue (**8–15 minutos**: descarga PyTorch e instala todo).
6. Al terminar tendrás una URL como:

   ```
   https://siemia-api.onrender.com
   ```

### Comprobar que funciona

Abre en el navegador `https://TU-URL.onrender.com/health`. Debe responder:

```json
{ "status": "ok", "model": "ecg_arrhythmia_model_v5_mita.pt", "device": "cpu", ... }
```

Si ves eso, el backend está en línea y el modelo cargó correctamente.

---

## Paso 2 — Frontend en Vercel

### 2.1 Compilar apuntando a Render

Sustituye la URL por la tuya:

```bash
cd frontend
flutter build web --dart-define=API_URL=https://siemia-api.onrender.com
```

La URL queda incrustada en el build; no hay que configurar nada dentro de la app.

### 2.2 Publicar

```bash
npm install -g vercel      # sólo la primera vez
cd build/web
vercel --prod
```

Vercel hará unas preguntas (nombre del proyecto, etc.). Acepta los valores por
defecto: detecta que es un sitio estático automáticamente.

Al terminar te da una URL como `https://siemia.vercel.app`. Ábrela e inicia sesión
con **admin / admin**.

> Alternativa sin instalar nada: entra a [vercel.com/new](https://vercel.com/new) y
> arrastra la carpeta `frontend/build/web` a la página.

---

## Actualizar la aplicación

- **Backend**: `git push origin mari` → Render redespliega solo (observa esa rama).
- **Frontend**: hay que recompilar y volver a publicar:
  ```bash
  cd frontend
  flutter build web --dart-define=API_URL=https://siemia-api.onrender.com
  cd build/web && vercel --prod
  ```

---

## Limitaciones del plan gratuito (importante para la defensa)

**1. El backend se duerme.** Render suspende el servicio tras 15 minutos sin uso.
La primera petición después tarda **30–60 segundos** en responder, porque tiene que
levantar el proceso y cargar PyTorch más el modelo en memoria.

> **Antes de la defensa: abre la aplicación 3 minutos antes** para despertar el
> backend. Después responde con normalidad.

**2. La base de datos es efímera.** Render gratuito no incluye disco persistente:
`ecg_app.db` se reinicia con cada redespliegue. Los pacientes y ECG que cargues se
pierden, pero la cuenta demo `admin / admin` se vuelve a crear sola al arrancar.

> Para la demostración, carga los pacientes de ejemplo el mismo día. Si necesitas
> persistencia real, la solución es migrar de SQLite a PostgreSQL (Render ofrece
> uno gratuito por 90 días), lo que implica reescribir `db.py`.

**3. Memoria ajustada.** El plan gratuito da 512 MB de RAM y PyTorch consume unos
300 MB. Por eso `render.yaml` fija `--workers 1`: con dos workers el servicio se
quedaría sin memoria.

---

## Nota sobre el historial de Git

`ecg_app.db` ya no se sube, pero **sigue presente en los commits anteriores** del
repositorio. Si esa base contenía datos reales de pacientes (no de prueba), habría
que reescribir el historial con `git filter-repo` o BFG Repo-Cleaner para
eliminarla del pasado. Si sólo eran datos de prueba, no hace falta.
