# Despliegue de SIEMIA

La aplicación se despliega en dos servicios gratuitos:

| Componente | Plataforma | Cómo se publica |
|---|---|---|
| Backend (Flask + PyTorch + SQLite) | **Render** | Lee el repositorio (rama `mari`) |
| Frontend (Flutter web) | **GitHub Pages** | GitHub Actions lo compila y publica solo |

El orden importa: **primero Render**, porque su URL se incrusta en el frontend al
compilarlo.

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

## Paso 2 — Frontend en GitHub Pages

No hay que compilar a mano: el archivo `.github/workflows/deploy-frontend.yml`
hace que GitHub compile la app y la publique en cada push.

### 2.1 Activar Pages (una sola vez)

1. En GitHub, entra al repositorio → **Settings** → **Pages**.
2. En **Source**, elige **GitHub Actions** (no "Deploy from a branch").
3. Guarda.

### 2.2 Lanzar el despliegue

Con el workflow ya en el repositorio, basta con un push que toque `frontend/`.
Para lanzarlo sin cambiar nada: **Actions** → *Publicar frontend en GitHub Pages*
→ **Run workflow**.

Tarda unos 3–5 minutos. Al terminar, la app queda en:

```
https://firnex360.github.io/Arrythmia-detection-categorization-by-machine-learning/
```

Ábrela e inicia sesión con **admin / admin**.

### Si quieres compilar en local

Sólo hace falta para probar antes de subir:

```bash
cd frontend
flutter build web \
  --base-href /Arrythmia-detection-categorization-by-machine-learning/ \
  --dart-define=API_URL=https://siemia-api.onrender.com
```

> **En Windows con Git Bash**, antepón `MSYS_NO_PATHCONV=1` al comando. Si no,
> Git Bash convierte `/Arrythmia-.../` en una ruta de Windows
> (`C:/Program Files/Git/...`) y el `base href` queda mal, dejando la página en
> blanco. En PowerShell o en el runner de GitHub (Linux) no ocurre.

---

## Actualizar la aplicación

Ambas partes se actualizan solas al subir cambios:

```bash
git push origin mari
```

- **Backend**: Render detecta el push y redespliega.
- **Frontend**: GitHub Actions recompila y republica (sólo si cambió `frontend/`).

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
