# Heavy Metal Heatmap Generator

A browser-based Streamlit application that generates publication-ready A4 concentration heatmaps from CSV data of geographic sampling points and heavy-metal concentrations.

Upload a CSV file, configure map settings, and download individual or bulk heatmaps as PNG and PDF files — no programming required.

## Features

- **IDW interpolation** (Inverse Distance Weighting) on a projected coordinate grid (UTM/AzEq)
- **Per-metal A4 maps** at 300 DPI with colour bar, scale bar, north arrow, graticule, and metadata notes
- **Combined N-panel publication map**: automatically generates a multi-panel composite figure ((a), (b), (c)...) containing all analysed metals, with odd last panels aligned in the horizontal center
- **PNG and PDF exports** plus a complete ZIP archive (including individual maps and combined multi-panel maps)
- **Robust CSV parsing**: auto-detects encoding (UTF-8, Windows-1252), delimiters (comma, semicolon, tab), and decimal separators
- **Data validation**: coordinate checking, non-detect handling, duplicate averaging, per-metal quality reports
- **Leave-one-out cross-validation** RMSE reported per metal
- **Configurable**: colour ramp, scale, resolution, IDW power, surface extent, labels, and more

## CSV Format

Your CSV needs at minimum columns for longitude, latitude, and one or more metal/analyte concentration columns:

| Sampling Sites | Longitudes | Latitudes | Pb | Cd | Cr | Cu |
|---|---:|---:|---:|---:|---:|---:|
| Site 1 | 90.412 | 23.810 | 12.5 | 0.8 | 25.4 | 18.2 |
| Site 2 | 90.425 | 23.796 | 15.2 | 1.1 | 21.7 | 16.9 |
| Site 3 | 90.438 | 23.785 | 11.8 | 0.6 | 28.1 | 20.5 |

**Column names** are matched case-insensitively. Recognised aliases:
- **Site**: `Sampling Sites`, `Site`, `Sample`, `Station`, `ID`, `Name`
- **Longitude**: `Longitudes`, `Longitude`, `Lon`, `Long`, `Lng`
- **Latitude**: `Latitudes`, `Latitude`, `Lat`

Metal columns can include units in parentheses, e.g. `Pb (mg/kg)`.

Coordinates must be **WGS84 decimal degrees**.

## Controls

| Setting | Default | Description |
|---|---|---|
| Map Presentation Style | GIS Journal / Reference Style (Publication) | **GIS Journal / Reference Style**: 9-class discrete contours, 8-point compass rose, bracket scale bar, 4-sided DMS ticks, boxed legend, grey hull outline. **Continuous Surface**: smooth gradient with separate color bar panel. |
| Panel Letter | (a) | Panel identifier for GIS journal layout, e.g. (a), (b), (c) |
| Colour ramp | gis_yellow_brown (GIS Style) / viridis | GIS 9-class yellow-to-brown, or Matplotlib: viridis, plasma, inferno, magma, cividis; diverging: coolwarm, RdYlBu, Spectral |
| Reverse ramp | off | Invert the colour direction |
| Colour scale | linear | linear, log (values must be >0), percentile clip (2nd–98th) |
| Grid resolution | 300 | 100–800 cells along the longer side |
| IDW power | 2.0 | 1.0–6.0, step 0.5; higher values give sharper transitions |
| Surface extent | Interpolated area only | Or full buffered extent (with caveat) |
| Opacity | 1.0 | 0.3–1.0 |
| Sampling points | Shown, size 40 | Show/hide, size 10–200, optional colour by value |
| Site labels | Off | Max 40 sites; offset from marker, truncated to 18 characters |
| Orientation | Landscape | Landscape (3508×2480 px) or portrait (2480×3508 px) |
| Title prefix | blank | Free text, max 80 characters |
| Source note | blank | Free text, max 120 characters |
| Unit per metal | From header | Free text, max 20 characters |

## Local Setup (Windows)

```powershell
# 1. Clone or download the project, then open a terminal in the project folder

# 2. Create a virtual environment
py -m venv .venv

# 3. Activate it
.venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the app
streamlit run app.py

# The app opens at http://localhost:8501
```

### Running tests

```powershell
# With the venv activated:
pip install -r requirements-dev.txt
pytest tests/ -v
```

## Deployment to Streamlit Community Cloud

1. **Create a GitHub repository** for this project:
   ```powershell
   git init
   git add .
   git commit -m "Initial commit: Heavy Metal Heatmap Generator"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

2. **Sign in** to [Streamlit Community Cloud](https://share.streamlit.io/) with your GitHub account.

3. **Create a new app**:
   - Select your repository and branch (`main`)
   - Set `app.py` as the main file path
   - Choose a Python version (3.10 or later recommended)
   - Click **Deploy**

4. **Receive your public URL** (e.g. `https://your-app.streamlit.app`).

### Embedding into your GitHub / Vercel Portfolio

Vercel is built for static frontend sites and serverless functions, whereas Streamlit requires a persistent Python runtime with active WebSockets for interactive rendering.

The industry-standard, seamless way to display this app directly inside your Vercel portfolio:

1. **Deploy the Streamlit app** to [Streamlit Community Cloud](https://share.streamlit.io/) (free, auto-updates on git push).
2. **Embed it into your portfolio page** using an `<iframe>`:

```html
<!-- Inside your Next.js / React / HTML portfolio on Vercel -->
<div style={{ width: '100%', height: '900px', borderRadius: '12px', overflow: 'hidden', border: '1px solid #e2e8f0' }}>
  <iframe
    src="https://your-app-name.streamlit.app/?embed=true"
    width="100%"
    height="100%"
    frameBorder="0"
    title="Heavy Metal Heatmap Generator"
  />
</div>
```

3. **Or add a Project Card** with a high-resolution export preview, live app link (`https://your-app-name.streamlit.app`), and GitHub source link.

### Alternatives

- **Hugging Face Spaces**: Create a new Space with the Streamlit SDK, push your code, and receive a public URL.
- **Render**: Create a new Web Service, connect your GitHub repo, set the start command to `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`, and deploy.

### Deployment limitations

- **Memory and CPU**: Free-tier hosts impose resource limits; large datasets or high resolutions may exceed them. Check your host's current documentation for specific figures.
- **Idle sleeping**: Community Cloud apps sleep after a period of inactivity and take a moment to restart.
- **Upload cap**: The app limits uploads to 5 MB (configurable in `.streamlit/config.toml`).
- **No built-in rate limiting**: The app does not rate-limit requests; rely on the host's infrastructure.
- **Session data**: All data lives in the browser session and is lost on restart or timeout.
- **Not yet deployed**: This app is not deployed or public until you complete the steps above and confirm the deployment.

## Limitations

- The interpolated surface is an **IDW estimate**, not measured data. It ignores geology, pollution sources, and measurement uncertainty.
- The surface is weakest where samples are sparse or absent.
- Projected CRS is UTM for data within one 6° zone; an azimuthal equidistant projection (approximate distances) is used otherwise.
- Validate results against field data before regulatory or health decisions.

## License

This project is provided as-is for research and educational use.
