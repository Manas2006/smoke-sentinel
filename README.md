# SmokeSentinel: Wildfire Smoke Exposure Nowcaster

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Manas2006/smoke-sentinel/blob/main/notebooks/SmokeSentinel_Wildfire_Nowcast.ipynb)

SmokeSentinel is a machine learning system that predicts wildfire smoke exposure using multi-source data and graph-aware spatiotemporal models. The system combines NASA FIRMS hotspot data, NOAA HRRR-Smoke forecasts, and ground-based sensor measurements to provide accurate PM₂.₅ predictions.

## Features

- **Multi-source Data Integration**: Combines satellite, weather model, and ground sensor data
- **Graph-aware Models**: Uses ConvLSTM and GATv2 for spatiotemporal prediction
- **Interactive Explorer**: Visualize smoke plumes and get location-specific forecasts
- **Explainable AI**: SHAP analysis for model interpretability

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/smoke-sentinel.git
   cd smokesentinel
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate smokesentinel
   ```

3. Create a `.env` file with your API keys:
   ```bash
   PURPLEAIR_API_KEY=your_key_here
   ```

4. Download sample data:
   ```bash
   python -m scripts.collect_data --days 2
   ```

5. Build the tract graph:
   ```bash
   python -m scripts.build_graph
   ```

6. Train models:
   ```bash
   python -m scripts.train_models --model convlstm
   python -m scripts.train_models --model gat
   ```

## Data Pipeline

The data collection script downloads data from multiple sources and organizes it by date:

```bash
# Download last 3 days of data
python -m scripts.collect_data --days 3 --out-dir data/raw --log-level INFO
```

This creates the following directory structure:
```
data/raw/
├── firms/
│   ├── FIRMS_2024-01-01.tif
│   ├── FIRMS_2024-01-02.tif
│   └── FIRMS_2024-01-03.tif
├── hrrr/
│   ├── HRRR_20240101.grib2
│   ├── HRRR_20240102.grib2
│   └── HRRR_20240103.grib2
├── purpleair/
│   ├── PurpleAir_2024-01-01.json
│   ├── PurpleAir_2024-01-02.json
│   └── PurpleAir_2024-01-03.json
└── openaq/
    ├── OpenAQ_2024-01-01.json
    ├── OpenAQ_2024-01-02.json
    └── OpenAQ_2024-01-03.json
```

### Data Sources

- **NASA FIRMS**: Fire hotspot data from VIIRS and MODIS satellites
- **NOAA HRRR-Smoke**: High-resolution weather and smoke forecasts
- **PurpleAir/OpenAQ**: Ground-based PM₂.₅ measurements

## Models

1. **ConvLSTM**: Processes rasterized smoke data with convolutional LSTM layers
2. **GATv2**: Graph attention network for census tract-level predictions

## Development

Run tests:
```bash
pytest tests/
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Quickstart

### Run in Google Colab

Click the badge above → press **Runtime ▸ Run all**.  
The first cell clones the repo and installs just the extra packages, so runtime setup stays under 2 minutes.  If you only want a demo dataset, execute:

```python
!python scripts/download_sample.py
``` 