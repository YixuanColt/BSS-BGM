
# BGM: Demand Prediction for Expanding Bike-Sharing Systems with Dynamic Graph Modeling

Official acceptance of our IJCAI 2025 AI and Social Good paper:  
**BGM: Demand Prediction for Expanding Bike-Sharing Systems with Dynamic Graph Modeling**  

---

### 🔍 Overview

The **BGM** framework addresses the key challenge of predicting bike-sharing demand at **newly deployed stations**, where historical data is unavailable. To ensure **equitable mobility access** and **efficient operations** in expanding urban systems, BGM captures **dynamic inter-station relationships** via spatio-temporal graph modeling and enhances prediction accuracy through **embedding transfer** and **feature fusion**.

![Framework Overview](Framework.png)


---

### 🧠 Key Features

- **Dynamic Graph Construction**: Models time-evolving spatial and temporal dependencies between existing and new stations.
- **Knowledge Transfer Module**: Learns an orthogonal mapping from existing to new stations using spatial embedding similarities.
- **Feature Fusion Mechanism**: Integrates transferred embeddings with intrinsic features using a gated vector.

---

### 📊 Experimental Results

Extensive experiments on **NYC Citi-Bike** and **Chicago Divvy-Bike** datasets show that BGM outperforms traditional baselines and recent graph models across all settings, particularly for demand prediction at **new stations**.

| Model           | RMSE (New) ↓ | MAE (New) ↓ |
|----------------|--------------|-------------|
| Linear Regression | 2.36       | 1.98        |
| Function Zone     | 1.91       | 1.36        |
| GraphSAGE         | 2.17       | 1.69        |
| Spatial-MGAT      | 1.83       | 0.99        |
| **BGM (Ours)**     | **1.51**   | **0.85**    |

> ✅ BGM achieves **up to 17.7% lower RMSE** compared to previous state-of-the-art models.


---

### 📁 Datasets

We integrate the following public datasets:

- 🛴 **Bike-sharing trip data** (NYC Citi-Bike, Chicago Divvy-Bike)  
- 🌦️ **Weather data** from NYC Mesowest  
- 📍 **POI data** via Google Place API  
- 🚕 **NYC Taxi trips**  
- 🛣️ **Road network** from NYC Open Data

All datasets are available via public APIs or open portals. See `data/README.md` for processing instructions.

### 📬 Contact

If you have any questions, feel free to reach out:

- Yixuan Zhao: [yz776@exeter.ac.uk](mailto:yz776@exeter.ac.uk)  
