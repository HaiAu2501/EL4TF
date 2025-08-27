# 🌟 EL4TF: A Study on Ensemble Learning in Time-Series Forecasting 📊

## News

- **03/2025:** Project initiated.
- **04/2025:** Initial results obtained.
- **05/2025:** Report published.
- **06/2025:** Phase 1 completed.
- **08/2025:** Phase 2 initiated.

📌 Stay tuned for upcoming updates! 🚀

## Introduction

⏳ Time series forecasting is a crucial task in various fields such as stock market prediction, weather forecasting, and sales forecasting. Traditional forecasting methods often rely on a single model to predict future values, which may not capture the complexity of the underlying data.

🔍 In this study, we explore the effectiveness of **ensemble learning** in time series forecasting, leveraging different publicly available datasets.

---

## Contributors

We are grateful to the dedicated individuals who have contributed to the success of this project.

**Phase 1 Contributors:**  
The following members played a pivotal role in completing the first phase:

- Nguyễn Viết Tuấn Kiệt
- Bùi Quang Phong
- Nguyễn Thái Hòa
- Lưu Thịnh Khang
- Nguyễn Thanh Tuyển

**Phase 2 Contributors:**  
As we move into the second phase, we extend our appreciation to those who are helping us enhance our models and explore new datasets:

- Nguyễn Đức An
- Nguyễn Trọng Tâm
- Phạm Quang Nguyên Hoàng
- Trần Ngọc Lâm
- Đào Chí Hiển

---

## Experiments

### Dataset

**All datasets used in this study are publicly available on Kaggle and stored in the `data/` folder.**

<div align="center">

| **Name**                | **Period**  | **Frequency** | **Sources**                                                                                       | **Folder**                                             | **Task**       | **Type**    |
| ----------------------- | ----------- | ------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------ | -------------- | ----------- |
| **3 Stocks & Bitcoin**  | 2013 - 2019 | Daily         | [Kaggle](https://www.kaggle.com/datasets/hershyandrew/amzn-dpz-btc-ntfx-adjusted-may-2013may2019) | [`data/stock`](data/stock)                             | Regression     | Multi-task  |
| **Tesla Stock Price**   | 2017 - 2017 | Daily         | [Kaggle](https://www.kaggle.com/datasets/rpaguirre/tesla-stock-price)                             | [`data/tesla`](data/tesla)                             | Regression     | Multi-task  |
| **Daily Delhi Climate** | 2013 - 2017 | Daily         | [Kaggle](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data)              | [`data/delhi`](data/delhi)                             | Regression     | Multi-task  |
| **Weather Prediction**  | 2012 - 2015 | Daily         | [Kaggle](https://www.kaggle.com/datasets/ananthr1/weather-prediction)                             | [`data/rainy`](data/rainy/)                            | Classification | Binary      |
| **VN30**                | 2019 - 2025 | Daily         | Real-world                                                                                        | [`regression`](data/vn30/regression/)                  | Regression     | Multi-task  |
| **VN30**                | 2019 - 2025 | Daily         | Real-world                                                                                        | [`multi_class`](data/vn30/multi_class_classification/) | Classification | Multi-class |
| **VN30**                | 2019 - 2025 | Daily         | Real-world                                                                                        | [`multi_task`](data/vn30/multi_task_classification/)   | Classification | Multi-task  |
| **VN30**                | 2019 - 2025 | Daily         | Real-word                                                                                         | [`multi_label`](data/vn30/multi_label_classification/) | Classification | Multi-label |

</div>

### Methodology

🔬 The study implements various **ensemble learning models** to analyze their impact on forecasting accuracy.

We aim to enhance the performance of individual models by combining their predictions using ensemble techniques. To achieve this, we implement several advanced methods to improve the robustness and accuracy of our forecasts, including:

- **Metric Learning in K-Nearest Neighbors (KNN):** Optimizing distance metrics to improve neighbor selection and prediction accuracy.
- **Forest Building via Diversity-Oriented Subsampling and Posterior Estimation:** Leveraging diverse subsets of data to construct more generalized and accurate ensemble models.
- **Feature Engineering and Selection Techniques Based on Distributional Properties:** Identifying and selecting features that capture the underlying data distribution effectively.
- **Semi-Supervised Learning as an Intermediate Approach:** Utilizing both labeled and unlabeled data to enhance model training and prediction performance.

---

## Quick Start

1. Clone the repository:

```bash
git clone https://github.com/HaiAu2501/EL4TF.git
cd EL4TF
```

2. Create a virtual environment:

```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Results

📊 The final results demonstrate how ensemble learning improves forecasting performance across multiple datasets.

---

## Author & Contact
