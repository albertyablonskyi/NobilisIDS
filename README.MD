# NobilisIDS

**NobilisIDS** is a project that focuses on building an Intrusion Detection System (IDS) using machine learning algorithms. This IDS is trained on network traffic data to classify network activities as either normal or malicious.

## Dependencies

This project depends on the following external project:

- **[NobilisIDS CICFlowMeter (cicflowmeter-py)](https://github.com/albertyablonskyi/cicflowmeter-py.git)**: This project is a fork of the Python CICFlowMeter, customized to suit the needs of the NobilisIDS project. It includes changes to the output format of the .csv file and a workaround for dropping packets other than IP TCP/UDP.

Please follow the installation instructions provided in the **NobilisIDS CICFlowMeter** repository to set it up as a dependency for **NobilisIDS**.

## Getting Started

### Prerequisites
- Python (3.6 or higher)
- pip package manager

### Installation
1. Clone the **NobilisIDS** repository:
   ```sh
   git clone https://github.com/albertyablonskyi/NobilisIDS.git
   cd NobilisIDS
   ```
2. Install the required Python packages:
   ```sh
   pip install -r requirements.txt
   ```
3. Clone the NobilisIDS CICFlowMeter repository:
   ```sh
   git clone https://github.com/albertyablonskyi/cicflowmeter-py.git
   ```
4. Follow the installation instructions in the **NobilisIDS CICFlowMeter** repository to set it up.

## Usage

### Training the model (train.py)

The **train.py** script is used to train the IDS classifier model. It follows the steps below:

- Define the dataset and model paths:
   ```sh
   csv_file_path = 'datasets/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
   model_file_path = 'models/ddos.pkl'
   ```
- To make predictions, run the following command:
   ```sh
   python train.py
   ```

### Making Predictions (predict.py)

The **predict.py** script is used to make predictions using the trained model. It follows the steps below:

- Define the dataset and model paths:
   ```sh
   csv_file_path = 'datasets/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
   model_file_path = 'models/ddos.pkl'
   ```
- To make predictions, run the following command:
   ```sh
   python predict.py
   ```

### Datasets

The following datasets are used for training and prediction:

   - **datasets/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv**

   - **datasets/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv**

   - **datasets/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv**

   - **datasets/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv**

Please ensure that the datasets are placed in the appropriate directory before running the scripts.

### Models

The following pre-trained models are used for prediction:

   - **models/ddos.pkl**
   - **models/portscan.pkl**
   - **models/infiltration.pkl**
   - **models/webattacks.pkl**

## CICIDS2017 Dataset

The CICIDS2017 dataset is used in this project for training and evaluation purposes. It consists of labeled network flows, including full packet payloads in pcap format, along with corresponding profiles and labeled flows. The dataset is publicly available for researchers.

If you are using the CICIDS2017 dataset in your work, it is important to cite the following paper, which provides detailed information about the dataset and its underlying principles:

Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization", 4th International Conference on Information Systems Security and Privacy (ICISSP), Portugal, January 2018.

Please refer to the paper for a comprehensive understanding of the dataset and to comply with the required citation guidelines.
