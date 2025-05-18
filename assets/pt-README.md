# 🧠 Diagnóstico de Alzheimer com Machine Learning

![🧠 Diagnóstico de Alzheimer com Machine Learning - Banner](./assets/banner.png)

🔗 **[Acesse o aplicativo Streamlit](https://seulinkstreamlit.com)**

## 📌 Objetivo do Projeto

Este projeto visa **prever o diagnóstico de Alzheimer** com base em dados clínicos, comportamentais e cognitivos de pacientes. Através de uma abordagem de machine learning supervisionado, utilizamos uma **regressão logística implementada em PyTorch** para classificar pacientes quanto à presença ou ausência da doença.

## 🗂️ Fonte dos Dados

- Conjunto de dados: **Alzheimer’s Disease Dataset**
- Origem: Kaggle  
- Link: [https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset)

Este conjunto de dados contém informações abrangentes sobre a saúde de 2.149 pacientes, cada um identificado exclusivamente com IDs que variam de 4.751 a 6.900. O conjunto de dados inclui detalhes demográficos, fatores de estilo de vida, histórico médico, medidas clínicas, avaliações cognitivas e funcionais, sintomas e um diagnóstico de Alzheimer. Dentre as informações, temos:

### 🆔 Patient Identification

| Variable  | Description                                | Range / Values |
| --------- | ------------------------------------------ | -------------- |
| PatientID | Unique identifier assigned to each patient | 4751 to 6900   |

### 👤 Demographic Details

| Variable       | Description                   | Values / Range                                              |
| -------------- | ----------------------------- | ----------------------------------------------------------- |
| Age            | Age of the patient            | 60 to 90 years                                              |
| Gender         | Gender (0 = Male, 1 = Female) | 0, 1                                                        |
| Ethnicity      | Ethnicity of the patient      | 0: Caucasian<br>1: African American<br>2: Asian<br>3: Other |
| EducationLevel | Education level               | 0: None<br>1: High School<br>2: Bachelor's<br>3: Higher     |

### 🧬 Lifestyle Factors

| Variable           | Description                        | Range / Values |
| ------------------ | ---------------------------------- | -------------- |
| BMI                | Body Mass Index                    | 15 to 40       |
| Smoking            | Smoking status (0 = No, 1 = Yes)   | 0, 1           |
| AlcoholConsumption | Weekly alcohol consumption (units) | 0 to 20        |
| PhysicalActivity   | Weekly physical activity (hours)   | 0 to 10        |
| DietQuality        | Diet quality score                 | 0 to 10        |
| SleepQuality       | Sleep quality score                | 4 to 10        |

### 🏥 Medical History

| Variable                | Description                                     | Values |
| ----------------------- | ----------------------------------------------- | ------ |
| FamilyHistoryAlzheimers | Family history of Alzheimer's (0 = No, 1 = Yes) | 0, 1   |
| CardiovascularDisease   | Cardiovascular disease (0 = No, 1 = Yes)        | 0, 1   |
| Diabetes                | Diabetes presence (0 = No, 1 = Yes)             | 0, 1   |
| Depression              | Depression presence (0 = No, 1 = Yes)           | 0, 1   |
| HeadInjury              | History of head injury (0 = No, 1 = Yes)        | 0, 1   |
| Hypertension            | Hypertension presence (0 = No, 1 = Yes)         | 0, 1   |

### 🩺 Clinical Measurements

| Variable                 | Description              | Range (Units)    |
| ------------------------ | ------------------------ | ---------------- |
| SystolicBP               | Systolic blood pressure  | 90 to 180 mmHg   |
| DiastolicBP              | Diastolic blood pressure | 60 to 120 mmHg   |
| CholesterolTotal         | Total cholesterol        | 150 to 300 mg/dL |
| CholesterolLDL           | Low-density lipoprotein  | 50 to 200 mg/dL  |
| CholesterolHDL           | High-density lipoprotein | 20 to 100 mg/dL  |
| CholesterolTriglycerides | Triglycerides level      | 50 to 400 mg/dL  |

### 🧠 Cognitive and Functional Assessments

| Variable             | Description                                             | Range / Values |
| -------------------- | ------------------------------------------------------- | -------------- |
| MMSE                 | Mini-Mental State Examination (lower = more impairment) | 0 to 30        |
| FunctionalAssessment | Functional score (lower = more impairment)              | 0 to 10        |
| MemoryComplaints     | Memory complaints (0 = No, 1 = Yes)                     | 0, 1           |
| BehavioralProblems   | Behavioral problems (0 = No, 1 = Yes)                   | 0, 1           |
| ADL                  | Activities of Daily Living (lower = more impairment)    | 0 to 10        |

### 😕 Symptoms

| Variable                  | Description                                   | Values |
| ------------------------- | --------------------------------------------- | ------ |
| Confusion                 | Confusion presence (0 = No, 1 = Yes)          | 0, 1   |
| Disorientation            | Disorientation presence (0 = No, 1 = Yes)     | 0, 1   |
| PersonalityChanges        | Personality changes (0 = No, 1 = Yes)         | 0, 1   |
| DifficultyCompletingTasks | Difficulty completing tasks (0 = No, 1 = Yes) | 0, 1   |
| Forgetfulness             | Forgetfulness presence (0 = No, 1 = Yes)      | 0, 1   |

### 🧾 Diagnosis Information

| Variable  | Description                             | Values |
| --------- | --------------------------------------- | ------ |
| Diagnosis | Alzheimer's diagnosis (0 = No, 1 = Yes) | 0, 1   |

### 🔒 Confidential Information

| Variable       | Description            | Value       |
| -------------- | ---------------------- | ----------- |
| DoctorInCharge | Confidential doctor ID | "XXXConfid" |

---

![Logistic Regression Coefficients Feature Importance](./assets/logistic-regression-coefficients-feature-importance.png)
## 🔍 Projeto

![Project Pipeline](./assets/project-pipeline.png)

### 1. **Análise Exploratória de Dados (EDA)**

#### 📊 Histogramas das variáveis numéricas
Utilizamos histogramas para:
- Entender a **distribuição** de cada atributo
- Identificar **valores extremos (outliers)**
- Observar possíveis **problemas de escala ou codificação**

![Histograms](./assets/histograms-of-numeric-features.png)

#### 🔗 Matriz de Correlação
Criamos uma **matriz de correlação entre atributos numéricos** para:
- Detectar **relações lineares** entre variáveis
- Identificar possíveis **redundâncias**
- Auxiliar na **engenharia de atributos**

![Correlation Matrix](./assets/correlation-matrix.png)

### 2. 🧼 Preparação e Limpeza dos Dados

| Etapa                         | Descrição                                                                 |
|------------------------------|---------------------------------------------------------------------------|
| Tratamento de nulos           | Remoção de registros ou preenchimento (ex: média para valores contínuos) |
| Conversão de tipos            | Garantia de que colunas estavam no formato adequado (`int`, `float`)     |
| Exclusão de variáveis inúteis | Remoção de `PatientID`, por não contribuir para o modelo                 |
| Codificação de classes        | Conversão da variável `Diagnosis` em `0` e `1`                            |

### 3. ⚙️ Engenharia de Atributos

- **Normalização**: Variáveis numéricas foram normalizadas com `StandardScaler` para facilitar o treinamento.
- **Feature selection**: Através da correlação e da análise de distribuição, mantivemos apenas variáveis **relevantes e não redundantes**.
- **Label Encoding**: A variável `Diagnosis` foi transformada de texto para rótulos binários:
  - `Demented` → `1`
  - `Non-Demented` → `0`

### 🔥 Modelagem com PyTorch - Arquitetura

Utilizamos uma **classe `LogisticRegressionModel(nn.Module)`** com:
- Camada linear de entrada para saída (sem camadas ocultas)
- Sigmoid como função de ativação

```python
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))
```

### 🧪 Treinamento do Modelo

#### 🔧 Configurações

* **Train/test split**: 80% treino, 20% teste
* **Batch size**: 32
* **Épocas**: 100
* **Otimizador**: `torch.optim.Adam`
* **Função de perda (loss)**: `BCELoss()` (Binary Cross Entropy)

> A função de perda avalia a **distância entre a predição do modelo e o rótulo real**, e os **gradientes** são retropropagados para ajustar os pesos da rede.

📌 **Equação da função de perda:**

$$
\mathcal{L}(y, \hat{y}) = -[y \cdot \log(\hat{y}) + (1 - y) \cdot \log(1 - \hat{y})]
$$

## 📈 Avaliação do Modelo

Usamos métricas clássicas de classificação binária para avaliar o modelo:

| Métrica                    |Value| Descrição                                  |
| -------------------------- | ------------------------------------------ |--|
| **Acurácia**               |0.8186| Percentual de previsões corretas           |
| **Precisão**               |0.7434| Correção entre positivos preditos          |
| **Recall** (Sensibilidade) |0.7434| Capacidade de encontrar os positivos reais |
| **F1-Score**               |0.7434| Média harmônica entre precisão e recall    |

## 📌 Resultados Obtidos

![Confusion Matrix](./assets/confusion-matrix.png)

A matriz de confusão revela que o modelo apresenta um bom desempenho, com uma acurácia geral de 81,86%. Ele identificou corretamente 112 pacientes com Alzheimer (verdadeiros positivos) e 239 pacientes sem a doença (verdadeiros negativos).

Entretanto, gerou 40 falsos negativos — casos em que o Alzheimer não foi detectado — o que é clinicamente relevante e potencialmente grave. A precisão (74,34%) e a sensibilidade (73,34%) demonstram um equilíbrio razoável entre a identificação correta de casos positivos e a redução de falsos alarmes. Ainda assim, para diagnósticos mais detalhados, a avaliação de um especialista continua sendo indispensável.

## 💻 Como Executar

1. Clone este repositório:

   ```bash
   git clone https://github.com/ernanej/machine-learning-alzheimer-risk-prediction.git
   cd machine-learning-alzheimer-risk-prediction
   ```

2. Instale as dependências:

   ```bash
   cd streamlit_app
   pip install -r requirements.txt
   ```

3. Execute o Streamlit:

   ```bash
   streamlit run app.py
   ```

> Caso queira gerar novos dados de treinamento basta mudar o caminho do arquivo e executar as células novamente até a geração dos arquivos `features.pkl`, `logistic_model_weights.pth`, `logistic_model.pkl` e `scaler.pkl`, atualiza-los o diretório `/streamlit_app/models` e executar o app novamente.

## 📚 Tecnologias Utilizadas

* Python
* Pandas, NumPy
* PyTorch
* Matplotlib, Seaborn, Plotly
* Streamlit

## 📄 Licença

MIT License. Veja o arquivo `LICENSE` para mais detalhes.