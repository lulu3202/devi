## Curated Portfolio: Data Science, Machine Learning, Computer Vision, and NLP Projects

---

### Face Mask Detection Using Computer Vision and Keras 

This project builds a model to detect whether individuals are wearing masks using Keras and computer vision techniques. A folder-based dataset with masked and unmasked faces is processed by leveraging transfer learning from MobileNetV2 model (a popular CNN used for vision tasks). The model is trained by cropping faces, converting them into numerical formats, and making predictions. The best-performing model is saved as an .h5 file for deployment. 

<img src="images/with_facemask.png?raw=true" /> <img src="images/without_facemask.png?raw=true" />

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-ffffff?logo=matplotlib) ![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy)

[View code on GitHub](https://github.com/lulu3202/Deep_Learning)

---

### NASA NEO Dataset: Predicting Hazardous Near-Earth Objects with Machine Learning

This  project utilizes the NASA NEO dataset to predict whether near-Earth objects are hazardous. With 338,199 observations and 9 features, I applied supervised learning techniques, preprocessing the data to optimize inputs. Random Forest emerged as the best-performing algorithm, achieving an accuracy of 91% in classifying hazardous objects.

<img src="images/nasa_neo.png?raw=true" width="300" />


![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy) ![Matplotlib](https://img.shields.io/badge/Matplotlib-ffffff?logo=matplotlib) ![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas)

[View code on GitHub](https://github.com/lulu3202/ML-DS-Capstone-Project/tree/main)

---

### Airflow ETL Pipeline with Postgres and NASA API Integration

This project creates an ETL pipeline using Apache Airflow to extract data from NASA's Astronomy Picture of the Day (APOD) API. The extracted data is transformed and loaded into a PostgreSQL database for further analysis. Airflow orchestrates the workflow, managing task dependencies and scheduling. Docker is used to run Airflow and Postgres in isolated containers. The pipeline automates the daily extraction, transformation, and loading of astronomy-related data.

<img src="images/nasa.jpg?raw=true" width="300" />

![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white) ![Apache Airflow](https://img.shields.io/badge/Airflow-017CEE?logo=apache-airflow&logoColor=white) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-336791?logo=postgresql&logoColor=white) ![Astronomer](https://img.shields.io/badge/Astronomer-00A3E0?logo=astronomer&logoColor=white)

[View code on GitHub](https://github.com/lulu3202/etl)

---

### Photo Critique App

This project develops a Photo Critique App using Streamlit and Google's Gemini-1.5-Flash-8B model. It involves setting up a virtual environment, securely storing API keys, and configuring the app to interact with the Gemini model for generating critiques based on uploaded images. The Streamlit interface allows users to run and test different versions of the app seamlessly.

<img src="images/critique_app.jpg?raw=true" width="300" />

![Google Generative AI](https://img.shields.io/badge/Google%20Generative%20AI-4285F4?logo=google&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white) ![VS Code](https://img.shields.io/badge/VS%20Code-007ACC?logo=visual-studio-code&logoColor=white)

[View code on GitHub](https://github.com/lulu3202/photo_critique_app1)  |  [Watch my YouTube tutorial](https://www.youtube.com/watch?si=h0Bqhnu2vOidbIZY&v=tUB6nulmk3s&feature=youtu.be)

---

### Animated Movie Poster Design App

This project utilizes AWS Lambda and the Stability AI Stable Diffusion model from AWS Bedrock to generate animated movie posters based on user-provided text prompts. The generated images are stored in an S3 bucket, and a pre-signed URL is returned for easy access. The implementation includes setting up an API Gateway for user interaction and testing with Postman for seamless functionality.

<img src="images/unicorn.png?raw=true" width="300" />
Generated image for the prompt "Unicorn in the style of Dr.Seuss"

![Boto3](https://img.shields.io/badge/Boto3-FF9900?logo=amazonaws&logoColor=white) ![AWS](https://img.shields.io/badge/Amazon%20Web%20Services-232F3E?logo=amazonaws&logoColor=white) ![Amazon Bedrock](https://img.shields.io/badge/Amazon%20Bedrock-FF9900?logo=amazonaws&logoColor=white)

[View code on GitHub](https://github.com/lulu3202/Image_Generation_with_Bedrock/blob/main/README.md)

---

### Fuji X-S20 Camera Q&A Application

This application provides an interactive Q&A platform for users to ask questions about the Fuji X-S20 camera, leveraging the power of Gemini Flash API and Retrieval Augmented Generation (RAG) to deliver real-time, concise and accurate answers.

<img src="images/pdf_r.png?raw=true" />

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![Gradio](https://img.shields.io/badge/Gradio-00B4D8?logo=gradio&logoColor=white) ![LangChain](https://img.shields.io/badge/LangChain-00A3E0?logo=python&logoColor=white) ![Google Generative AI](https://img.shields.io/badge/Google%20Generative%20AI-4285F4?logo=google&logoColor=white) ![Conda](https://img.shields.io/badge/Conda-40A9E0?logo=anaconda&logoColor=white)

[View code on GitHub](https://github.com/lulu3202/PDF_RAG_Reader)  |  [View Blog on Medium](https://medium.com/@devipriyakaruppiah/building-a-fuji-x-s20-camera-q-a-app-with-gemini-langchain-and-gradio-befc8d620721)

---

### End To End Data Science Project Deployment with Flask, Docker and AWS EC2

This project is an end-to-end machine learning application that predicts student performance based on various input features. It utilizes Flask for the web interface and Scikit-Learn for model development, along with a complete CI/CD workflow using GitHub Actions, Docker, and AWS ECR. The application encompasses data ingestion, transformation, and a prediction pipeline, resulting in an effective model for forecasting student grades.

<img src="images/e2e.png?raw=true" width="300" />

![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white) ![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white) ![Scikit Learn](https://img.shields.io/badge/Scikit%20Learn-F7931E?logo=scikit-learn&logoColor=white) ![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![AWS](https://img.shields.io/badge/Amazon%20Web%20Services-232F3E?logo=amazonaws&logoColor=white) ![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)

[View code on GitHub](https://github.com/lulu3202/e2emlproject)

---

### Mobile Price Classification using AWS SageMaker

This project leverages AWS SageMaker to classify mobile price ranges using a dataset with multiple features like battery power, RAM, and more. The data is split for training and testing, and a Random Forest model is trained to predict price categories. Finally, the model is deployed via an endpoint for real-time inference, showcasing end-to-end machine learning, all performed with AWS SageMaker.

<img src="images/sagemaker.png?raw=true" width="300" />

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white) ![AWS SageMaker](https://img.shields.io/badge/AWS_SageMaker-FF9900?logo=amazon-aws&logoColor=white) ![Boto3](https://img.shields.io/badge/Boto3-569A31?logo=amazon-aws&logoColor=white)

[View code on GitHub](https://github.com/lulu3202/awssagemaker1)

---
## Cloud Certifications 
My active cloud certifications include:

- [Microsoft Certified: Azure AI Engineer Associate](https://learn.microsoft.com/en-us/users/devi-6391/credentials/bd28630d2b036a1a?ref=https%3A%2F%2Fwww.linkedin.com%2F)
- [Microsoft Certified: Azure Solutions Architect Expert](https://learn.microsoft.com/en-us/users/devi-6391/credentials/7aacac48819cc637?ref=https%3A%2F%2Fwww.linkedin.com%2F)
- [Microsoft Certified: Azure Administrator Associate](https://learn.microsoft.com/en-us/users/devi-6391/credentials/6e72329de036849d?ref=https%3A%2F%2Fwww.linkedin.com%2F)
- [AWS Certified Solutions Architect – Associate](https://www.credly.com/badges/fe5d9495-2ca7-4f0e-b376-3379ed63b025/linked_in_profile)
- [Google Certified Cloud Digital Leader](https://www.credential.net/48420cc4-5689-4376-a7e0-a21429b939df#gs.0me98g)
