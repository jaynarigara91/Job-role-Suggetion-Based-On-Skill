# Job Recommendation System Based on Skills

This project is a Job Recommendation System that suggests job roles based on user-input skills. Model builed on only 25 job roles with their required skills. It utilizes machine learning techniques and cosine similarity to provide accurate recommendations, helping users find suitable job opportunities.

## 🚀 Project Overview

This Job Recommendation System suggests job roles based on user-input skills, using machine learning and NLP techniques. Data was collected and preprocessed from online sources, then vectorized with CountVectorizer and TF-IDF. Cosine Similarity is used to match user skills with job roles, while TextBlob corrects spelling errors in the input. The system is deployed with a user-friendly Streamlit interface, making it easy for users to find job recommendations tailored to their skills. In project at below example given of job role and required skills also project deployed on docker.

https://github.com/user-attachments/assets/2454266d-555a-4a0c-99f7-56ac2721a5b8

## 🎯 Features

- Input multiple skills to receive job recommendations.
- Spelling correction for user input to enhance accuracy.
- Sorted job recommendations based on similarity scores.

## 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- TextBlob (for spelling correction)
- Pandas
- Pickle (for model and data storage)

## 🐳 Docker Hub Repository

You can find the Docker image for this project on Docker Hub:
[Docker Hub Repository](https://hub.docker.com/repository/docker/jay7556/job_recommend_system/general)

### How to Pull the Image
You can pull the image using the following command:

docker pull jay7556/job_recommend_system

## 🤝 Contributing

Contributions are welcome! Please feel free to open issues or pull requests for improvements.
