import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from textblob import TextBlob

data = pickle.load(open('job.pkl','rb'))
vector= pickle.load(open('vector.pkl','rb'))
model= pickle.load(open('model.pkl','rb'))

st.header("Job Recommand System")

user_skill = st.text_input('Enter Your Programing Skills: ')

def correct_spelling(skill):
    blob = TextBlob(skill)
    return str(blob.correct())

def recommend(user_skill):
    user_skilled = [correct_spelling(skill.strip(" ")) for skill in user_skill.split(',')]
    user_skill = " ".join(user_skilled)
    user_vector = model.transform([user_skill])
    simil = cosine_similarity(user_vector,vector).flatten()
    data['similarity'] = simil
    jobs = sorted(list(enumerate(data['similarity'])),reverse=True,key=lambda x:x[1])[0:3]
    
    recommended_jobs = []
    for i in jobs:
        job_role = data.iloc[i[0]]['Job Role']
        expected_salary = data.iloc[i[0]]['Expected Salary (INR)']
        recommended_jobs.append({'Job Role': job_role, 'Expected Salary (INR)': expected_salary})

    return pd.DataFrame(recommended_jobs)

if st.button('Recommend'):
    if user_skill:
        recommendation_df = recommend(user_skill)
        if not recommendation_df.empty:
            st.write("Top 3 Job Recommendations Based On Skills:")
            st.dataframe(recommendation_df)  
        else:
            st.write("No recommendations found. Please try different skills.")
    else:
        st.write("Please enter your skills to get recommendations.")
        

example_data = pd.DataFrame({
    "Job Role": [
        "Data Scientist",
        "Software Engineer",
        "Data Analyst",
        "Machine Learning Engineer",
        "Web Developer",
        "DevOps Engineer",
        "Cloud Engineer",
        "Business Analyst",
        "Cybersecurity Analyst",
        "UX/UI Designer",
        "Product Manager",
        "Mobile App Developer",
        "Network Engineer",
        "Digital Marketing Manager",
        "Financial Analyst",
        "AI Engineer",
        "Full Stack Developer",
        "Backend Developer",
        "Frontend Developer",
        "QA Engineer",
        "Big Data Engineer",
        "Database Administrator",
        "Systems Engineer",
        "Blockchain Developer",
        "Embedded Systems Engineer",
        "IT Support Specialist",
        "Content Strategist",
        "Game Developer",
        "Operations Analyst",
        "HR Analyst",
    ],
    "Required Skills": [
        "Python, R, SQL, Machine Learning, Data Analysis, Statistics, NLP, Data Visualization",
        "Java, Python, C++, JavaScript, Git, Algorithms, Data Structures, OOP",
        "Excel, SQL, Python, Data Cleaning, Data Visualization, Tableau, Statistics",
        "Python, TensorFlow, PyTorch, Deep Learning, Machine Learning, Data Preprocessing, SQL",
        "HTML, CSS, JavaScript, React, Node.js, Git, Responsive Design",
        "Docker, Kubernetes, Jenkins, CI/CD, AWS, Linux, Python, Bash",
        "AWS, Azure, Google Cloud, Docker, Kubernetes, Networking, Security, Terraform",
        "SQL, Excel, Data Analysis, Project Management, Business Strategy, Power BI, Communication",
        "Network Security, Firewalls, Ethical Hacking, Python, Risk Analysis, SIEM, Incident Response",
        "Figma, Adobe XD, Sketch, Wireframing, Prototyping, User Research, CSS",
        "Project Management, Agile, Communication, Market Research, Product Roadmaps, UX/UI Understanding",
        "Swift, Kotlin, Java, Android SDK, iOS SDK, UI/UX Design, RESTful APIs",
        "TCP/IP, Cisco, Routing, Switching, Network Security, Firewall Configuration, VPN",
        "SEO, SEM, Google Analytics, Content Creation, Social Media, Marketing Strategy, PPC",
        "Excel, Financial Modeling, Forecasting, SQL, Data Analysis, Market Research, Financial Reporting",
        "Python, TensorFlow, PyTorch, Machine Learning, Deep Learning, NLP, Computer Vision",
        "JavaScript, HTML, CSS, React, Node.js, SQL, MongoDB, Git, REST APIs",
        "Python, Java, Node.js, SQL, Databases, REST APIs, Microservices, Git",
        "HTML, CSS, JavaScript, React, Angular, Responsive Design, CSS Preprocessing",
        "Selenium, Java, Python, Manual Testing, Automated Testing, JIRA, Test Automation Frameworks",
        "Hadoop, Spark, SQL, Python, Java, Data Warehousing, ETL, Data Modeling",
        "SQL, MySQL, Oracle, Database Management, Data Backup, Performance Tuning, Security",
        "Linux, Windows Server, Networking, Scripting, Virtualization, Troubleshooting, Systems Architecture",
        "Solidity, Ethereum, Smart Contracts, Python, Cryptography, dApps, Blockchain Protocols",
        "C, C++, Embedded C, Microcontrollers, RTOS, Firmware Development, Debugging",
        "Troubleshooting, Windows, Linux, Networking, Customer Service, Ticketing Systems, Hardware Configuration",
        "SEO, Content Writing, Copywriting, Social Media, Content Marketing, Analytics, Research",
        "C++, Unity, Unreal Engine, Game Design, 3D Modeling, Animation, Scripting",
        "Excel, SQL, Data Analysis, Process Improvement, Project Management, Business Analysis",
        "Excel, SQL, Data Analysis, HR Metrics, Communication, Organizational Behavior, Reporting",
    ]
})

# Display the example job role and skills in a table
st.subheader("Example Job Role")
st.dataframe(example_data)

st.subheader("Description")
st.write("You can add any skill you want i just give you UP example of skills and input multiple skills with comma ',' (example : python,sql,machine learning). don't worry if you write incorrect spelling it will correct automatically while prediction ")
