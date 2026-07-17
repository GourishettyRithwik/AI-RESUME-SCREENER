import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# External skills knowledge (beyond dataset - my expertise)
CATEGORY_SKILLS = {
    'Data Science': ['python', 'r', 'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'sql', 'statistics', 'data analysis', 'big data', 'hadoop', 'spark'],
    'HR': ['recruitment', 'talent acquisition', 'employee relations', 'performance management', 'training', 'payroll', 'compensation', 'labor law', 'succession planning', 'hr analytics'],
    'Business Analyst': ['requirements gathering', 'data analysis', 'sql', 'excel', 'tableau', 'power bi', 'agile', 'jira', 'user stories', 'process improvement', 'stakeholder management'],
    'Java Developer': ['java', 'spring boot', 'hibernate', 'microservices', 'rest api', 'maven', 'jenkins', 'docker', 'kubernetes', 'sql', 'nosql'],
    # ... (full 24 categories - abbreviated for brevity)
    'Testing': ['selenium', 'junit', 'testng', 'cypress', 'jmeter', 'postman', 'automation', 'manual testing', 'agile', 'defect tracking'],
    'DevOps Engineer': ['docker', 'kubernetes', 'jenkins', 'ansible', 'terraform', 'aws', 'azure', 'ci cd', 'monitoring', 'prometheus', 'grafana'],
    'Python Developer': ['python', 'django', 'flask', 'fastapi', 'pandas', 'numpy', 'sqlalchemy', 'celery', 'docker', 'aws'],
    'Blockchain': ['solidity', 'ethereum', 'hyperledger', 'smart contracts', 'web3', 'truffle', 'ganache', 'dapp', 'ipfs'],
    'C++': ['c++', 'stl', 'boost', 'multithreading', 'opencv', 'qt', 'linux', 'performance optimization'],
    'Hadoop': ['hadoop', 'spark', 'hive', 'hbase', 'sqoop', 'flume', 'kafka', 'mapreduce', 'yarn', 'pig']
    # Add remaining categories as needed
}

print("Loading dataset...")
df = pd.read_csv('resume_dataset.csv', encoding='utf-8')

def clean_resume(text):
    import re
    text = re.sub('http\\S+\\s*', ' ', text)
    text = re.sub('RT|cc', ' ', text)
    text = re.sub('#\\S+', '', text)
    text = re.sub('@\\S+', ' ', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\\x00-\\x7f]', r' ', text)
    text = re.sub('\\s+', ' ', text)
    return text.lower().strip()

print("Cleaning data...")
df['cleaned_resume'] = df['Resume'].apply(clean_resume)

le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['Category'])

X = df['cleaned_resume'].values
y = df['category_encoded'].values

print("Creating TF-IDF...")
vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english', max_features=1500)
X_tfidf = vectorizer.fit_transform(X)

print("Training ensemble model...")
knn = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5))
nb = OneVsRestClassifier(MultinomialNB(alpha=1.0))
ensemble = VotingClassifier(estimators=[('knn', knn), ('nb', nb)], voting='soft')
ensemble.fit(X_tfidf, y)

print("Model accuracies:")
print(f"Train acc: {ensemble.score(X_tfidf, y):.3f}")

# Save everything
joblib.dump(ensemble, 'model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(le, 'label_encoder.joblib')
joblib.dump(CATEGORY_SKILLS, 'category_skills.pkl')

print("[OK] Saved: model.joblib, vectorizer.joblib, label_encoder.joblib, category_skills.pkl")
print("Run: python models.py to train!")
