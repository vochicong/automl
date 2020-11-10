apt-get -y update 
apt-get install -y --no-install-recommends wget curl nginx ca-certificates bzip2 build-essential cmake openjdk-8-jdk git-core maven
rm -rf /var/lib/apt/lists/*

curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh >> miniconda.sh
bash ./miniconda.sh -b -p /miniconda; rm ./miniconda.sh;

export PATH="/miniconda/bin:$PATH"
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export GUNICORN_CMD_ARGS="--timeout 60 -k gevent"
cd /opt/mlflow
pip install mlflow==1.11.0
mvn  --batch-mode dependency:copy -Dartifact=org.mlflow:mlflow-scoring:1.11.0:pom -DoutputDirectory=/opt/java
mvn  --batch-mode dependency:copy -Dartifact=org.mlflow:mlflow-scoring:1.11.0:jar -DoutputDirectory=/opt/java/jars
cp /opt/java/mlflow-scoring-1.11.0.pom /opt/java/pom.xml
cd /opt/java && mvn --batch-mode dependency:copy-dependencies -DoutputDirectory=/opt/java/jars
cp model_dir/pyfunc /opt/ml/model
python -c 'from mlflow.models.container import _install_pyfunc_deps;                _install_pyfunc_deps("/opt/ml/model", install_mlflow=True)'