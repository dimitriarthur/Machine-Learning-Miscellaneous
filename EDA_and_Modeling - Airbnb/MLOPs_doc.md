# Useful MLOps Open-Source Tools

## List of the main MLOps Tools
These are the main MLOPs Tools and their ML application fields



## AutoML

*Tools for performing AutoML.*

* [NNI](https://github.com/microsoft/nni) - An open source AutoML toolkit for automate machine learning lifecycle.

## Data Management

*Tools for performing data management.*
* [Milvus](https://github.com/milvus-io/milvus/) - An open source embedding vector similarity search engine powered by Faiss, NMSLIB and Annoy.

## Feature Store

*Feature store tools for data serving.*

* [Butterfree](https://github.com/quintoandar/butterfree) - A tool for building feature stores. Transform your raw data into beautiful features.
* [ByteHub](https://github.com/bytehub-ai/bytehub) - An easy-to-use feature store. Optimized for time-series data.
* [Feast](https://feast.dev/) - End-to-end open source feature store for machine learning.

## Hyperparameter Tuning

*Tools and libraries to perform hyperparameter tuning.*

* [Hyperas](https://github.com/maxpumperla/hyperas) - A very simple wrapper for convenient hyperparameter optimization.
* [Hyperopt](https://github.com/hyperopt/hyperopt) - Distributed Asynchronous Hyperparameter Optimization in Python.
* [Katib](https://github.com/kubeflow/katib) - Kubernetes-based system for hyperparameter tuning and neural architecture search.
* [Optuna](https://optuna.org/) - Open source hyperparameter optimization framework to automate hyperparameter search.
* [Scikit Optimize](https://github.com/scikit-optimize/scikit-optimize) - Simple and efficient library to minimize expensive and noisy black-box functions.
* [Talos](https://github.com/autonomio/talos) - Hyperparameter Optimization for TensorFlow, Keras and PyTorch.
* [Tune](https://docs.ray.io/en/latest/tune.html) - Python library for experiment execution and hyperparameter tuning at any scale.


## Machine Learning Platform

*Complete machine learning platform solutions.*

* [DAGsHub](https://dagshub.com/) - A platform built on open source tools for data, model and pipeline management.
* [H2O](https://www.h2o.ai/) - Open source leader in AI with a mission to democratize AI for everyone.
* [Hopsworks](https://www.hopsworks.ai/) - Open-source platform for developing and operating machine learning models at scale.
* [Iguazio](https://www.iguazio.com/) - Data science platform that automates MLOps with end-to-end machine learning pipelines.
* [Knime](https://www.knime.com/) - Create and productionize data science using one easy and intuitive environment.
* [Kubeflow](https://www.kubeflow.org/) - Making deployments of ML workflows on Kubernetes simple, portable and scalable.
* [LynxKite](https://lynxkite.com/) - A complete graph data science platform for very large graphs and other datasets.
* [ML Workspace](https://github.com/ml-tooling/ml-workspace) - All-in-one web-based IDE specialized for machine learning and data science.
* [Modzy](https://www.modzy.com/) - AI platform and marketplace offering scalable, secure, and ready-to-deploy AI models.
* [Neu.ro](https://neu.ro) - MLOps platform that integrates open-source and proprietary tools into client-oriented systems.
* [Pachyderm](https://www.pachyderm.com/) - Combines data lineage with end-to-end pipelines on Kubernetes, engineered for the enterprise.
* [Polyaxon](https://www.github.com/polyaxon/polyaxon/) - A platform for reproducible and scalable machine learning and deep learning on kubernetes.
* [Sagemaker](https://aws.amazon.com/sagemaker/) - Fully managed service that provides the ability to build, train, and deploy ML models quickly.
* [Valohai](https://valohai.com/) - Takes you from POC to production while managing the whole model lifecycle.
* 
## Model Lifecycle

*Tools for managing model lifecycle (tracking experiments, parameters and metrics).*

* [Guild AI](https://guild.ai/) - Open source experiment tracking, pipeline automation, and hyperparameter tuning.
* [Mlflow](https://mlflow.org/) - Open source platform for the machine learning lifecycle.
* [ModelDB](https://github.com/VertaAI/modeldb/) - Open source ML model versioning, metadata, and experiment management.


## Optimization Tools

*Optimization tools related to model scalability in production.*

* [Dask](https://dask.org/) - Provides advanced parallelism for analytics, enabling performance at scale for the tools you love.
* [DeepSpeed](https://github.com/microsoft/DeepSpeed) - Deep learning optimization library that makes distributed training easy, efficient, and effective.
* [Fiber](https://uber.github.io/fiber/) - Python distributed computing library for modern computer clusters.
* [Horovod](https://github.com/horovod/horovod) - Distributed deep learning training framework for TensorFlow, Keras, PyTorch, and Apache MXNet.
* [Mahout](https://mahout.apache.org/) - Distributed linear algebra framework and mathematically expressive Scala DSL.
* [MLlib](https://spark.apache.org/mllib/) - Apache Spark's scalable machine learning library.
* [Modin](https://github.com/modin-project/modin) - Speed up your Pandas workflows by changing a single line of code.
* [Petastorm](https://github.com/uber/petastorm) - Enables single machine or distributed training and evaluation of deep learning models.
* [Rapids](https://rapids.ai/index.html) - Gives the ability to execute end-to-end data science and analytics pipelines entirely on GPUs.
* [Ray](https://github.com/ray-project/ray) - Fast and simple framework for building and running distributed applications.
* [Singa](http://singa.apache.org/en/index.html) - Apache top level project, focusing on distributed training of DL and ML models.
* [Tpot](https://github.com/EpistasisLab/tpot) - Automated ML tool that optimizes machine learning pipelines using genetic programming.

# Detailing MLOPs Tools
## Kubeflow
Kubeflow and its components allow you to manage almost every aspect of your ML experiments.

### What is KubeFlow?
The Kubeflow project is dedicated to making deployments of machine learning (ML) workflows on Kubernetes *simple, portable and scalable*. The main goal is not to recreate other services, but to provide a straightforward way to deploy best-of-breed open-source systems for ML to diverse infrastructures. Anywhere you are running Kubernetes, you should be able to run Kubeflow.

![image](https://user-images.githubusercontent.com/50504364/125621287-6b136449-47fd-4c61-9bc0-24a0396a1910.png)


### Features
It provides a wide range of possibilities that make model deployment easier. 
* **Notebooks**: Kubeflow includes services to create and manage interactive Jupyter notebooks. You can customize your notebook deployment and your compute resources to suit your data science needs. Experiment with your workflows locally, then deploy them to a cloud when you're ready.
* **TensorFlow Model Training**: Kubeflow provides a custom TensorFlow training job operator that you can use to train your ML model. In particular, Kubeflow's job operator can handle distributed TensorFlow training jobs. Configure the training controller to use CPUs or GPUs and to suit various cluster sizes.
* **Model Serving**: Kubeflow supports a TensorFlow Serving container to export trained TensorFlow models to Kubernetes. Kubeflow is also integrated with Seldon Core, an open source platform for deploying machine learning models on Kubernetes, and NVIDIA Triton Inference Server for maximized GPU utilization when deploying ML/DL models at scale.
* **Pipelines**: Kubeflow Pipelines is a comprehensive solution for deploying and managing end-to-end ML workflows. Use Kubeflow Pipelines for rapid and reliable experimentation. You can schedule and compare runs, and examine detailed reports on each run.
* **Multi-framework**: Our development plans extend beyond TensorFlow. We're working hard to extend the support of PyTorch, Apache MXNet, MPI, XGBoost, Chainer, and more. We also integrate with Istio and Ambassador for ingress, Nuclio as a fast multi-purpose serverless framework, and Pachyderm for managing your data science pipelines.


### Central Dashboard 
Overview of the Kubeflow user interfaces (UIs).

Your Kubeflow deployment includes a central dashboard that provides quick access to the Kubeflow components deployed in your cluster. The dashboard includes the following features:
* Shortcuts to specific actions, a list of recent pipelines and notebooks, and metrics, giving you an overview of your jobs and cluster in one view.
* A housing for the UIs of the components running in the cluster, including **Pipelines, Katib, Notebooks**, and more.
* A registration flow that prompts new users to set up their namespace if necessary.

#### Overview of Kubeflow UIs 

![image](https://user-images.githubusercontent.com/50504364/125623335-d2f5cfbc-88c5-4b87-b1a9-17e8c98cf3d1.png)


The Kubeflow UIs include the following:

* **Home**, a central dashboard for navigation between the Kubeflow components.
* **Pipelines** for a Kubeflow Pipelines dashboard.
* **Notebook** Servers for Jupyter notebooks.
* **Katib** for hyperparameter tuning.
* **Artifact Store** for tracking of artifact metadata.
* **Manage Contributors** for sharing user access across namespaces in the Kubeflow deployment.


#### Example: Using Kubeflow for Financial Time Series

In this example, there is an exploration of training and serving of a machine learning model by leveraging Kubeflow's main components.  The use case is a [Machine Learning with Financial Time Series Data](https://cloud.google.com/architecture/machine-learning-with-financial-time-series-data) problem. For the detailed step by step solution, check out this [GitHub repository](https://github.com/kubeflow/examples/tree/master/financial_time_series).

## DAGsHub
Free Dataset & Model Hosting with Zero Configuration.

![image](https://user-images.githubusercontent.com/50504364/125634284-7f8fde7d-acf5-4920-b04c-67e49d6d1514.png)


### What is DAGsHub
DAGsHub is a web platform for data version control and collaboration for data scientists and machine learning engineers. The platform is built on DVC, an open-source version control system for machine learning projects that works seamlessly with Git.

With DAGsHub Storage, sharing data and models becomes as easy as sharing a link, offering collaborators an easy overview of project data, models, code, experiments, and pipelines.

All of this provides a better collaborative experience for data science teams and will hopefully aid in massive development and acceptance of Open Source Data Science (OSDS) projects.

### Using DVC + DAGsHub Storage for Version Control and Remote Collaboration

DAGsHub Storage is an alternative (and free-to-use) DVC remote that requires zero configurations. It is a new tool from the makers of DAGsHub, a web platform for data version control and collaboration for data scientists and machine learning engineers (DAGsHub is to DVC what Github is to Git).

With DAGsHub storage, you don’t have to go through the stress of setting up anything. It works the same way as adding a Git remote.

When you create a repository on DagsHub, it automatically provides you with a DVC remote URL. With this URL, you can quickly push and pull data using your existing DAGsHub credentials (via HTTPS basic authentication).

This also means that sharing work with non-DVC users is much easier, as there is no cloud setup required on their end. Isn’t that so much better?

To connect DAGsHub Storage as your remote storage, you need to create an account on DAGsHub and create a project. You can do this either by creating one from scratch or connecting to an existing project on another platform like Github or Bitbucket, and setting up DVC for local data versioning. The figure illustrates the creating of a new repository on DAGsHub.

![image](https://user-images.githubusercontent.com/50504364/125634781-372af554-1629-4c01-9858-cb4cc2b15083.png)


When you create a repo on DAGsHub, you get two remotes: Git and DVC.


![image](https://user-images.githubusercontent.com/50504364/125635010-9548f274-c893-4bf9-9c43-290b3b2d3b7d.png)

To get started using DAGsHub storage, copy the DVC link (which can be found on your repo’s homepage) and add it as a remote for your local project.

Open your project in a terminal and add the DVC remote:
```
dvc remote add <--dvc remote link-->
```
Next thing is to set up DAGsHub credentials for your local machine, just the way you would for GitHub:

```
dvc remote modify origin --local auth basic
dvc remote modify origin --local user Linda-Ikechukwu
dvc remote modify origin --local ask_password true
```

You can now either `push` or `pull` datasets and models seamlessly with dvc `push -r origin` or  `dvc pull -r origin`.

If you want to switch to different versions of your data, just like you do git checkout, all you have to do is run:

```
git checkout <..branch or commit..>
dvc checkout
```

### Examples of Usage
* [Defining the Pipeline](https://dagshub.com/docs/tutorial/pipeline/)
* [Experimentation and Reproducibility](https://dagshub.com/docs/tutorial/experiment_repro/)


## KNIME
KNIME its a end to end platform thats allow you to build datascience applications.

### What is Knime?
The Knime project is a platform focused on building data science processing flows.

### Features
A variety of tools for flow-based data processing and control are offered. KNIME makes understanding data and designing data science workflows and reusable components accessible.

**Create**

* **Gather & Wrangle** Access, merge and transform all of your data
* **Model & Visualize** Make sense of your data with the tools you choose

**Productionize**

* **Deploy & Manage** Support enterprise wide data science practices
* **Consume & Optimize** Leverage insights gained from your data

### Knime User Interface
Overview of the Knime user interfaces (UIs).

Knime Software includes three main modules that provides a quick access to the Knime tools. The platform includes the following modules:
* Server, a enterprise solution for data science.
* Analytics Platform, its an open source solution for data science.
* Extensions, provide additional functionalities such as access to and processing of complex data types, as well as the addition of advanced machine learning algorithms. Extensions can come from the community, partners and by the Knime extensios.

#### Overview of Knime UIs 

![image](https://www.knime.com/sites/default/files/styles/content_width_with_sidebar/public/media/images/KNIME-Analytics-Platform-Panels.png)


The Knime UIs include the following:

* **KNIME explorer**
* **Workflow editor**
* **Workflow coach**
* **KNIME Hub**
* **Description**
* **Node monitor**
* **Node repository**

#### Example: Using Knime for Calculating a Annual Finantial Metric

In this example, there is an exploration of training and serving of a machine learning model by leveraging Knime's main components.  
In this use case  [Calculating Annually Recurring Revenue](https://hub.knime.com/knime/spaces/Finance,%20Accounting,%20and%20Audit/latest/Calculating%20Annually%20Recurring%20Revenue~kGLPHMcsksS-2l39) problem.
