# emb-predict

This project contains a framework for building binary classifiers from embeddings to predict relations between two entities of interest.

A first implementation focuses on drug (re)positioning. The classifier is built from drug indications provided by [OpenTargets](https://www.opentargets.org/), which features molecules from [CHEMBL](https://www.ebi.ac.uk/chembl/) and diseases from [MONDO](https://mondo.monarchinitiative.org/), [HPO](https://hpo.jax.org/), and [EFO](https://www.ebi.ac.uk/efo/). The classifier is learned using [Molecular Transformer](https://github.com/mpcrlab/MolecularTransformerEmbeddings) embeddings for molecules and text embeddings from [FASTEMBED](https://qdrant.github.io/fastembed/) and [OLLAMA](https://github.com/ollama/ollama) models.

### Structure:

The framework has 3 components:
* **Prepare**: Prepare 3 CSV files and to store these in the vector store:
    * a (minimum) 2 column table of a and b pairs
    * a (minimum) 3 column table of a, which includes its id, name, and embedding
    * a (minimum) 3 column table of b, which includes its id, name, and embedding
* **Train**: Perform X fold cross validation and to select the best model to store for prediction
* **Predict**: Predict the score for a given pair, or to perform a scan of one set of the pair to all other.


### Services deployed:
* Predict API, for which a user can indicate any set of a or b's to predict
* [Translator Reasoner API](https://github.com/NCATSTranslator/ReasonerAPI)
* [Qdrant](https://qdrant.tech/) vector database to store the computed embeddings for drugs and disease.



# Developer install

## Setup environment
Clone the repository:

```bash
git clone https://github.com/MaastrichtU-IDS/emb-predict
cd emb-predict
```

Update pip
```bash
python3 -m pip install --upgrade pip
```

Install [`hatch`](https://hatch.pypa.io) to manage the project

```bash
pip install --upgrade hatch
```

Use hatch to install the dependencies.

```bash
hatch -v env create
```

## Setup the vectordb
You must have an instance of QdrantDB for this project. You can either install one locally, or have the project point to an existing instance.

### Local deployment:
edit the vectordb/qdrant_config.yml to setup any api-key, and indicate the storage paths.
```bash
cd vectordb
docker-compose up -d
```
### Remote instance
You must edit the config.yml file to indicate the url and api-key to an existing QdrantDB.

```bash
vectordb-url:
vectordb-api-key:
```

### Prepare

Prepare the training data:

```bash
hatch run prepare
```

### Train

Run the training function to train the model:

```bash
hatch run train
```

### Predict

Run the prediction function providing an input ID:

```bash
hatch run predict
```

### Deploy

Run the API, normally at http://localhost:8808

```bash
hatch run deploy
```



### Test

Run the tests locally:

```bash
hatch run test -s
```


## Fast deploy
(under development)

Use the provided docker container to prepare, train, and deploy the best model.

You first need to edit the config.yml file to indicate where the vectordb you want to use (and any API key)

```bash
vectordb-url: http://localhost:6333
vectordb-api-key:
```

Next come and deploy the image

```bash
docker-compose up -d ./run_all.sh

```
