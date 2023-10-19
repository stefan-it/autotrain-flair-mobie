import os

from experiment import ExperimentConfiguration, run_experiment

from huggingface_hub import login, HfApi

from mobie_dataset import NER_GERMAN_MOBIE

corpus = NER_GERMAN_MOBIE()

# See MobIE paper, table 2
ref_sentences = 7_077
ref_tokens = 90_971

actual_sentences = sum([1 for sentence in corpus.train + corpus.dev + corpus.test if sentence[0].text != "-DOCSTART-"])
actual_tokens = sum([len(sentence) for sentence in corpus.train + corpus.dev + corpus.test if sentence[0].text != "-DOCSTART-"])

assert ref_sentences == actual_sentences, "Number of parsed sentences does not match with reported number of sentences!"
assert ref_tokens == actual_tokens, "Number of parsed tokens does not match with reported number of tokens!"

# Hyper-Parameter search definitions
batch_sizes = [16]
learning_rates = [3e-05, 5e-05]
seeds = [1, 2, 3, 4, 5]
epochs = [10]
context_sizes = [64]

# Backbone LM definitions
base_model = "deepset/gbert-base"
base_model_short = "gbert_base"

# Hugging Face Model Hub configuration
hf_token = os.environ.get("HF_TOKEN")
hf_hub_org_name = os.environ.get("HUB_ORG_NAME")

login(token=hf_token, add_to_git_credential=True)
api = HfApi()

# Loop around hyper-parameter search:
for seed in seeds:
    for batch_size in batch_sizes:
        for epoch in epochs:
            for learning_rate in learning_rates:
                for context_size in context_sizes:
                    experiment_configuration = ExperimentConfiguration(
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        epoch=epoch,
                        context_size=context_size,
                        seed=seed,
                        base_model=base_model,
                        base_model_short=base_model_short,
                    )
                    output_path = run_experiment(experiment_configuration=experiment_configuration)

                    repo_url = api.create_repo(
                        repo_id=f"{hf_hub_org_name}/{output_path}",
                        token=hf_token,
                        private=True,
                        exist_ok=True,
                    )

                    if experiment_configuration.use_tensorboard:
                        api.upload_folder(
                            folder_path=f"{output_path}/runs",
                            path_in_repo="./runs",
                            repo_id=f"{hf_hub_org_name}/{output_path}",
                            repo_type="model"
                        )

                    api.upload_file(
                        path_or_fileobj=f"{output_path}/best-model.pt",
                        path_in_repo="./pytorch_model.bin",
                        repo_id=f"{hf_hub_org_name}/{output_path}",
                        repo_type="model"
                    )
                    api.upload_file(
                        path_or_fileobj=f"{output_path}/training.log",
                        path_in_repo="./training.log",
                        repo_id=f"{hf_hub_org_name}/{output_path}",
                        repo_type="model"
                    )
