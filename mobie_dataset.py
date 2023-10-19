import flair

from flair.datasets.sequence_labeling import ColumnCorpus
from flair.file_utils import cached_path

from pathlib import Path
from typing import Optional, Union


class NER_GERMAN_MOBIE(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)
        dataset_name = self.__class__.__name__.lower()
        data_folder = base_path / dataset_name
        data_path = flair.cache_root / "datasets" / dataset_name

        columns = {0: "text", 3: "ner"}

        train_data_file = data_path / "train.conll2003"
        if not train_data_file.is_file():
            temp_file = cached_path(
                "https://github.com/DFKI-NLP/MobIE/raw/master/v1_20210811/ner_conll03_formatted.zip",
                Path("datasets") / dataset_name,
            )
            from zipfile import ZipFile

            with ZipFile(temp_file, "r") as zip_file:
                zip_file.extractall(path=data_path)

        super().__init__(
            data_folder,
            columns,
            in_memory=in_memory,
            comment_symbol=None,
            document_separator_token="-DOCSTART-",
            **corpusargs,
        )
