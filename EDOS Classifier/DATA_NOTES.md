# Data notes

This package already includes the files needed to run the pipeline:

```text
data/edos/raw/edos_labelled_aggregated.csv
data/edos/raw/variations_augmentation_gpt4o_five_classes.csv
data/conan/WOMAN-Multitarget-CONAN.json
```

`variations_augmentation_gpt4o_five_classes.csv` is the definition-based augmentation file from the public Khan et al. ACL 2025 repository.

`WOMAN-Multitarget-CONAN.json` is the dataset to annotate. The EDOS labels produced for it are predicted labels, not gold annotations.
